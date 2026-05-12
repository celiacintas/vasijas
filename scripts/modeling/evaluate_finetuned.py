import argparse
import json
import torch
import numpy as np
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import AutoencoderKL, UNet2DConditionModel, UNet2DModel, DDPMScheduler
from peft import PeftModel
from torchmetrics.image.fid import FrechetInceptionDistance
from torchvision.transforms import ToTensor, ToPILImage
import sys 
import random
sys.path.insert(0, str(Path(__file__).parent))
from ceramic_dataset import create_train_test_splits
from generate_from_finetuned import generate_images

def load_finetuned_models(checkpoint_dir, device):
    checkpoint_path = Path(checkpoint_dir)
    model_index = checkpoint_path / "model_index.json"

    is_uncond = (
        model_index.exists()
        and json.loads(model_index.read_text()).get("_class_name") == "DDPMPipeline"
    )

    if is_uncond:
        noise_scheduler = DDPMScheduler.from_pretrained(str(checkpoint_path / "scheduler"))
        unet = UNet2DModel.from_pretrained(
            str(checkpoint_path / "unet"),
            torch_dtype=torch.float16 if device.type == "cuda" else torch.float32
        ).to(device).eval()
        return {
            "noise_scheduler": noise_scheduler,
            "unet": unet,
            "device": device,
        }

    tokenizer = CLIPTokenizer.from_pretrained(str(checkpoint_path / "tokenizer"))
    text_encoder = CLIPTextModel.from_pretrained(
        str(checkpoint_path / "text_encoder"),
        torch_dtype=torch.float16 if device.type == "cuda" else torch.float32
    ).to(device).eval()
    vae = AutoencoderKL.from_pretrained(
        str(checkpoint_path / "vae"),
        torch_dtype=torch.float16 if device.type == "cuda" else torch.float32
    ).to(device).eval()
    noise_scheduler = DDPMScheduler.from_pretrained(str(checkpoint_path / "scheduler"))

    unet_path = checkpoint_path / "unet_lora"
    if unet_path.exists():
        base_unet = UNet2DConditionModel.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            subfolder="unet",
            torch_dtype=torch.float16 if device.type == "cuda" else torch.float32
        )
        unet = PeftModel.from_pretrained(base_unet, str(unet_path))
    else:
        unet = UNet2DConditionModel.from_pretrained(
            str(checkpoint_path / "unet"),
            torch_dtype=torch.float16 if device.type == "cuda" else torch.float32
        )
    unet = unet.to(device).eval()

    return {
        "tokenizer": tokenizer,
        "text_encoder": text_encoder,
        "vae": vae,
        "unet": unet,
        "noise_scheduler": noise_scheduler,
        "device": device,
    }


def generate_unconditional_images(models, num_images=9, num_inference_steps=100, seed=None):
    unet = models["unet"]
    noise_scheduler = models["noise_scheduler"]
    device = models["device"]
    dtype = unet.dtype

    if seed is not None:
        torch.manual_seed(seed)

    noise_scheduler.set_timesteps(num_inference_steps)
    image_size = unet.config.sample_size

    images = []
    for i in range(num_images):
        noise = torch.randn((1, unet.config.in_channels, image_size, image_size), device=device, dtype=dtype)
        latents = noise

        for t in tqdm(noise_scheduler.timesteps, desc=f"Denoising [{i+1}/{num_images}]", leave=False):
            with torch.no_grad():
                noise_pred = unet(latents, t).sample
            latents = noise_scheduler.step(noise_pred, t, latents).prev_sample

        img = (latents / 2 + 0.5).clamp(0, 1)
        img = img.permute(0, 2, 3, 1).detach().cpu().to(torch.float32).numpy()[0]
        img = (img * 255).round().clip(0, 255).astype("uint8")
        images.append(Image.fromarray(img))

    return images


def compute_fid(real_images, fake_images, device):
    if len(real_images) < 2 or len(fake_images) < 2:
        print(f"  ⚠ FID skipped: need ≥2 samples per distribution (real={len(real_images)}, fake={len(fake_images)})")
        return 0.0
    fid = FrechetInceptionDistance(feature=64).to(device)
    for img in real_images:
        img_255 = (img.unsqueeze(0).to(device) * 255).to(torch.uint8)
        fid.update(img_255, real=True)
    for img in fake_images:
        img_255 = (img.unsqueeze(0).to(device) * 255).to(torch.uint8)
        fid.update(img_255, real=False)
    return fid.compute().item()


def compute_clip_score(images, prompts, device):
    from torchmetrics.multimodal.clip_score import CLIPScore

    metric = CLIPScore(model_name_or_path="zer0int/LongCLIP-L-Diffusers").to(device)

    scores = []
    for img, prompt in zip(images, prompts):
        if not isinstance(img, Image.Image):
            img = ToPILImage()(img)
        score = metric(img, prompt)
        scores.append(score.detach().round().item())

    mean_score = np.mean(scores) if scores else 0.0
    return mean_score, scores


def collect_real_images(image_dir, descriptions_file, num_images, device):
    _, test_dataset = create_train_test_splits(
        image_dir=image_dir,
        descriptions_file=descriptions_file,
        image_size=256,
        train_ratio=0.5,
    )
    real_images = []
    for item in test_dataset:
        if len(real_images) >= num_images:
            break
        img = item["image"]
        img = (img + 1) / 2
        real_images.append(img)
    return real_images


def find_lora_checkpoints(base_dir="."):
    checkpoints = []
    for p in sorted(Path(base_dir).glob("vanilla_finetuned_lora_*/final")):
        if p.is_dir():
            rank = p.parent.name.replace("vanilla_finetuned_lora_", "")
            checkpoints.append({"path": str(p), "rank": rank})
    for p in sorted(Path(base_dir).glob("vanilla_finetuned_uncond/final")):
        if p.is_dir():
            checkpoints.append({"path": str(p), "rank": "uncond"})
    return checkpoints


def evaluate_checkpoints(
    checkpoints,
    image_dir="data/artifacts_with_descriptions",
    descriptions_file="data/all_artifacts.json",
    num_inference_steps=100,
    guidance_scale=7.5,
    num_generated_per_prompt=3,
    output_file="evaluation_results.json",
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    

    train_dataset, test_dataset = create_train_test_splits(
        image_dir, descriptions_file, image_size=256, train_ratio=0.5, seed=42
    )
    test_prompts = [test_dataset[i]["text"] for i in range(len(test_dataset))]
    random.seed(42)
    prompts = random.sample(test_prompts, min(5, len(test_prompts)))
    print(f"Using {len(prompts)} prompts from test split")

    real_images = collect_real_images(image_dir, descriptions_file, 10_000, device)
    print(f"Collected {len(real_images)} real images for FID reference")

    results = {}

    for ckpt in checkpoints:
        ckpt_path = ckpt["path"]
        rank = ckpt["rank"]
        label = "uncond" if rank == "uncond" else f"LoRA rank={rank}"
        print(f"\n{'='*70}")
        print(f"Evaluating {label} ({ckpt_path})")
        print(f"{'='*70}")

        models = load_finetuned_models(ckpt_path, device)

        is_uncond = "tokenizer" not in models

        all_gen_pil = []
        used_prompts = []
        all_seeds = []
        all_gen_tensors = []

        if is_uncond:
            num_uncond = len(prompts) * num_generated_per_prompt
            uncond_imgs = generate_unconditional_images(
                models, num_images=num_uncond,
                num_inference_steps=num_inference_steps, seed=42,
            )
            for pil_img in uncond_imgs:
                all_gen_pil.append(pil_img)
                all_gen_tensors.append(ToTensor()(pil_img))
                used_prompts.append("")
        else:
            for i, prompt in enumerate(tqdm(prompts, desc="Generating images")):
                for j in range(num_generated_per_prompt):
                    seed = i * num_generated_per_prompt + j + 42
                    all_seeds.append(seed)
                    imgs = generate_images(
                        [prompt], models,
                        num_inference_steps=num_inference_steps,
                        guidance_scale=guidance_scale,
                        seed=seed,
                    )
                    _, pil_img = imgs[0]
                    all_gen_pil.append(pil_img)
                    all_gen_tensors.append(ToTensor()(pil_img))
                    used_prompts.append(prompt)

        print(f"Generated {len(all_gen_pil)} images")

        fid_score = compute_fid(real_images, all_gen_tensors, device)
        print(f"FID: {fid_score:.4f}")

        if is_uncond:
            clip_mean, clip_per_image = 0.0, []
        else:
            clip_mean, clip_per_image = compute_clip_score(all_gen_pil, used_prompts, device)
        print(f"CLIP Score (mean): {clip_mean:.4f}")

        rank_num = 0 if is_uncond else int(rank)
        label = "uncond" if is_uncond else f"lora_{rank}"
        results[label] = {
            "rank": rank_num,
            "checkpoint": ckpt_path,
            "fid": fid_score,
            "clip_score_mean": clip_mean,
            "clip_score_per_image": clip_per_image,
            "num_generated": len(all_gen_pil),
            "num_real_for_fid": len(real_images),
            "prompts": prompts,
            "seeds": all_seeds,
            "num_inference_steps": num_inference_steps,
            "guidance_scale": guidance_scale,
        }

    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_file}")

    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"{'Model':<20} {'FID':<12} {'CLIP Score':<12}")
    print("-" * 44)
    for key, val in results.items():
        print(f"{key:<20} {val['fid']:<12.4f} {val['clip_score_mean']:<12.4f}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate finetuned models with FID and CLIP score")
    parser.add_argument("--image-dir", type=str, default="data/artifacts_with_descriptions")
    parser.add_argument("--descriptions-file", type=str, default="data/all_artifacts.json")
    parser.add_argument("--num-generated-per-prompt", type=int, default=5,
                        help="How many images to generate per prompt for metrics")
    parser.add_argument("--num-inference-steps", type=int, default=100)
    parser.add_argument("--guidance-scale", type=float, default=7.5)
    parser.add_argument("--output-file", type=str, default="evaluation_results.json")
    parser.add_argument("--folder", type=str, default=None,
                        help="Evaluate a single folder (e.g. vanilla_finetuned_lora_256/final or vanilla_finetuned_uncond/final)")
    args = parser.parse_args()

    if args.folder:
        folder_path = Path(args.folder)
        if folder_path.is_dir():
            parent = folder_path.parent.name
            checkpoints = [{"path": str(folder_path), "rank": parent}]
        else:
            print(f"Folder not found: {args.folder}")
            exit(1)
    else:
        checkpoints = find_lora_checkpoints()
        if not checkpoints:
            print("No finetuned checkpoints found matching 'vanilla_finetuned_lora_*/final' or 'vanilla_finetuned_uncond/final'")
            exit(1)

    print(f"Found {len(checkpoints)} checkpoint(s):")
    for ckpt in checkpoints:
        print(f"  - {ckpt['rank']} ({ckpt['path']})")

    evaluate_checkpoints(
        checkpoints,
        image_dir=args.image_dir,
        descriptions_file=args.descriptions_file,
        num_generated_per_prompt=args.num_generated_per_prompt,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        output_file=args.output_file,
    )
