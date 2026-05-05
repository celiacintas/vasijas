import argparse
import json
import torch
import numpy as np
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import AutoencoderKL, UNet2DConditionModel, DDPMScheduler
from peft import PeftModel
from torchmetrics.image.fid import FrechetInceptionDistance
from torchvision.transforms import ToTensor
import sys 

sys.path.insert(0, str(Path(__file__).parent))
from ceramic_dataset import create_train_test_splits

PROMPTS = [
    "a ceramic plate with iberian geometric, linear-based decoration with alternating cream and red fields; hatching and stippling create depth and visual interest across fragmented vessel.",
    "a ceramic plate with a central solid red circle and a concentric design featuring an outer ring of alternating red and white rectangular segments arranged radially geometric, highly symmetrical composition with regular spacing",
    "a ceramic vessel with graduated complexity from base to rim, with decoration increasing in density toward the top. The combination of simple lines and crosshatched triangles creates a dynamic visual hierarchy. The vessel demonstrates controlled, red geometric patterning typical of iberian ceramic design."
]


def load_finetuned_models(checkpoint_dir, device):
    checkpoint_path = Path(checkpoint_dir)

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


def compute_fid(real_images, fake_images, device):
    fid = FrechetInceptionDistance(feature=64).to(device)
    for img in real_images:
        img_255 = (img.unsqueeze(0).to(device) * 255).to(torch.uint8)
        fid.update(img_255, real=True)
    for img in fake_images:
        img_255 = (img.unsqueeze(0).to(device) * 255).to(torch.uint8)
        fid.update(img_255, real=False)
    return fid.compute().item()


def compute_clip_score(images, prompts, device):
    from transformers import CLIPModel, CLIPProcessor

    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    scores = []
    for img, prompt in zip(images, prompts):
        inputs = processor(text=[prompt], images=img, return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            outputs = model(**inputs)
            logits_per_image = outputs.logits_per_image
            scores.append(logits_per_image.item())

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
    count = 0
    for item in test_dataset:
        if count >= num_images:
            break
        img = item["image"]
        real_images.append(img)
        count += 1
    return real_images


def find_lora_checkpoints(base_dir="."):
    checkpoints = []
    for p in sorted(Path(base_dir).glob("vanilla_finetuned_lora_*/final")):
        if p.is_dir():
            rank = p.parent.name.replace("vanilla_finetuned_lora_", "")
            checkpoints.append({"path": str(p), "rank": rank})
    for p in sorted(Path(base_dir).glob("vanilla_finetuned_full/final")):
        if p.is_dir():
            checkpoints.append({"path": str(p), "rank": "full"})
    return checkpoints


def evaluate_checkpoints(
    checkpoints,
    image_dir="data/cropped_artifacts",
    descriptions_file="data/all_artifacts.json",
    num_inference_steps=100,
    guidance_scale=7.5,
    num_generated_per_prompt=3,
    output_file="evaluation_results.json",
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_real_for_fid = max(len(PROMPTS) * num_generated_per_prompt, 50)

    real_images = collect_real_images(image_dir, descriptions_file, num_real_for_fid, device)
    print(f"Collected {len(real_images)} real images for FID reference")

    results = {}

    for ckpt in checkpoints:
        ckpt_path = ckpt["path"]
        rank = ckpt["rank"]
        print(f"\n{'='*70}")
        label = "full" if rank == "full" else f"LoRA rank={rank}"
        print(f"Evaluating {label} ({ckpt_path})")
        print(f"{'='*70}")

        models = load_finetuned_models(ckpt_path, device)

        all_gen_pil = []
        used_prompts = []
        all_seeds = []
        all_gen_tensors = []

        for i, prompt in enumerate(tqdm(PROMPTS, desc="Generating images")):
            for j in range(num_generated_per_prompt):
                seed = i * num_generated_per_prompt + j + 42
                all_seeds.append(seed)

                generator = torch.Generator(device=device).manual_seed(seed)
                latents = torch.randn(
                    (1, 4, 256 // 8, 256 // 8),
                    generator=generator, device=device, dtype=torch.float16,
                )
                noise_scheduler.set_timesteps(num_inference_steps)
                latents = latents * noise_scheduler.init_noise_sigma

                text_input = models["tokenizer"](
                    prompt, padding="max_length",
                    max_length=models["tokenizer"].model_max_length, truncation=True,
                    return_tensors="pt",
                )
                text_embeds = models["text_encoder"](text_input.input_ids.to(device))[0]
                uncond_input = models["tokenizer"](
                    [""], padding="max_length",
                    max_length=models["tokenizer"].model_max_length, return_tensors="pt",
                )
                uncond_embeds = models["text_encoder"](uncond_input.input_ids.to(device))[0]
                cond_embeds = torch.cat([uncond_embeds, text_embeds])

                for t in noise_scheduler.timesteps:
                    latent_model_input = torch.cat([latents] * 2)
                    latent_model_input = noise_scheduler.scale_model_input(latent_model_input, t)
                    noise_pred = models["unet"](
                        latent_model_input, t,
                        encoder_hidden_states=cond_embeds,
                    ).sample
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
                    latents = noise_scheduler.step(noise_pred, t, latents).prev_sample

                with torch.no_grad():
                    denoised = latents / 0.18215
                    image = models["vae"].decode(denoised).sample
                    image = (image.float() / 2 + 0.5).clamp(0, 1).squeeze(0).cpu().permute(1, 2, 0).numpy()

                pil_img = Image.fromarray((image * 255).astype("uint8"))
                all_gen_pil.append(pil_img)
                all_gen_tensors.append(ToTensor()(pil_img))
                used_prompts.append(prompt)

        print(f"Generated {len(all_gen_pil)} images")

        fid_score = compute_fid(real_images, all_gen_tensors, device)
        print(f"FID: {fid_score:.4f}")

        clip_mean, clip_per_image = compute_clip_score(all_gen_pil, used_prompts, device)
        print(f"CLIP Score (mean): {clip_mean:.4f}")

        rank_num = 0 if rank == "full" else int(rank)
        label = f"full" if rank == "full" else f"lora_{rank}"
        results[label] = {
            "rank": rank_num,
            "checkpoint": ckpt_path,
            "fid": fid_score,
            "clip_score_mean": clip_mean,
            "clip_score_per_image": clip_per_image,
            "num_generated": len(all_gen_pil),
            "num_real_for_fid": len(real_images),
            "prompts": PROMPTS,
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
    parser = argparse.ArgumentParser(description="Evaluate finetuned LoRA models with FID and CLIP score")
    parser.add_argument("--image-dir", type=str, default="data/cropped_artifacts")
    parser.add_argument("--descriptions-file", type=str, default="data/all_artifacts.json")
    parser.add_argument("--num-generated-per-prompt", type=int, default=5,
                        help="How many images to generate per prompt for metrics")
    parser.add_argument("--num-inference-steps", type=int, default=100)
    parser.add_argument("--guidance-scale", type=float, default=7.5)
    parser.add_argument("--output-file", type=str, default="evaluation_results.json")
    parser.add_argument("--folder", type=str, default=None,
                        help="Evaluate a single folder (e.g. vanilla_finetuned_full/final or vanilla_finetuned_lora_256/final)")
    args = parser.parse_args()

    if args.folder:
        folder_path = Path(args.folder)
        if folder_path.is_dir():
            checkpoints = [{"path": str(folder_path), "rank": folder_path.name}]
        else:
            print(f"Folder not found: {args.folder}")
            exit(1)
    else:
        checkpoints = find_lora_checkpoints()
        if not checkpoints:
            print("No finetuned checkpoints found matching 'vanilla_finetuned_lora_*/final' or 'vanilla_finetuned_full/final'")
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
