import argparse
import sys
import torch
from pathlib import Path
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import AutoencoderKL, UNet2DConditionModel, DDPMScheduler
from peft import PeftModel

sys.path.insert(0, str(Path(__file__).parent))
from finetune_vanilla_diffuser import save_denoising_sequence


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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate denoising sequence from a finetuned checkpoint")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to checkpoint directory (e.g. vanilla_finetuned_lora_256/final)")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory (default: same as checkpoint)")
    parser.add_argument("--num-inference-steps", type=int, default=50)
    parser.add_argument("--guidance-scale", type=float, default=7.5)
    parser.add_argument("--num-images", type=int, default=3)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--prompt", type=str, action="append", default=None,
                        help="Custom prompt(s) to use (can be specified multiple times)")
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"Loading models from {args.checkpoint}...")

    models = load_finetuned_models(args.checkpoint, device)

    output_dir = args.output_dir or args.checkpoint
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    print(f"Generating denoising sequences to {output_dir}...")
    save_denoising_sequence(
        models["unet"], models["vae"], models["text_encoder"], models["tokenizer"],
        models["noise_scheduler"], device, output_dir,
        prompts=args.prompt,
        num_inference_steps=args.num_inference_steps,
        num_images=args.num_images,
        image_size=args.image_size,
        guidance_scale=args.guidance_scale,
    )

    print("\nDone!")
