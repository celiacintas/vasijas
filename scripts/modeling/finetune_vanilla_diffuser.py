import argparse
import json
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from pathlib import Path
from tqdm import tqdm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import AutoencoderKL, UNet2DConditionModel, DDPMScheduler
from peft import get_peft_model, LoraConfig, TaskType
from ceramic_dataset import create_train_test_splits

# Configuration
CONFIG = {
    "model_name": "runwayml/stable-diffusion-v1-5",
    #"OFA-Sys/small-stable-diffusion-v0", #
    "output_dir": "vanilla_finetuned",
    "learning_rate": 1e-5,
    "batch_size": 16,
    "num_epochs": 5,
    "image_size": 256,
    "use_lora": True,
    "lora_rank": 16,
    "gpu": 0,
    "steps_per_epoch": None,
    "train_ratio": 0.5,
}

torch.cuda.set_device(CONFIG["gpu"])

def save_denoising_sequence(
    unet, vae, text_encoder, tokenizer, noise_scheduler, device, output_dir,
    prompts=None, num_inference_steps=50, num_images=4, image_size=256,
):
    """Generate images and save frames showing the denoising process."""
    if prompts is None:
        prompts = [
            "a ceramic plate with iberian geometric, linear-based decoration with alternating cream and red fields; hatching and stippling create depth and visual interest across fragmented vessel.",
            "a ceramic plate with a central solid red circle and a concentric design featuring an outer ring of alternating red and white rectangular segments arranged radially geometric, highly symmetrical composition with regular spacing",
            "a ceramic vessel with graduated complexity from base to rim, with decoration increasing in density toward the top. The combination of simple lines and crosshatched triangles creates a dynamic visual hierarchy. The vessel demonstrates controlled, red geometric patterning typical of iberian ceramic design." ]

    save_steps = [0, 10, 30, 45]
    noise_scheduler.set_timesteps(num_inference_steps)

    for img_idx, prompt in enumerate(prompts[:num_images]):
        generator = torch.Generator(device=device).manual_seed(42 + img_idx)
        latents = torch.randn(
            (1, unet.config.in_channels, image_size // 8, image_size // 8),
            generator=generator, device=device, dtype=torch.float16,
        )

        text_input = tokenizer(
            prompt, padding="max_length",
            max_length=tokenizer.model_max_length, truncation=True,
            return_tensors="pt",
        )
        text_embeddings = text_encoder(text_input.input_ids.to(device))[0]

        frames = []
        step_labels = []
        latents = latents * noise_scheduler.init_noise_sigma

        for step_idx, t in enumerate(noise_scheduler.timesteps):
            latent_model_input = noise_scheduler.scale_model_input(latents, t)
            noise_pred = unet(
                latent_model_input, t,
                encoder_hidden_states=text_embeddings,
            ).sample
            latents = noise_scheduler.step(noise_pred, t, latents).prev_sample

            if step_idx in save_steps or step_idx == len(noise_scheduler.timesteps) - 1:
                with torch.no_grad():
                    denoised = latents / 0.18215
                    image = vae.decode(denoised.float()).sample
                    image = (image / 2 + 0.5).clamp(0, 1).squeeze(0).cpu().permute(1, 2, 0)
                frames.append(image.numpy())
                step_labels.append(f"t={t.item()}")

        fig, axes = plt.subplots(1, len(frames), figsize=(4 * len(frames), 4))
        if len(frames) == 1:
            axes = [axes]
        for ax, frame, label in zip(axes, frames, step_labels):
            ax.imshow(frame)
            ax.set_title(label, fontsize=10)
            ax.axis("off")
        fig.suptitle(prompt, fontsize=12, y=0.95)
        plt.tight_layout()
        out_path = Path(output_dir) / f"denoise_{img_idx:02d}.png"
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Denoising sequence saved to {out_path}")


def save_loss_plot(losses, output_dir):
    """Save training loss as a line plot."""
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(losses) + 1), losses, marker="o", linewidth=2, markersize=8)
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Training Loss", fontsize=12)
    plt.title("Training Loss Over Time", fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.xticks(range(1, len(losses) + 1))
    
    output_path = Path(output_dir) / "training_loss.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Training loss plot saved to {output_path}")


def finetune_vanilla_diffuser(
    image_dir,
    descriptions_file,
    config=CONFIG
):
    """Finetune diffuser model with text descriptions"""
    
    print("\n" + "="*70)
    print("LOADING MODELS")
    print("="*70)
    
    device = torch.device(f"cuda:{CONFIG['gpu']}" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # Load tokenizer and text encoder
    print("Loading tokenizer...")
    tokenizer = CLIPTokenizer.from_pretrained(
        config["model_name"],
        subfolder="tokenizer"
    )
    
    print("Loading text encoder...")
    text_encoder = CLIPTextModel.from_pretrained(
        config["model_name"],
        subfolder="text_encoder",
        torch_dtype=torch.float16 if device.type == "cuda" else torch.float32
    )
    text_encoder = text_encoder.to(device)
    text_encoder.requires_grad_(False)
    
    print("Loading VAE...")
    vae = AutoencoderKL.from_pretrained(
        config["model_name"],
        subfolder="vae",
        torch_dtype=torch.float16 if device.type == "cuda" else torch.float32
    )
    vae = vae.to(device)
    vae.requires_grad_(False)
    
    print("Loading UNet...")
    unet = UNet2DConditionModel.from_pretrained(
        config["model_name"],
        subfolder="unet",
        torch_dtype=torch.float16 if device.type == "cuda" else torch.float32
    )
    
    print("Loading scheduler...")
    noise_scheduler = DDPMScheduler.from_pretrained(
        config["model_name"],
        subfolder="scheduler"
    )
    
    # Apply LoRA to UNet
    if config["use_lora"]:
        print("Applying LoRA to UNet...")
        lora_config = LoraConfig(
            r=config["lora_rank"],
            lora_alpha=config["lora_rank"],
            target_modules=["to_k", "to_v", "to_q", "linear_1", "linear_2"],
            lora_dropout=0.1,
            bias="none"
        )
        unet = get_peft_model(unet, lora_config)
        unet.print_trainable_parameters()
    
    unet = unet.to(device)
    
    # Setup optimizer
    optimizer = torch.optim.AdamW(
        unet.parameters(),
        lr=config["learning_rate"],
        weight_decay=0.01
    )
    
    # Load dataset
    print("\n" + "="*70)
    print("LOADING DATASET")
    print("="*70)
    
    train_dataset, test_dataset = create_train_test_splits(
        image_dir=image_dir,
        descriptions_file=descriptions_file,
        image_size=config["image_size"],
        train_ratio=config.get("train_ratio", 0.5)
    )
    
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=4,
        pin_memory=True if device.type == "cuda" else False
    )
    
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=4,
        pin_memory=True if device.type == "cuda" else False
    )
    
    print(f"Train batches: {len(train_dataloader)}")
    print(f"Test batches: {len(test_dataloader)}")
    
    # Training loop
    print("\n" + "="*70)
    print("STARTING TRAINING")
    print("="*70)
    
    unet.train()
    
    epoch_losses = []
    epoch_test_losses = []
    
    for epoch in range(config["num_epochs"]):
        print(f"\n[Epoch {epoch + 1}/{config['num_epochs']}]")
        
        steps_per_epoch = config.get("steps_per_epoch") or len(train_dataloader)
        progress_bar = tqdm(train_dataloader, desc="Training", total=steps_per_epoch)
        total_loss = 0
        
        for batch_idx, batch in enumerate(progress_bar):
            if batch_idx >= steps_per_epoch:
                break
            # Move to device
            images = batch["image"].to(device)
            texts = batch["text"]
            
            with torch.no_grad():
                # Encode text
                text_input = tokenizer(
                    texts,
                    padding="max_length",
                    max_length=tokenizer.model_max_length,
                    truncation=True,
                    return_tensors="pt"
                )
                text_embeddings = text_encoder(
                    text_input.input_ids.to(device)
                )[0]
                
                # Encode images to latent space
                latents = vae.encode(images.half()).latent_dist.sample()
                latents = latents * 0.18215  # VAE scaling factor
            
            # Sample noise and timesteps
            noise = torch.randn_like(latents)
            timesteps = torch.randint(
                0,
                len(noise_scheduler),
                (latents.shape[0],),
                device=device
            )
            
            # Add noise to latents (forward diffusion)
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
            
            # Predict noise residual
            noise_pred = unet(
                noisy_latents,
                timesteps,
                encoder_hidden_states=text_embeddings
            ).sample
            
            # Calculate loss
            loss = F.mse_loss(noise_pred, noise)
            
            # Backward pass
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(unet.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()
            
            total_loss += loss.item()
            progress_bar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "avg_loss": f"{total_loss / (batch_idx + 1):.4f}"
            })
        
        avg_loss = total_loss / len(train_dataloader)
        epoch_losses.append(avg_loss)
        print(f"Epoch {epoch + 1} average loss: {avg_loss:.4f}")
        
        # Evaluate on test set
        unet.eval()
        test_loss = 0
        with torch.no_grad():
            for batch in test_dataloader:
                images = batch["image"].to(device)
                texts = batch["text"]
                
                text_input = tokenizer(
                    texts,
                    padding="max_length",
                    max_length=tokenizer.model_max_length,
                    truncation=True,
                    return_tensors="pt"
                )
                text_embeddings = text_encoder(
                    text_input.input_ids.to(device)
                )[0]
                
                latents = vae.encode(images.half()).latent_dist.sample()
                latents = latents * 0.18215
                
                noise = torch.randn_like(latents)
                timesteps = torch.randint(
                    0,
                    len(noise_scheduler),
                    (latents.shape[0],),
                    device=device
                )
                
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
                noise_pred = unet(
                    noisy_latents,
                    timesteps,
                    encoder_hidden_states=text_embeddings
                ).sample
                
                test_loss += F.mse_loss(noise_pred, noise).item()
        
        avg_test_loss = test_loss / len(test_dataloader)
        epoch_test_losses.append(avg_test_loss)
        print(f"Epoch {epoch + 1} test loss: {avg_test_loss:.4f}")
        unet.train()
        
        # Save checkpoint
        output_path = Path(config["output_dir"]) / f"checkpoint_epoch_{epoch + 1}"
        output_path.mkdir(parents=True, exist_ok=True)
        
        print(f"\nSaving checkpoint to {output_path}...")
        
        if config["use_lora"]:
            unet.save_pretrained(str(output_path / "unet_lora"))
        else:
            unet.save_pretrained(str(output_path / "unet"))
        
        # Also save tokenizer and config
        tokenizer.save_pretrained(str(output_path / "tokenizer"))
        noise_scheduler.save_pretrained(str(output_path / "scheduler"))
    
    # Save final model
    print("\n" + "="*70)
    print("SAVING FINAL MODEL")
    print("="*70)
    
    final_path = Path(config["output_dir"]) / "final"
    final_path.mkdir(parents=True, exist_ok=True)
    
    if config["use_lora"]:
        unet.save_pretrained(str(final_path / "unet_lora"))
    else:
        unet.save_pretrained(str(final_path / "unet"))
    
    tokenizer.save_pretrained(str(final_path / "tokenizer"))
    noise_scheduler.save_pretrained(str(final_path / "scheduler"))
    vae.save_pretrained(str(final_path / "vae"))
    text_encoder.save_pretrained(str(final_path / "text_encoder"))
    
    save_loss_plot(epoch_losses, config["output_dir"])
    
    print("\n" + "="*70)
    print("GENERATING DENOISING SEQUENCES")
    print("="*70)
    unet.eval()
    save_denoising_sequence(
        unet, vae, text_encoder, tokenizer, noise_scheduler, device,
        config["output_dir"], image_size=config["image_size"],
    )
    
    training_log = {
        "config": config,
        "train_losses": epoch_losses,
        "test_losses": epoch_test_losses,
    }
    log_path = Path(config["output_dir"]) / "training_log.json"
    with open(log_path, "w") as f:
        json.dump(training_log, f, indent=2)
    print(f"Training log saved to {log_path}")
    
    print(f"\n✓ Finetuning complete! Model saved to {final_path}")
    
    return {
        "unet": unet,
        "text_encoder": text_encoder,
        "vae": vae,
        "tokenizer": tokenizer,
        "noise_scheduler": noise_scheduler,
        "device": device
    }

# Usage
if __name__ == "__main__":
    import numpy as np
    
    parser = argparse.ArgumentParser(description="Finetune vanilla diffuser model")
    parser.add_argument("--output-dir", type=str, default=CONFIG["output_dir"],
                        help="Name of the model output folder (default: vanilla_finetuned)")
    parser.add_argument("--model-name", type=str, default=CONFIG["model_name"],
                        help="Base model name or path (default: runwayml/stable-diffusion-v1-5)")
    parser.add_argument("--steps-per-epoch", type=int, default=None,
                        help="Limit training steps per epoch (default: all batches)")
    parser.add_argument("--train-ratio", type=float, default=CONFIG["train_ratio"],
                        help="Ratio of data for training (default: 0.8)")
    parser.add_argument("--lora-rank", type=int, default=CONFIG["lora_rank"],
                        help="LoRA rank for fine-tuning (default: 16)")
    args = parser.parse_args()
    
    CONFIG["output_dir"] = args.output_dir
    CONFIG["model_name"] = args.model_name
    CONFIG["steps_per_epoch"] = args.steps_per_epoch
    CONFIG["train_ratio"] = args.train_ratio
    CONFIG["lora_rank"] = args.lora_rank
    
    # Prepare dataset first
    print("Make sure you have run: python prepare_dataset.py\n")
    
    # Finetune
    models = finetune_vanilla_diffuser(
        image_dir="data/cropped_artifacts",
        descriptions_file="data/all_artifacts.json",
        config=CONFIG
    )
    
    print("\n✓ Finetuning complete!")
    print(f"Models saved in: {CONFIG['output_dir']}")