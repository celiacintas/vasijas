import argparse
import json
import sys
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from pathlib import Path
from tqdm import tqdm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from diffusers import UNet2DModel, DDPMScheduler
from peft import get_peft_model, LoraConfig

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from ceramic_dataset import create_train_test_splits

CONFIG = {
    "model_name": "google/ddpm-ema-celebahq-256",
    "output_dir": "vanilla_finetuned_uncond",
    "learning_rate": 1e-5,
    "batch_size": 4,
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
    unet, noise_scheduler, device, output_dir,
    num_inference_steps=50, num_images=4, image_size=256,
):
    """Generate images and save frames showing the denoising process."""
    save_steps = [0, 10, 20, 30, 40, 45, 49]
    noise_scheduler.set_timesteps(num_inference_steps)

    for img_idx in range(num_images):
        generator = torch.Generator(device=device).manual_seed(42 + img_idx)
        noisy_image = torch.randn(
            (1, unet.config.in_channels, image_size, image_size),
            generator=generator, device=device, dtype=torch.float32,
        )
        noisy_image = noisy_image * noise_scheduler.init_noise_sigma

        frames = []
        step_labels = []

        for step_idx, t in enumerate(noise_scheduler.timesteps):
            model_input = noise_scheduler.scale_model_input(noisy_image, t)
            noise_pred = unet(model_input, t).sample
            noisy_image = noise_scheduler.step(noise_pred, t, noisy_image).prev_sample

            if step_idx in save_steps or step_idx == len(noise_scheduler.timesteps) - 1:
                image = (noisy_image.float() / 2 + 0.5).clamp(0, 1).squeeze(0).cpu().detach().permute(1, 2, 0)
                frames.append(image.numpy())
                step_labels.append(f"step {step_idx}")

        fig, axes = plt.subplots(1, len(frames), figsize=(4 * len(frames), 4))
        if len(frames) == 1:
            axes = [axes]
        for ax, frame, label in zip(axes, frames, step_labels):
            ax.imshow(frame)
            ax.set_title(label, fontsize=10)
            ax.axis("off")
        fig.text(0.5, 0.01, "Unconditional generation", ha="center", va="bottom", fontsize=10, wrap=True)
        plt.tight_layout()
        out_path = Path(output_dir) / f"denoise_{img_idx:02d}.png"
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Denoising sequence saved to {out_path}")


def save_loss_plot(losses, test_losses, output_dir):
    plt.figure(figsize=(10, 6))
    epochs = range(1, len(losses) + 1)
    plt.plot(epochs, losses, marker="o", linewidth=2, markersize=8, label="Train")
    plt.plot(epochs, test_losses, marker="s", linewidth=2, markersize=8, label="Test")
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Loss", fontsize=12)
    plt.title("Training Loss Over Time", fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks(epochs)
    output_path = Path(output_dir) / "training_loss.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Training loss plot saved to {output_path}")


def finetune_unconditional_diffuser(
    image_dir,
    config=CONFIG
):
    print("\n" + "="*70)
    print("LOADING MODELS")
    print("="*70)

    device = torch.device(f"cuda:{CONFIG['gpu']}" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print("Loading pretrained UNet...")
    unet = UNet2DModel.from_pretrained(
        config["model_name"],
        subfolder="unet",
        torch_dtype=torch.float32
    )

    print("Loading scheduler...")
    noise_scheduler = DDPMScheduler.from_pretrained(
        config["model_name"],
        subfolder="scheduler"
    )

    if config["use_lora"]:
        print("Applying LoRA to UNet...")
        lora_config = LoraConfig(
            r=config["lora_rank"],
            lora_alpha=config["lora_rank"],
            target_modules=["conv1", "conv2", "conv", "conv_out"],
            lora_dropout=0.1,
            bias="none"
        )
        unet = get_peft_model(unet, lora_config)
        unet.print_trainable_parameters()

    unet = unet.to(device)

    optimizer = torch.optim.AdamW(
        unet.parameters(),
        lr=config["learning_rate"],
        weight_decay=0.01
    )

    # Dataset
    print("\n" + "="*70)
    print("LOADING DATASET")
    print("="*70)

    train_dataset, test_dataset = create_train_test_splits(
        image_dir=image_dir,
        descriptions_file="data/all_artifacts.json",
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

    Path(config["output_dir"]).mkdir(parents=True, exist_ok=True)

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
            images = batch["image"].to(device)

            noise = torch.randn_like(images)
            timesteps = torch.randint(
                0,
                len(noise_scheduler),
                (images.shape[0],),
                device=device
            )

            noisy_images = noise_scheduler.add_noise(images, noise, timesteps)

            noise_pred = unet(noisy_images, timesteps).sample

            loss = F.mse_loss(noise_pred, noise)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(unet.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()

            total_loss += loss.item()
            progress_bar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "avg_loss": f"{total_loss / (batch_idx + 1):.4f}"
            })

        avg_loss = total_loss / min(steps_per_epoch, len(train_dataloader))
        epoch_losses.append(avg_loss)
        print(f"Epoch {epoch + 1} average loss: {avg_loss:.4f}")

        # Test evaluation
        unet.eval()
        test_loss = 0
        with torch.no_grad():
            for batch in test_dataloader:
                images = batch["image"].to(device)

                noise = torch.randn_like(images)
                timesteps = torch.randint(
                    0,
                    len(noise_scheduler),
                    (images.shape[0],),
                    device=device
                )

                noisy_images = noise_scheduler.add_noise(images, noise, timesteps)
                noise_pred = unet(noisy_images, timesteps).sample

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

    noise_scheduler.save_pretrained(str(final_path / "scheduler"))

    save_loss_plot(epoch_losses, epoch_test_losses, config["output_dir"])

    print("\n" + "="*70)
    print("GENERATING DENOISING SEQUENCES")
    print("="*70)
    unet.eval()
    save_denoising_sequence(
        unet, noise_scheduler, device,
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
        "noise_scheduler": noise_scheduler,
        "device": device
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Finetune unconditional diffuser model (images only, no text)")
    parser.add_argument("--output-dir", type=str, default=CONFIG["output_dir"],
                        help="Name of the model output folder (default: vanilla_finetuned_uncond)")
    parser.add_argument("--model-name", type=str, default=CONFIG["model_name"],
                        help="Base model name or path (default: google/ddpm-ema-celebahq-256)")
    parser.add_argument("--image-dir", type=str, default="data/cropped_artifacts",
                        help="Directory containing images (default: data/cropped_artifacts)")
    parser.add_argument("--steps-per-epoch", type=int, default=None,
                        help="Limit training steps per epoch (default: all batches)")
    parser.add_argument("--train-ratio", type=float, default=CONFIG["train_ratio"],
                        help="Ratio of data for training (default: 0.5)")
    parser.add_argument("--lora-rank", type=int, default=CONFIG["lora_rank"],
                        help="LoRA rank for fine-tuning (default: 16)")
    args = parser.parse_args()

    CONFIG["output_dir"] = args.output_dir
    CONFIG["model_name"] = args.model_name
    CONFIG["steps_per_epoch"] = args.steps_per_epoch
    CONFIG["train_ratio"] = args.train_ratio
    CONFIG["lora_rank"] = args.lora_rank
    CONFIG["use_lora"] = args.lora_rank > 0

    models = finetune_unconditional_diffuser(
        image_dir=args.image_dir,
        config=CONFIG
    )

    print("\n✓ Finetuning complete!")
    print(f"Models saved in: {CONFIG['output_dir']}")
