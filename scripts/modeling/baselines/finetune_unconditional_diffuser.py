import argparse
import json
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from pathlib import Path
from tqdm import tqdm
from accelerate import Accelerator
from diffusers import UNet2DModel, DDPMScheduler, DDPMPipeline
from diffusers.utils import make_image_grid
from diffusers.optimization import get_cosine_schedule_with_warmup
from torchvision import transforms
from PIL import Image
import os
# following https://huggingface.co/docs/diffusers/tutorials/basic_training

CONFIG = {
    "output_dir": "vanilla_finetuned_uncond",
    "learning_rate": 1e-4,
    "batch_size": 4,
    "num_epochs": 50,
    "image_size": 128,
    "gradient_accumulation_steps": 1,
    "lr_warmup_steps": 500,
    "mixed_precision": "no",
    "save_image_epochs": 10,
    "save_model_epochs": 30,
}


def make_grid(images, rows, cols):
    w, h = images[0].size
    grid = Image.new("RGB", (cols * w, rows * h))
    for i, img in enumerate(images):
        grid.paste(img, (i % cols * w, i // cols * h))
    return grid


def evaluate(config, epoch, pipeline):
    images = pipeline(
        batch_size=4,
        generator=torch.Generator(device="cpu").manual_seed(0),
    ).images
    image_grid = make_grid(images, rows=2, cols=2)
    test_dir = Path(config["output_dir"]) / "samples"
    test_dir.mkdir(parents=True, exist_ok=True)
    image_grid.save(str(test_dir / f"{epoch:04d}.png"))


def finetune_unconditional_diffuser(
    image_dir,
    config=CONFIG,
):
    accelerator = Accelerator(
        mixed_precision=config["mixed_precision"],
        gradient_accumulation_steps=config["gradient_accumulation_steps"],
        log_with="tensorboard",
        project_dir=os.path.join(config["output_dir"], "logs"),
    )

    if accelerator.is_main_process:
        Path(config["output_dir"]).mkdir(parents=True, exist_ok=True)
        accelerator.init_trackers("train_example")

    model = UNet2DModel(
        sample_size=config["image_size"],
        in_channels=3,
        out_channels=3,
        layers_per_block=2,
        block_out_channels=(128, 128, 256, 256, 512, 512),
        down_block_types=(
            "DownBlock2D",
            "DownBlock2D",
            "DownBlock2D",
            "DownBlock2D",
            "AttnDownBlock2D",
            "DownBlock2D",
        ),
        up_block_types=(
            "UpBlock2D",
            "AttnUpBlock2D",
            "UpBlock2D",
            "UpBlock2D",
            "UpBlock2D",
            "UpBlock2D",
        ),
    )

    noise_scheduler = DDPMScheduler(num_train_timesteps=1000)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"])

    preprocess = transforms.Compose(
        [
            transforms.Resize((config["image_size"], config["image_size"])),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )

    paths = sorted(Path(image_dir).glob("*.png"))
    dataset = [(preprocess(Image.open(p).convert("RGB")), p.name) for p in paths]
    print(f"Loaded {len(dataset)} images")

    class ImageDataset(torch.utils.data.Dataset):
        def __init__(self, data):
            self.data = data

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            return {"images": self.data[idx][0]}

    train_dataset = ImageDataset(dataset)

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=0,
    )

    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=config["lr_warmup_steps"],
        num_training_steps=(len(train_dataloader) * config["num_epochs"]),
    )

    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, lr_scheduler,
    )

    global_step = 0
    epoch_losses = []

    for epoch in range(config["num_epochs"]):
        progress_bar = tqdm(
            total=len(train_dataloader),
            disable=not accelerator.is_local_main_process,
        )
        progress_bar.set_description(f"Epoch {epoch}")

        total_loss = 0

        for step, batch in enumerate(train_dataloader):
            clean_images = batch["images"]
            noise = torch.randn(clean_images.shape, device=clean_images.device)
            bs = clean_images.shape[0]

            timesteps = torch.randint(
                0,
                noise_scheduler.config.num_train_timesteps,
                (bs,),
                device=clean_images.device,
                dtype=torch.int64,
            )

            noisy_images = noise_scheduler.add_noise(clean_images, noise, timesteps)

            with accelerator.accumulate(model):
                noise_pred = model(noisy_images, timesteps, return_dict=False)[0]
                loss = F.mse_loss(noise_pred, noise)
                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            progress_bar.update(1)
            total_loss += loss.detach().item()
            logs = {
                "loss": loss.detach().item(),
                "lr": lr_scheduler.get_last_lr()[0],
                "step": global_step,
            }
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)
            global_step += 1

        avg_loss = total_loss / len(train_dataloader)
        epoch_losses.append(avg_loss)

        if accelerator.is_main_process:
            pipeline = DDPMPipeline(
                unet=accelerator.unwrap_model(model),
                scheduler=noise_scheduler,
            )

            if (epoch + 1) % config["save_image_epochs"] == 0 or epoch == config["num_epochs"] - 1:
                evaluate(config, epoch, pipeline)

            if (epoch + 1) % config["save_model_epochs"] == 0 or epoch == config["num_epochs"] - 1:
                output_path = Path(config["output_dir"]) / f"checkpoint_epoch_{epoch + 1}"
                output_path.mkdir(parents=True, exist_ok=True)
                pipeline.save_pretrained(str(output_path))
                noise_scheduler.save_pretrained(str(output_path / "scheduler"))
                print(f"Checkpoint saved to {output_path}")

        print(f"Epoch {epoch + 1}/{config['num_epochs']} average loss: {avg_loss:.4f}")

    accelerator.end_training()

    if accelerator.is_main_process:
        final_path = Path(config["output_dir"]) / "final"
        final_path.mkdir(parents=True, exist_ok=True)
        pipeline = DDPMPipeline(
            unet=accelerator.unwrap_model(model),
            scheduler=noise_scheduler,
        )
        pipeline.save_pretrained(str(final_path))

        training_log = {
            "config": config,
            "train_losses": epoch_losses,
        }
        log_path = Path(config["output_dir"]) / "training_log.json"
        with open(log_path, "w") as f:
            json.dump(training_log, f, indent=2)
        print(f"Training log saved to {log_path}")
        print(f"\n✓ Finetuning complete! Model saved to {final_path}")

    return {"pipeline": pipeline if accelerator.is_main_process else None}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Finetune unconditional diffuser model")
    parser.add_argument("--output-dir", type=str, default=CONFIG["output_dir"])
    parser.add_argument("--image-dir", type=str, default="data/cropped_artifacts")
    parser.add_argument("--batch-size", type=int, default=CONFIG["batch_size"])
    parser.add_argument("--num-epochs", type=int, default=CONFIG["num_epochs"])
    parser.add_argument("--learning-rate", type=float, default=CONFIG["learning_rate"])
    parser.add_argument("--image-size", type=int, default=CONFIG["image_size"])
    args = parser.parse_args()

    CONFIG["output_dir"] = args.output_dir
    CONFIG["batch_size"] = args.batch_size
    CONFIG["num_epochs"] = args.num_epochs
    CONFIG["learning_rate"] = args.learning_rate
    CONFIG["image_size"] = args.image_size

    finetune_unconditional_diffuser(
        image_dir=args.image_dir,
        config=CONFIG,
    )
