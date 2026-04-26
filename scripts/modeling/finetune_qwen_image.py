#!/usr/bin/env python3
"""Fine-tune Qwen-Image with LoRA."""

import torch
from torch.utils.data import DataLoader, Dataset
from pathlib import Path
from PIL import Image
import json
from tqdm import tqdm
from diffusers import QwenImagePipeline, QwenImageTransformer2DModel
from peft import LoraConfig, get_peft_model_state_dict
import math


CONFIG = {
    "model_name": "Qwen/Qwen-Image",
    "output_dir": "qwen_finetuned",
    "learning_rate": 2e-4,
    "batch_size": 1,
    "num_epochs": 20,
    "image_size": 256,
    "lora_rank": 16,
    "gpu": 0,
    "instance_prompt": "a ceramic artifact",
    "max_train_steps": 100,
    "gradient_accumulation_steps": 2,
}


class CeramicArtifactDataset(Dataset):
    def __init__(self, image_dir, descriptions_file, image_size=1024):
        self.image_dir = Path(image_dir)
        self.image_size = image_size
        self.descriptions = {}

        if Path(descriptions_file).exists():
            with open(descriptions_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        try:
                            data = json.loads(line)
                            self.descriptions[data['filename']] = data['description']
                        except json.JSONDecodeError:
                            continue

        self.image_paths = sorted(self.image_dir.glob("*.png"))
        print(f"Loaded {len(self.image_paths)} images")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]

        image = Image.open(image_path).convert("RGB")
        image = image.resize((self.image_size, self.image_size), Image.Resampling.LANCZOS)

        description = self.descriptions.get(
            image_path.name,
            "ceramic artifact with decorative patterns"
        )

        return {
            "image": image,
            "text": description,
            "filename": image_path.name
        }


def collate_fn(examples, vae, text_encoder, tokenizer, device):
    images = []
    texts = []

    for example in examples:
        img_array = torch.from_numpy(__import__('numpy').array(example["image"]))
        img_tensor = img_array.permute(2, 0, 1).float() / 127.5 - 1
        images.append(img_tensor)
        texts.append(example["text"])

    images = torch.stack(images).to(device)

    with torch.no_grad():
        latents = vae.encode(images).latent_dist.sample()
        latents = latents * vae.config.scaling_factor

        text_inputs = tokenizer(
            texts,
            padding="max_length",
            max_length=tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt"
        )
        text_embeddings = text_encoder(
            text_inputs.input_ids.to(device)
        )[0]

    return latents, text_embeddings


def finetune_qwen_image(
    image_dir,
    descriptions_file,
    config=CONFIG
):
    device = torch.device(f"cuda:{config['gpu']}" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    weight_dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    print(f"Dtype: {weight_dtype}")

    print("\n" + "="*70)
    print("LOADING MODELS")
    print("="*70)

    print("Loading transformer...")
    transformer = QwenImageTransformer2DModel.from_pretrained(
        config["model_name"],
        subfolder="transformer",
        torch_dtype=weight_dtype,
    )

    print("Loading tokenizer...")
    tokenizer = QwenImagePipeline.from_pretrained(
        config["model_name"],
        vae=None,
        transformer=None,
        tokenizer=None,
        text_encoder=None,
        scheduler=None,
    ).tokenizer

    print("Loading text encoder...")
    text_encoder = QwenImagePipeline.from_pretrained(
        config["model_name"],
        subfolder="text_encoder",
        torch_dtype=weight_dtype,
    ).text_encoder

    print("Loading VAE...")
    vae = QwenImagePipeline.from_pretrained(
        config["model_name"],
        subfolder="vae",
        torch_dtype=weight_dtype,
    ).vae

    text_encoder.requires_grad_(False)
    vae.requires_grad_(False)
    vae.eval()
    text_encoder.eval()

    vae = vae.to(device)
    text_encoder = text_encoder.to(device)

    print("\n" + "="*70)
    print("APPLYING LORA")
    print("="*70)

    target_modules = ["to_k", "to_q", "to_v", "to_out.0"]
    lora_config = LoraConfig(
        r=config["lora_rank"],
        lora_alpha=32,
        target_modules=target_modules,
        lora_dropout=0.1,
        bias="none",
        init_lora_weights="gaussian",
    )
    transformer.add_adapter(lora_config)
    transformer.train()
    transformer = transformer.to(device, dtype=weight_dtype)

    transformer.print_trainable_parameters()

    optimizer = torch.optim.AdamW(
        transformer.parameters(),
        lr=config["learning_rate"],
        weight_decay=0.01,
    )

    print("\n" + "="*70)
    print("LOADING DATASET")
    print("="*70)

    dataset = CeramicArtifactDataset(
        image_dir=image_dir,
        descriptions_file=descriptions_file,
        image_size=config["image_size"]
    )
    dataloader = DataLoader(
        dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=0,
    )
    print(f"Dataset size: {len(dataset)}")

    print("\n" + "="*70)
    print("STARTING TRAINING")
    print("="*70)

    global_step = 0
    max_steps = config["max_train_steps"]
    progress_bar = tqdm(total=max_steps, desc="Training")

    transformer.train()
    for epoch in range(config["num_epochs"]):
        for batch_idx, batch in enumerate(dataloader):
            latents, text_embeddings = collate_fn(
                [batch[i] for i in range(len(batch))],
                vae, text_encoder, tokenizer, device
            )

            noise = torch.randn_like(latents)
            timesteps = torch.randint(
                0, 1000, (latents.shape[0],), device=device
            )

            sigma = (
                timesteps.float() / 1000
            ).view(-1, 1, 1, 1).repeat(1, 4, latents.shape[2], latents.shape[3])

            noisy_latents = latents + noise * sigma

            model_pred = transformer(
                noisy_latents,
                timesteps,
                encoder_hidden_states=text_embeddings,
            ).sample

            loss = torch.nn.functional.mse_loss(
                model_pred.float(),
                noise.float(),
                reduction="mean"
            )

            loss = loss / config["gradient_accumulation_steps"]
            loss.backward()

            if (global_step + 1) % config["gradient_accumulation_steps"] == 0:
                torch.nn.utils.clip_grad_norm_(transformer.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()

            global_step += 1
            progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})
            progress_bar.update(1)

            if global_step >= max_steps:
                break

        if global_step >= max_steps:
            break

    progress_bar.close()

    print("\n" + "="*70)
    print("SAVING MODEL")
    print("="*70)

    output_path = Path(config["output_dir"]) / "final"
    output_path.mkdir(parents=True, exist_ok=True)

    transformer.save_pretrained(str(output_path / "transformer_lora"))
    tokenizer.save_pretrained(str(output_path / "tokenizer"))

    print(f"✓ Finetuning complete! Saved to {output_path}")
    return output_path


if __name__ == "__main__":
    print("Make sure you have run: python prepare_dataset.py\n")

    output = finetune_qwen_image(
        image_dir="data/cropped_artifacts",
        descriptions_file="data/all_artifacts.json",
        config=CONFIG
    )

    print(f"\n✓ Model saved to: {output}")