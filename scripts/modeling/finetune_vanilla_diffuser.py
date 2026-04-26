import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from pathlib import Path
from PIL import Image
import json
from tqdm import tqdm
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import AutoencoderKL, UNet2DConditionModel, DDPMScheduler
from peft import get_peft_model, LoraConfig, TaskType

# Configuration
CONFIG = {
    "model_name": "runwayml/stable-diffusion-v1-5",
    "output_dir": "vanilla_finetuned",
    "learning_rate": 1e-4,
    "batch_size": 2,
    "num_epochs": 5,
    "image_size": 512,
    "use_lora": True,
    "lora_rank": 16,
    "gpu": 0,
}

torch.cuda.set_device(CONFIG["gpu"])

class CeramicArtifactDataset(Dataset):
    """Dataset for ceramic artifacts with descriptions"""
    
    def __init__(self, image_dir, descriptions_file, image_size=512):
        self.image_dir = Path(image_dir)
        self.image_size = image_size
        self.descriptions = {}
        
        # Load descriptions from JSONL
        if Path(descriptions_file).exists():
            with open(descriptions_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        try:
                            data = json.loads(line)
                            self.descriptions[data['filename']] = data['description']
                        except json.JSONDecodeError:
                            continue
        
        # Get all images
        print(list(self.image_dir.glob("*.png")))
        self.image_paths = sorted(self.image_dir.glob("*.png"))
        print(f"Loaded {len(self.image_paths)} images")
        print(f"Loaded {len(self.descriptions)} descriptions")
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        
        try:
            # Load and resize image
            image = Image.open(image_path).convert("RGB")
            image = image.resize((self.image_size, self.image_size), Image.Resampling.LANCZOS)
            
            # Convert to tensor and normalize
            image_array = torch.from_numpy(
                __import__('numpy').array(image)
            ).permute(2, 0, 1).float()
            image_array = image_array / 127.5 - 1  # Normalize to [-1, 1]
            
            # Get description
            description = self.descriptions.get(
                image_path.name,
                "ceramic artifact with decorative patterns"
            )
            
            return {
                "image": image_array,
                "text": description,
                "filename": image_path.name
            }
        except Exception as e:
            print(f"Error loading {image_path}: {e}")
            # Return dummy data
            return {
                "image": torch.randn(3, self.image_size, self.image_size),
                "text": "ceramic artifact",
                "filename": image_path.name
            }

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
            lora_alpha=32,
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
        pin_memory=True if device.type == "cuda" else False
    )
    
    print(f"Dataset size: {len(dataset)}")
    print(f"Batch size: {config['batch_size']}")
    print(f"Number of batches: {len(dataloader)}")
    
    # Training loop
    print("\n" + "="*70)
    print("STARTING TRAINING")
    print("="*70)
    
    unet.train()
    
    for epoch in range(config["num_epochs"]):
        print(f"\n[Epoch {epoch + 1}/{config['num_epochs']}]")
        
        progress_bar = tqdm(dataloader, desc="Training")
        total_loss = 0
        
        for batch_idx, batch in enumerate(progress_bar):
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
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(unet.parameters(), 1.0)
            optimizer.step()
            
            total_loss += loss.item()
            progress_bar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "avg_loss": f"{total_loss / (batch_idx + 1):.4f}"
            })
        
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