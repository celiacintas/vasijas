import torch
import torch.nn.functional as F
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import AutoencoderKL, UNet2DConditionModel, DDPMScheduler
from peft import PeftModel

def load_finetuned_models(checkpoint_dir="qwen_finetuned/final"):
    """Load finetuned models from checkpoint directory"""
    
    checkpoint_path = Path(checkpoint_dir)
    
    print("="*70)
    print("LOADING FINETUNED MODELS")
    print("="*70)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")
    
    # Check if checkpoint exists
    if not checkpoint_path.exists():
        print(f"❌ Checkpoint directory not found: {checkpoint_path}")
        print(f"Available checkpoints:")
        for checkpoint in Path("qwen_finetuned").glob("checkpoint_*"):
            print(f"  - {checkpoint.name}")
        return None
    
    try:
        # Load tokenizer
        print("Loading tokenizer...")
        tokenizer = CLIPTokenizer.from_pretrained(
            str(checkpoint_path / "tokenizer")
        )
        
        # Load text encoder
        print("Loading text encoder...")
        text_encoder = CLIPTextModel.from_pretrained(
            str(checkpoint_path / "text_encoder")
        )
        text_encoder = text_encoder.to(device)
        text_encoder.eval()
        
        # Load VAE
        print("Loading VAE...")
        vae = AutoencoderKL.from_pretrained(
            str(checkpoint_path / "vae")
        )
        vae = vae.to(device)
        vae.eval()
        
        # Load scheduler
        print("Loading scheduler...")
        noise_scheduler = DDPMScheduler.from_pretrained(
            str(checkpoint_path / "scheduler")
        )
        
        # Load UNet with LoRA
        print("Loading UNet...")
        unet_path = checkpoint_path / "unet_lora"
        
        if unet_path.exists():
            # Load base UNet from pretrained
            base_unet = UNet2DConditionModel.from_pretrained(
                "runwayml/stable-diffusion-v1-5",
                subfolder="unet"
            )
            
            # Load LoRA weights
            unet = PeftModel.from_pretrained(base_unet, str(unet_path))
            print("  ✓ Loaded with LoRA weights")
        else:
            # Load full UNet
            unet = UNet2DConditionModel.from_pretrained(
                str(checkpoint_path / "unet")
            )
            print("  ✓ Loaded full UNet")
        
        unet = unet.to(device)
        unet.eval()
        
        print("\n✓ All models loaded successfully!")
        
        return {
            "tokenizer": tokenizer,
            "text_encoder": text_encoder,
            "vae": vae,
            "unet": unet,
            "noise_scheduler": noise_scheduler,
            "device": device
        }
    
    except Exception as e:
        print(f"❌ Error loading models: {e}")
        import traceback
        traceback.print_exc()
        return None

def generate_images(
    prompts,
    models,
    num_inference_steps=50,
    guidance_scale=7.5,
    seed=None
):
    """Generate images from text prompts using finetuned model"""
    
    if models is None:
        print("❌ Models not loaded")
        return None
    
    tokenizer = models["tokenizer"]
    text_encoder = models["text_encoder"]
    vae = models["vae"]
    unet = models["unet"]
    noise_scheduler = models["noise_scheduler"]
    device = models["device"]
    
    if seed is not None:
        torch.manual_seed(seed)
    
    print("\n" + "="*70)
    print("GENERATING IMAGES")
    print("="*70)
    
    generated_images = []
    
    with torch.no_grad():
        for prompt_idx, prompt in enumerate(prompts, 1):
            print(f"\n[{prompt_idx}/{len(prompts)}] Prompt: {prompt}")
            
            # Encode text
            text_input = tokenizer(
                prompt,
                padding="max_length",
                max_length=tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt"
            )
            text_embeddings = text_encoder(
                text_input.input_ids.to(device)
            )[0]
            
            # Unconditional embeddings for classifier-free guidance
            uncond_input = tokenizer(
                "",
                padding="max_length",
                max_length=tokenizer.model_max_length,
                return_tensors="pt"
            )
            uncond_embeddings = text_encoder(
                uncond_input.input_ids.to(device)
            )[0]
            
            # Concatenate embeddings
            text_embeddings = torch.cat([uncond_embeddings, text_embeddings])
            
            # Initialize latents
            latents = torch.randn(
                (1, 4, 64, 64),
                device=device,
                dtype=text_embeddings.dtype
            )
            
            # Denoise
            progress_bar = tqdm(noise_scheduler.timesteps, desc="Denoising")
            
            for t in progress_bar:
                latent_model_input = torch.cat([latents] * 2)
                
                noise_pred = unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=text_embeddings
                ).sample
                
                # Classifier-free guidance
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = (
                    noise_pred_uncond +
                    guidance_scale * (noise_pred_text - noise_pred_uncond)
                )
                
                latents = noise_scheduler.step(
                    noise_pred,
                    t,
                    latents
                ).prev_sample
            
            # Decode latents to image
            latents = 1 / 0.18215 * latents
            image = vae.decode(latents).sample
            image = (image / 2 + 0.5).clamp(0, 1)
            image = image.permute(0, 2, 3, 1).cpu().numpy()[0]
            image = (image * 255).astype('uint8')
            
            # Convert to PIL Image
            pil_image = Image.fromarray(image)
            generated_images.append((prompt, pil_image))
            
            print(f"  ✓ Generated image size: {pil_image.size}")
    
    return generated_images

def save_generated_images(images, output_dir="generated_images"):
    """Save generated images to directory"""
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*70)
    print("SAVING GENERATED IMAGES")
    print("="*70)
    
    for idx, (prompt, image) in enumerate(images, 1):
        filename = f"generated_{idx:02d}.png"
        filepath = output_path / filename
        image.save(filepath)
        
        print(f"\n[{idx}] {filename}")
        print(f"  Prompt: {prompt}")
        print(f"  Saved to: {filepath}")
    
    print(f"\n✓ All images saved to {output_path}")
    return output_path

def display_image_grid(images, cols=2):
    """Display generated images in a grid"""
    
    try:
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        
        rows = (len(images) + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(15, 5*rows))
        
        if rows == 1 and cols == 1:
            axes = [axes]
        elif rows == 1:
            axes = axes.flatten()
        else:
            axes = axes.flatten()
        
        for idx, (prompt, image) in enumerate(images):
            ax = axes[idx]
            ax.imshow(image)
            ax.set_title(f"{idx+1}. {prompt}", fontsize=10, wrap=True)
            ax.axis('off')
        
        # Hide extra subplots
        for idx in range(len(images), len(axes)):
            axes[idx].axis('off')
        
        plt.tight_layout()
        plt.savefig("generated_images_grid.png", dpi=100, bbox_inches='tight')
        print("\n✓ Grid saved to generated_images_grid.png")
        plt.show()
        
    except ImportError:
        print("Matplotlib not available for display")

# Main execution
if __name__ == "__main__":
    
    # Load finetuned models
    print("Loading finetuned models...\n")
    models = load_finetuned_models(checkpoint_dir="qwen_finetuned/final")
    
    if models is None:
        print("\nTrying to find latest checkpoint...")
        checkpoints = sorted(Path("qwen_finetuned").glob("checkpoint_*"))
        if checkpoints:
            latest = checkpoints[-1]
            print(f"Found: {latest}")
            models = load_finetuned_models(checkpoint_dir=str(latest))
        else:
            print("No checkpoints found!")
            exit(1)
    
    if models is None:
        exit(1)
    
    # Example prompts from ceramic artifacts
    test_prompts = [
        "ceramic artifact with red decorative patterns and horizontal stripes",
        "pottery vessel with scalloped borders and geometric designs",
        "ceramic bowl with intricate red paint patterns and curved shapes",
        #"pottery fragment with red and white decorative motifs",
        "ceramic artifact with feathered design patterns in red",
    ]
    
    # Generate images
    generated_images = generate_images(
        test_prompts,
        models,
        num_inference_steps=50,
        guidance_scale=7.5,
        seed=42
    )
    
    if generated_images:
        # Save images
        output_dir = save_generated_images(generated_images)
        
        # Display grid
        display_image_grid(generated_images, cols=2)
        
        # Create summary
        print("\n" + "="*70)
        print("GENERATION SUMMARY")
        print("="*70)
        print(f"Total images generated: {len(generated_images)}")
        print(f"Images saved to: {output_dir}")
        print(f"Grid preview: generated_images_grid.png")