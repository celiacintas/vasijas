"""Download only the required subfolders of HuggingFace models to models/."""

from pathlib import Path
from huggingface_hub import snapshot_download

REPO_DIR = Path(__file__).resolve().parent.parent.parent
MODELS_DIR = REPO_DIR / "models"

# Only the subfolders required by finetune_vanilla_diffuser.py
REQUIRED_PATTERNS = [
    "model_index.json",
    "tokenizer/**",
    "text_encoder/**",
    "vae/**",
    "unet/**",
    "scheduler/**",
]

MODELS = [
    "runwayml/stable-diffusion-v1-5",
]

for model_id in MODELS:
    local_dir = MODELS_DIR / model_id
    print(f"\nDownloading {model_id} → {local_dir}")
    snapshot_download(
        repo_id=model_id,
        local_dir=str(local_dir),
        allow_patterns=REQUIRED_PATTERNS
    )
    print(f"Done: {model_id}")

print("\nAll models downloaded.")
