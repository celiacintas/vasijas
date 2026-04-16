# Vasijas

## Preprocessing Scripts

### Extract Images from PDFs

Extract images from PDF documents using [Docling](https://github.com/docling-project/docling).

```bash
uv run python scripts/preprocessing/extract_images.py --pdf data/thesis_texture.pdf --output output_images
```

**Arguments:**
- `--pdf`: Path to the input PDF file (default: `data/thesis_texture.pdf`)
- `--output`: Path to the output directory for extracted images (default: `output_images`)

**Examples:**
```bash
# Using default paths
uv run python scripts/preprocessing/extract_images.py

# Custom PDF path
uv run python scripts/preprocessing/extract_images.py --pdf path/to/document.pdf

# Custom output directory
uv run python scripts/preprocessing/extract_images.py --output extracted_figures
```

### Extract Individual Samples with YOLOv8

Extract individual ceramic artifacts from images using YOLOv8 segmentation with overlap removal.

> **Note:** The YOLO model (`yolov8m-seg.pt`) will be downloaded automatically on first run.

```bash
uv run python scripts/preprocessing/extract_individual_samples.py --input filtered_images --output cropped_artifacts_new
```

**Arguments:**
- `--input`: Input directory containing images (default: `filtered_images`)
- `--output`: Output directory for cropped artifacts (default: `cropped_artifacts`)
- `--iou-threshold`: IoU threshold for overlap removal 0-1 (default: `0.3`)
- `--min-width`: Minimum bounding box width in pixels (default: `50`)
- `--min-height`: Minimum bounding box height in pixels (default: `50`)
- `--min-area`: Minimum bounding box area in pixels (default: `2500`)

**Examples:**
```bash
# Using defaults
uv run python scripts/preprocessing/extract_individual_samples.py

# Custom paths with stricter overlap removal
uv run python scripts/preprocessing/extract_individual_samples.py --input my_images --output artifacts --iou-threshold 0.2

# Relaxed size constraints
uv run python scripts/preprocessing/extract_individual_samples.py --min-width 30 --min-height 30 --min-area 1000
```

### Prepare Dataset

Convert artifact descriptions to JSONL format and create comprehensive CSV for model finetuning. Combines cropped images with descriptions and verifies the dataset.

```bash
uv run python scripts/preprocessing/prepare_dataset.py
```

**What it does:**
1. Creates `all_artifacts.json` - JSONL file with image paths and descriptions
2. Creates `all_artifacts_comprehensive.csv` - CSV with full artifact metadata
3. Verifies the dataset is correctly formatted

**Default paths:**
- Images: `data/cropped_artifacts`
- Descriptions CSV: `data/artifact_descriptions.csv`
- Output JSONL: `data/all_artifacts.json`
- Output CSV: `data/all_artifacts_comprehensive.csv`

**Examples:**
```bash
# Using default paths
uv run python scripts/preprocessing/prepare_dataset.py

# Custom paths
uv run python scripts/preprocessing/prepare_dataset.py
# (Modify the paths in the script if needed)
```

## Modeling Scripts

### Finetune Vanilla Diffuser

Finetune Stable Diffusion v1.5 with LoRA on ceramic artifact images.

> **Prerequisites:** Run `prepare_dataset.py` first to prepare the dataset.

```bash
uv run python scripts/modeling/finetune_vanilla_diffuser.py
```

**Configuration options in script:**
- `model_name`: Base model (default: `runwayml/stable-diffusion-v1-5`)
- `output_dir`: Output directory (default: `vanilla_finetuned`)
- `learning_rate`: Training learning rate (default: `1e-4`)
- `batch_size`: Batch size (default: `2`)
- `num_epochs`: Number of training epochs (default: `3`)
- `image_size`: Image resolution (default: `512`)
- `use_lora`: Enable LoRA (default: `True`)
- `lora_rank`: LoRA rank (default: `16`)

**Output:**
- Checkpoints saved to `vanilla_finetuned/checkpoint_epoch_N/`
- Final model saved to `vanilla_finetuned/final/`

**Examples:**
```bash
# Using default config
uv run python scripts/modeling/finetune_vanilla_diffuser.py

# Modify CONFIG in script for custom settings
```

### Generate from Finetuned Model

Generate new ceramic artifact images using the finetuned model.

```bash
uv run python scripts/modeling/generate_from_finetuned.py
```

**How it works:**
1. Loads the finetuned model from `vanilla_finetuned/final/`
2. Generates images from text prompts using classifier-free guidance
3. Saves generated images to `generated_images/`
4. Creates a grid preview of all generated images

**Arguments:**
- `checkpoint_dir`: Path to finetuned checkpoint (default: `vanilla_finetuned/final`)
- `num_inference_steps`: Denoising steps (default: `50`)
- `guidance_scale`: Guidance scale for CFG (default: `7.5`)
- `seed`: Random seed for reproducibility

**Examples:**
```bash
# Using default paths
uv run python scripts/modeling/generate_from_finetuned.py

# Edit the prompts in the script for custom generation
```
