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

Finetune Stable Diffusion v1.5 with LoRA on ceramic artifact images. Splits data into train/test sets and evaluates on test set each epoch.

> **Prerequisites:** Run `prepare_dataset.py` first to prepare the dataset.

```bash
uv run python scripts/modeling/finetune_vanilla_diffuser.py --output-dir my_finetuned_model
```

**CLI Arguments:**
- `--output-dir`: Name of the model output folder (default: `vanilla_finetuned`)
- `--model-name`: Base model name or HuggingFace path (default: `runwayml/stable-diffusion-v1-5`)
- `--steps-per-epoch`: Limit training steps per epoch; `None` for all batches (default: `None`)
- `--train-ratio`: Ratio of data for training (default: `0.8`)
- `--lora-rank`: LoRA rank for fine-tuning (default: `16`)

**Configuration defaults in script (edit directly for non-CLI params):**
- `learning_rate`: `1e-5`
- `batch_size`: `16`
- `num_epochs`: `5`
- `image_size`: `256`
- `use_lora`: `True`
- `lora_rank`: `16`
- `gpu`: `0`

**Examples:**
```bash
# Using all defaults
uv run python scripts/modeling/finetune_vanilla_diffuser.py

# Custom output and model
uv run python scripts/modeling/finetune_vanilla_diffuser.py --output-dir sd21_finetuned --model-name stabilityai/stable-diffusion-2-1

# Quick test run (10 steps per epoch)
uv run python scripts/modeling/finetune_vanilla_diffuser.py --steps-per-epoch 10

# Use 90% of data for training
uv run python scripts/modeling/finetune_vanilla_diffuser.py --train-ratio 0.9

# Sweep LoRA ranks with different output directories
bash run_lora_sweep.sh
```

**Output:**
- Checkpoints saved to `<output_dir>/checkpoint_epoch_N/`
- Final model saved to `<output_dir>/final/`
- Training loss plot saved to `<output_dir>/training_loss.png`
- Training log saved to `<output_dir>/training_log.json` (config + per-epoch train/test losses)
- Denoising sequence images saved to `<output_dir>/denoise_00.png` through `denoise_03.png`

### Generate from Finetuned Model

Generate new ceramic artifact images using a finetuned model or any base Stable Diffusion model from HuggingFace. Uses three curated prompts describing Iberian ceramic styles (geometric decoration, concentric designs, vessel patterns).

```bash
uv run python scripts/modeling/generate_from_finetuned.py
```

**CLI Arguments:**
- `--checkpoint-dir`: Path to finetuned checkpoint directory. If `None`, loads base model from HuggingFace (default: `None`)
- `--model-name`: Base model name from HuggingFace (default: `runwayml/stable-diffusion-v1-5`)

**Examples:**
```bash
# Generate from finetuned model
uv run python scripts/modeling/generate_from_finetuned.py --checkpoint-dir vanilla_finetuned/final

# Generate from base HuggingFace model (no finetuning)
uv run python scripts/modeling/generate_from_finetuned.py --model-name runwayml/stable-diffusion-v1-5

# Use a different checkpoint
uv run python scripts/modeling/generate_from_finetuned.py --checkpoint-dir vanilla_finetuned_lora_64/final
```

**Output:**
- Individual images saved to `generated_images/<prefix>generated_01.png`, etc.
- Grid preview saved to `generated_images/<prefix>generated_images_grid.png`
- The `<prefix>` is derived from the checkpoint directory name (e.g., `vanilla_`, `small_`) or the base model name.
