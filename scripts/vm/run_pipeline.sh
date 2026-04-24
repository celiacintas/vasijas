#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo "=== Downloading data ==="
python "$PROJECT_ROOT/scripts/vm/download_folder.py" "$1" "$PROJECT_ROOT/data/cropped_artifacts"

echo "=== Running finetune ==="
python "$PROJECT_ROOT/scripts/modeling/finetune_vanilla_diffuser.py"

echo "=== Generating ==="
python "$PROJECT_ROOT/scripts/modeling/generate_from_finetuned.py"