#!/bin/bash

SCRIPT="scripts/modeling/finetune_vanilla_diffuser.py"

echo "=========================================="
echo "Starting full fine-tuning (no LoRA)"
echo "=========================================="

python "$SCRIPT" \
    --output-dir "vanilla_finetuned_full" \
    --lora-rank 0

echo "Finished full fine-tuning"
echo ""

for RANK in 16 64 128 256 512 1024 2048; do
    echo "=========================================="
    echo "Starting training with lora_rank=$RANK"
    echo "=========================================="
    
    python "$SCRIPT" \
        --output-dir "vanilla_finetuned_lora_${RANK}" \
        --lora-rank "$RANK"
    
    echo "Finished lora_rank=$RANK"
    echo ""
done

echo "All training runs complete."
