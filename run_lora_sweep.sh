#!/bin/bash

SCRIPT="scripts/modeling/finetune_vanilla_diffuser.py"

for RANK in 16 64 128 256 512 1028 2046; do
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
