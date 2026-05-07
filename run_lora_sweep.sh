#!/bin/bash

SCRIPT="scripts/modeling/finetune_vanilla_diffuser.py"
UNCOND_SCRIPT="scripts/modeling/baselines/finetune_unconditional_diffuser.py"

echo "=========================================="
echo "=== TEXT-CONDITIONED (vanilla_diffuser) ==="
echo "=========================================="

for RANK in 16 32 64 128 256 512 1024 2048; do
    echo "=========================================="
    echo "Starting training with lora_rank=$RANK"
    echo "=========================================="

    uv python "$SCRIPT" \
        --output-dir "vanilla_finetuned_lora_${RANK}" \
        --lora-rank "$RANK"

    echo "Finished lora_rank=$RANK"
done

echo "=========================================="
echo "=== UNCONDITIONAL (unconditional_diffuser) ==="
echo "=========================================="

for RANK in 16 32 64 128 256 512 1024 2048; do
    echo "=========================================="
    echo "Starting unconditional training with lora_rank=$RANK"
    echo "=========================================="

    uv python "$UNCOND_SCRIPT" \
        --output-dir "vanilla_uncond_lora_${RANK}" \
        --lora-rank "$RANK"

    echo "Finished unconditional lora_rank=$RANK"
done

echo "All training runs complete."

