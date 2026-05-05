#!/bin/bash

SCRIPT="scripts/modeling/finetune_vanilla_diffuser.py"
EVAL="scripts/modeling/evaluate_finetuned.py"
GEN_PROMPTS=(3 5 10)

evaluate_model() {
    local folder="$1"
    local label="$2"
    for g in "${GEN_PROMPTS[@]}"; do
        echo "Evaluating ${label} with --num-generated-per-prompt ${g}..."
        python "$EVAL" \
            --folder "${folder}/final" \
            --num-generated-per-prompt "$g" \
            --output-file "eval_${label}_gen${g}.json"
        echo ""
    done
}


for RANK in 16 32 64 128 256 512 1024 2048; do
    echo "=========================================="
    echo "Starting training with lora_rank=$RANK"
    echo "=========================================="

    python "$SCRIPT" \
        --output-dir "vanilla_finetuned_lora_${RANK}" \
        --lora-rank "$RANK"

    echo "Finished lora_rank=$RANK"
    evaluate_model "vanilla_finetuned_lora_${RANK}" "lora_${RANK}"
done

echo "All training runs complete."
