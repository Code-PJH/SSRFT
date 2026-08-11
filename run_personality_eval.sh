BATCH_SIZE=32
run_personality_eval() {
    MODEL_NAME=$1
    MODEL_PATH=$2
    OUTPUT_DIR="eval/personality/${MODEL_NAME}"

    echo "Starting personality evaluation for model: ${MODEL_NAME}"
    echo "Model path: ${MODEL_PATH}"
    
    echo "Running BFI-44 evaluation."
    python -m eval.bfi_eval \
        --input test_datasets/BFI-44.csv \
        --output "${OUTPUT_DIR}/bfi_results.json" \
        --batch-size ${BATCH_SIZE} \
        --target-model "${MODEL_PATH}"
        
    echo "Finished personality evaluation for model: ${MODEL_NAME}"
    echo "----------------------------------------"
}

# Exapmle:
run_personality_eval Qwen3-4B-Instruct Qwen/Qwen3-4B-Instruct-2507
