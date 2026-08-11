BATCH_SIZE=32
# Define the safety evaluation function
run_refusal_eval() {
    MODEL_NAME=$1
    MODEL_PATH=$2
    OUTPUT_DIR="eval/refusal/${MODEL_NAME}"
    JUDGE_MODEL='qwen3-max'

    echo "Starting safety evaluation for model: ${MODEL_NAME}"
    echo "Model path: ${MODEL_PATH}"
    
    # 1. forbidden_question_set.csv
    echo "Running refusal evaluation."
    # Use python environment variables for API keys
    python -m eval.refusal_eval \
        --input test_datasets/xstest_safe.csv \
        --output "${OUTPUT_DIR}/xstest_safe.json" \
        --batch-size  ${BATCH_SIZE}\
        --target-model "${MODEL_PATH}" \
        --judge-model ${JUDGE_MODEL} \
        
    echo "Finished safety evaluation for model: ${MODEL_NAME}"
    echo "----------------------------------------"
}


# Example:
run_refusal_eval Qwen3-4B-Instruct Qwen/Qwen3-4B-Instruct-2507
