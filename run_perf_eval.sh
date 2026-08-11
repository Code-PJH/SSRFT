perf_eval()
{
        MODEL_NAME=$1
        MODEL_PATH=$2
        OUTPUT_DIR="eval/perf/${MODEL_NAME}"
        echo "Starting performance evaluation for model: ${MODEL_NAME}"
        echo "Model path: ${MODEL_PATH}"
        lm_eval --model hf \
                --model_args pretrained=${MODEL_PATH},trust_remote_code=True \
                --output_path "${OUTPUT_DIR}" \
                --log_samples \
                --batch_size "auto" \
                --trust_remote_code \
                --tasks arc_easy,arc_challenge,sciq,logiqa,commonsense_qa,piqa,winogrande,boolq,mmlu
}

# Example:
perf_eval Qwen3-4B-Instruct Qwen/Qwen3-4B-Instruct-2507
