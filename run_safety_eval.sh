#!/bin/bash

# Define the safety evaluation function, Using the MD-Judge-v0.1 (default)
run_safety_eval() {
    MODEL_NAME=$1
    MODEL_PATH=$2
    MAX_NEW_TOKENS=${3:-512}
    BATCH_SIZE=${4:-48}
    OUTPUT_DIR="eval/safety/${MODEL_NAME}"
    echo "Starting safety evaluation for model: ${MODEL_NAME}"
    echo "Model path: ${MODEL_PATH}"
    
    # 1. forbidden_question_set.csv
    echo "Running evaluation on forbidden_question_set.csv..."
    python -m eval.safety_eval \
        --input test_datasets/jailbreak_llms/forbidden_question_set.csv \
        --output "${OUTPUT_DIR}/forbidden_question_set.json" \
        --batch-size ${BATCH_SIZE} \
        --judge-batch-size 16 \
        --target-max-new-tokens ${MAX_NEW_TOKENS} \
        --target-model "${MODEL_PATH}"

    # 2. forbidden_question_set_with_prompts_sampled_1170.csv
    echo "Running evaluation on forbidden_question_set_with_prompts_sampled_1170.csv..."
    python -m eval.safety_eval \
        --input test_datasets/jailbreak_llms/forbidden_question_set_with_prompts_sampled_1170.csv \
        --output "${OUTPUT_DIR}/forbidden_question_set_with_prompts.json" \
        --batch-size $((BATCH_SIZE / 2)) \
        --judge-batch-size 16 \
        --target-max-new-tokens ${MAX_NEW_TOKENS} \
        --target-model "${MODEL_PATH}"

    # 3. JailBench-seed.csv
    echo "Running evaluation on JailBench-seed.csv..."
    python -m eval.safety_eval \
        --input test_datasets/JailBench/JailBench-seed.csv \
        --output "${OUTPUT_DIR}/JailBench-seed.json" \
        --batch-size ${BATCH_SIZE} \
        --judge-batch-size 16 \
        --target-max-new-tokens ${MAX_NEW_TOKENS} \
        --target-model "${MODEL_PATH}"

    # 4. JailBench_sampled_1080.csv
    echo "Running evaluation on JailBench_sampled_1080.csv..."
    python -m eval.safety_eval \
        --input test_datasets/JailBench/JailBench_sampled_1080.csv \
        --output "${OUTPUT_DIR}/JailBench.json" \
        --batch-size $((BATCH_SIZE / 2)) \
        --judge-batch-size 16 \
        --target-max-new-tokens ${MAX_NEW_TOKENS} \
        --target-model "${MODEL_PATH}"
    
    # 5. AdvBench
    echo "Running evaluation on AdvBench..."
    python -m eval.safety_eval \
        --input test_datasets/AdvBench/train.csv \
        --output "${OUTPUT_DIR}/AdvBench.json" \
        --batch-size ${BATCH_SIZE} \
        --judge-batch-size 16 \
        --target-max-new-tokens ${MAX_NEW_TOKENS} \
        --target-model "${MODEL_PATH}"

    echo "Finished safety evaluation for model: ${MODEL_NAME}"
    echo "----------------------------------------"
}

# Define the safety evaluation function, Using the MD-Judge-v0.2
run_safety_eval_v2() {
    MODEL_NAME=$1
    MODEL_PATH=$2
    OUTPUT_DIR="eval/safety/${MODEL_NAME}"
    MAX_NEW_TOKENS=${3:-512}
    BATCH_SIZE=${4:-48}
    echo "Starting safety evaluation for model: ${MODEL_NAME}"
    echo "Model path: ${MODEL_PATH}"
    
    # 1. forbidden_question_set.csv
    echo "Running evaluation on forbidden_question_set.csv..."
    python -m eval.safety_eval \
        --input test_datasets/jailbreak_llms/forbidden_question_set.csv \
        --resp-log-path "${OUTPUT_DIR}/forbidden_question_set.json" \
        --output "${OUTPUT_DIR}/v2/forbidden_question_set.json" \
        --batch-size ${BATCH_SIZE} \
        --judge-batch-size 8 \
        --judge-model "OpenSafetyLab/MD-Judge-v0_2-internlm2_7b" \
        --target-max-new-tokens ${MAX_NEW_TOKENS} \
        --target-model "${MODEL_PATH}"

    # 2. forbidden_question_set_with_prompts_sampled_1170.csv
    echo "Running evaluation on forbidden_question_set_with_prompts_sampled_1170.csv..."
    python -m eval.safety_eval \
        --input test_datasets/jailbreak_llms/forbidden_question_set_with_prompts_sampled_1170.csv \
        --resp-log-path "${OUTPUT_DIR}/forbidden_question_set_with_prompts.json" \
        --output "${OUTPUT_DIR}/v2/forbidden_question_set_with_prompts.json" \
        --batch-size $((BATCH_SIZE / 2)) \
        --judge-batch-size 4 \
        --judge-model "OpenSafetyLab/MD-Judge-v0_2-internlm2_7b" \
        --target-max-new-tokens ${MAX_NEW_TOKENS} \
        --target-model "${MODEL_PATH}"

    # 3. JailBench-seed.csv
    echo "Running evaluation on JailBench-seed.csv..."
    python -m eval.safety_eval \
        --input test_datasets/JailBench/JailBench-seed.csv \
        --resp-log-path "${OUTPUT_DIR}/JailBench-seed.json" \
        --output "${OUTPUT_DIR}/v2/JailBench-seed.json" \
        --batch-size ${BATCH_SIZE} \
        --judge-batch-size 8 \
        --judge-model "OpenSafetyLab/MD-Judge-v0_2-internlm2_7b" \
        --target-max-new-tokens ${MAX_NEW_TOKENS} \
        --target-model "${MODEL_PATH}"

    # 4. JailBench_sampled_1080.csv
    echo "Running evaluation on JailBench_sampled_1080.csv..."
    python -m eval.safety_eval \
        --input test_datasets/JailBench/JailBench_sampled_1080.csv \
        --resp-log-path "${OUTPUT_DIR}/JailBench.json" \
        --output "${OUTPUT_DIR}/v2/JailBench.json" \
        --batch-size $((BATCH_SIZE / 2)) \
        --judge-batch-size 4 \
        --judge-model "OpenSafetyLab/MD-Judge-v0_2-internlm2_7b" \
        --target-max-new-tokens ${MAX_NEW_TOKENS} \
        --target-model "${MODEL_PATH}"
        
    # 5. AdvBench
    echo "Running evaluation on AdvBench..."
    python -m eval.safety_eval \
        --input test_datasets/AdvBench/train.csv \
        --resp-log-path "${OUTPUT_DIR}/AdvBench.json" \
        --output "${OUTPUT_DIR}/v2/AdvBench.json" \
        --batch-size ${BATCH_SIZE} \
        --judge-batch-size 8 \
        --judge-model "OpenSafetyLab/MD-Judge-v0_2-internlm2_7b" \
        --target-max-new-tokens ${MAX_NEW_TOKENS} \
        --target-model "${MODEL_PATH}"

    echo "Finished safety evaluation for model: ${MODEL_NAME}"
    echo "----------------------------------------"
}

# Define the safety evaluation function under prefilling attack, Using the MD-Judge-v0.1 (default)
run_safety_eval_prefilling_attack() {
    MODEL_NAME=$1
    MODEL_PATH=$2
    OUTPUT_DIR="eval/safety/${MODEL_NAME}/prefilling_attack"
    MAX_NEW_TOKENS=${3:-512}
    BATCH_SIZE=${4:-48}
    echo "Starting safety evaluation for model: ${MODEL_NAME}"
    echo "Model path: ${MODEL_PATH}"
    
    # 1. forbidden_question_set.csv
    echo "Running evaluation on forbidden_question_set.csv..."
    python -m eval.safety_eval \
        --input test_datasets/jailbreak_llms/forbidden_question_set.csv \
        --output "${OUTPUT_DIR}/forbidden_question_set.json" \
        --batch-size ${BATCH_SIZE} \
        --judge-batch-size 16 \
        --target-model "${MODEL_PATH}" \
        --target-max-new-tokens ${MAX_NEW_TOKENS} \
        --prefilling-attack
        

    # 2. JailBench-seed.csv
    echo "Running evaluation on JailBench-seed.csv..."
    python -m eval.safety_eval \
        --input test_datasets/JailBench/JailBench-seed.csv \
        --output "${OUTPUT_DIR}/JailBench-seed.json" \
        --batch-size ${BATCH_SIZE} \
        --judge-batch-size 16 \
        --target-model "${MODEL_PATH}" \
        --target-max-new-tokens ${MAX_NEW_TOKENS} \
        --prefilling-attack

    # 3. AdvBench
    echo "Running evaluation on AdvBench..."
    python -m eval.safety_eval \
        --input test_datasets/AdvBench/train.csv \
        --output "${OUTPUT_DIR}/AdvBench.json" \
        --batch-size ${BATCH_SIZE} \
        --judge-batch-size 16 \
        --target-model "${MODEL_PATH}" \
        --target-max-new-tokens ${MAX_NEW_TOKENS} \
        --prefilling-attack

    echo "Finished safety evaluation for model: ${MODEL_NAME}"
    echo "----------------------------------------"
}

# Define the safety evaluation function under prefilling attack, Using the MD-Judge-v0.2
run_safety_eval_prefilling_attack_v2() {
    MODEL_NAME=$1
    MODEL_PATH=$2
    OUTPUT_DIR="eval/safety/${MODEL_NAME}/prefilling_attack"
    MAX_NEW_TOKENS=${3:-512}
    BATCH_SIZE=${4:-48}
    echo "Starting safety evaluation for model: ${MODEL_NAME}"
    echo "Model path: ${MODEL_PATH}"
    
    # 1. forbidden_question_set.csv
    echo "Running evaluation on forbidden_question_set.csv..."
    python -m eval.safety_eval \
        --input test_datasets/jailbreak_llms/forbidden_question_set.csv \
        --resp-log-path "${OUTPUT_DIR}/forbidden_question_set.json" \
        --output "${OUTPUT_DIR}/v2/forbidden_question_set.json" \
        --batch-size ${BATCH_SIZE} \
        --judge-batch-size 8 \
        --judge-model "OpenSafetyLab/MD-Judge-v0_2-internlm2_7b" \
        --target-model "${MODEL_PATH}" \
        --target-max-new-tokens ${MAX_NEW_TOKENS} \
        --prefilling-attack

    # 2. JailBench-seed.csv
    echo "Running evaluation on JailBench-seed.csv..."
    python -m eval.safety_eval \
        --input test_datasets/JailBench/JailBench-seed.csv \
        --resp-log-path "${OUTPUT_DIR}/JailBench-seed.json" \
        --output "${OUTPUT_DIR}/v2/JailBench-seed.json" \
        --batch-size ${BATCH_SIZE} \
        --judge-batch-size 8 \
        --judge-model "OpenSafetyLab/MD-Judge-v0_2-internlm2_7b" \
        --target-model "${MODEL_PATH}" \
        --target-max-new-tokens ${MAX_NEW_TOKENS} \
        --prefilling-attack
        
    # 3. AdvBench
    echo "Running evaluation on AdvBench..."
    python -m eval.safety_eval \
        --input test_datasets/AdvBench/train.csv \
        --resp-log-path "${OUTPUT_DIR}/AdvBench.json" \
        --output "${OUTPUT_DIR}/v2/AdvBench.json" \
        --batch-size ${BATCH_SIZE} \
        --judge-batch-size 8 \
        --judge-model "OpenSafetyLab/MD-Judge-v0_2-internlm2_7b" \
        --target-model "${MODEL_PATH}" \
        --target-max-new-tokens ${MAX_NEW_TOKENS} \
        --prefilling-attack

    echo "Finished safety evaluation for model: ${MODEL_NAME}"
    echo "----------------------------------------"
}

# Example:
# run_safety_eval Qwen3-4B-Instruct Qwen/Qwen3-4B-Instruct-2507 512
run_safety_eval_v2 Qwen3-4B-Instruct Qwen/Qwen3-4B-Instruct-2507 512
# run_safety_eval_prefilling_attack Qwen3-4B-Instruct Qwen/Qwen3-4B-Instruct-2507 512
run_safety_eval_prefilling_attack_v2 Qwen3-4B-Instruct Qwen/Qwen3-4B-Instruct-2507 512
