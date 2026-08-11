# Example:
# train qwen3-4B-instruct
python -m train.train \
    --model-name-or-path Qwen/Qwen3-4B-Instruct-2507 \
    --data-dir sft_data \
    --output-dir /{output_path}/Qwen3-4B-SSRFT-bf16 \
    --val-ratio 0.1 \
    --num-train-epochs 10 \
    --per-device-train-batch-size 32 \
    --gradient-accumulation-steps 2 \
    --per-device-eval-batch-size 16 \
    --learning-rate 1e-5 \
    --eval-strategy "epoch" \
    --eval-steps 1 \
    --save-strategy "epoch" \
    --save-steps 1 \
    --logging-steps 1\
    --load-best-model-at-end \
    --bf16 \
    --save-as-bf16 \
    --gradient-checkpointing \
    --early-stopping-patience 1 \
    --logging_dir ./logs/Qwen3-4B-SSRFT-bf16

# train qwen3-4B-no_character-bf16
python -m train.train \
    --model-name-or-path Qwen/Qwen3-4B-Instruct-2507 \
    --data-dir sft_data_no_character \
    --output-dir /{output_path}/Qwen3-4B-no_character-bf16 \
    --val-ratio 0.1 \
    --num-train-epochs 10 \
    --per-device-train-batch-size 3 \
    --gradient-accumulation-steps 9 \
    --per-device-eval-batch-size 4 \
    --learning-rate 1e-5 \
    --eval-strategy "epoch" \
    --eval-steps 1 \
    --save-strategy "epoch" \
    --save-steps 1 \
    --logging-steps 1\
    --load-best-model-at-end \
    --bf16 \
    --save-as-bf16 \
    --gradient-checkpointing \
    --early-stopping-patience 1 \
    --logging_dir ./logs/Qwen3-4B-no_character-bf16

# EXPAND
# train qwen3-4B-instruct
python -m train.train \
    --model-name-or-path Qwen/Qwen3-4B-Instruct-2507 \
    --data-dir sft_data_expand \
    --output-dir /{expand_output_path}/Qwen3-4B-SSRFT-bf16 \
    --val-ratio 0.1 \
    --num-train-epochs 10 \
    --per-device-train-batch-size 32 \
    --gradient-accumulation-steps 2 \
    --per-device-eval-batch-size 16 \
    --learning-rate 1e-5 \
    --eval-strategy "epoch" \
    --eval-steps 1 \
    --save-strategy "epoch" \
    --save-steps 1 \
    --logging-steps 1\
    --load-best-model-at-end \
    --bf16 \
    --save-as-bf16 \
    --gradient-checkpointing \
    --early-stopping-patience 1 \
    --logging_dir ./logs_expand/Qwen3-4B-SSRFT-bf16

# train qwen3-4B-no_character-bf16
python -m train.train \
    --model-name-or-path Qwen/Qwen3-4B-Instruct-2507 \
    --data-dir sft_data_no_character_expand \
    --output-dir /{expand_output_path}/Qwen3-4B-no_character-bf16 \
    --val-ratio 0.1 \
    --num-train-epochs 10 \
    --per-device-train-batch-size 11 \
    --gradient-accumulation-steps 2 \
    --per-device-eval-batch-size 4 \
    --learning-rate 1e-5 \
    --eval-strategy "epoch" \
    --eval-steps 1 \
    --save-strategy "epoch" \
    --save-steps 1 \
    --logging-steps 1\
    --load-best-model-at-end \
    --bf16 \
    --save-as-bf16 \
    --gradient-checkpointing \
    --early-stopping-patience 1 \
    --logging_dir ./logs_expand/Qwen3-4B-no_character-bf16
