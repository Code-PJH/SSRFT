# SSRFT: Supervised Safe-Role Fine-Tuning

Official implementation for **Beyond Refusal Patterns: Safe-Role Internalization for Robust and Efficient LLM Safety Alignment**.

SSRFT constructs a Safe-Role Question-Answer (SRQA) dataset from psychometric questions, a small set of jailbreak prompts, and a predefined safe-role description. The generated responses are validated and optionally expanded into scenario-based interactions before supervised fine-tuning.

> **Content warning:** This repository contains safety-evaluation and jailbreak datasets that may include harmful or toxic language.

## Repository layout

```text
.
├── data_collection/              # SRQA collection, expansion, and quality evaluation
├── eval/                         # Safety, refusal, personality, and judge implementations
├── train/                        # Full-parameter and LoRA/DoRA training
├── utils/                        # Model wrappers and data helpers
├── sft_data/                     # Non-expanded SRQA training data
├── sft_data_expand/              # Expanded SRQA training data
├── sft_data_no_character/        # Non-expanded standard-SFT baseline data
├── sft_data_no_character_expand/ # Expanded standard-SFT baseline data
├── test_datasets/                # Evaluation datasets and split files
├── train.sh                      # Paper training examples
└── run_*_eval.sh                 # Paper evaluation examples
```

The analysis notebooks are intentionally excluded from this release because they depended on experiment-local result directories. The evaluation scripts produce JSON files that can be analyzed with paths appropriate to the local setup.

## JailBench data

Due to the JailBench dataset's usage requirements, the complete dataset is not redistributed in this repository. We retain the dataset-splitting script used in our experiments. To obtain the original dataset, please contact the [JailBench research team](https://github.com/STAIR-BUPT/JailBench/tree/main).

After obtaining `JailBench-seed.csv` and `JailBench.csv`, place them under `test_datasets/JailBench/` and run:

```bash
python jailbench_split.py
```

The script converts the original headers to `q_id, question, 一级领域, content_policy_name`, writes normalized copies of the two source CSVs to the output directory, and then reproduces the train, ID-test, OOD-test, combined-test, and 1,080-example evaluation files used in the paper. With the default arguments, the normalized source files replace the originals in place.

See [JailBench dataset setup](test_datasets/JailBench/README.md) for the expected source files, exact splitting procedure, and generated filenames.

## Installation

The code is intended to be run on Linux with Python 3.10+ and a CUDA-capable GPU.
All commands below assume the current directory is the repository root (the directory containing this README).

```bash
conda create -n ssrft python=3.10 -y
conda activate ssrft
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If the default PyTorch wheel does not match the local CUDA installation, install the appropriate PyTorch build first and then install the remaining requirements.

Some data-generation and refusal-evaluation steps call an OpenAI-compatible API. Copy the example configuration and fill in local credentials:

```bash
cp .env.example .env
```

Never commit or redistribute the resulting `.env` file.

## Training-data variants and naming

The paper uses the following terminology. The `no_character` directory name is retained for compatibility with the experiment files, but it denotes the standard SFT baseline in the paper.

| Paper setting | Training directory | Example model output name |
|---|---|---|
| Non-expanded SSRFT | `sft_data` | `Qwen3-4B-SSRFT-bf16` |
| Non-expanded SFT | `sft_data_no_character` | `Qwen3-4B-no_character-bf16` |
| Expanded SSRFT | `sft_data_expand` | `Qwen3-4B-SSRFT-bf16` |
| Expanded SFT | `sft_data_no_character_expand` | `Qwen3-4B-no_character-bf16` |

Expanded and non-expanded runs intentionally use the same model basename in the original experiments, so they must be placed under different output roots. Paths are case-sensitive on Linux; use `Qwen3-4B`, not `Qwen3-4b`.

## Data collection and expansion

Run Python modules from this directory so that their relative imports and data paths resolve correctly.

Generate one SRQA component:

```bash
python -m data_collection.data_collection --dataset mbti
```

Available selections are `mbti`, `wvs`, `ipip`, `jailbreak`, `jailbench`, `jailbreak_no_character`, `jailbench_no_character`, and `all`. Selecting `all` launches every configured API-backed collection job and can be expensive.

Expand one psychometric component:

```bash
python -m data_collection.data_expand --dataset wvs
```

Available expansion selections are `mbti`, `wvs`, `ipip`, and `all`. To interactively revisit failed expansion samples, add `--process-failures`:

```bash
python -m data_collection.data_expand --dataset all --process-failures
```

The principal file chain is:

| Collection output | Expansion output |
|---|---|
| `sft_data/mbti_QA.json` | `sft_data_expand/Expanded_mbti_QA.json` |
| `sft_data/WVS_7_QA.json` | `sft_data_expand/Expanded_WVS_7_QA.json` |
| `sft_data/IPIP_NEO_QA.json` | `sft_data_expand/Expanded_IPIP_NEO_QA.json` |
| `sft_data/jailbreak_llms_QA.json` | Used directly in the training directory |
| `sft_data/JailBench_QA.json` | Used directly in the training directory |

`sft_data_no_character_expand` is a preassembled, size-matched baseline containing sampled external instruction data and refusal data. This release includes the resulting training artifacts but not a script that reproduces the external sampling step.

## Training

`train.sh` records the four Qwen3-4B training configurations used in the experiments. It is an editable paper-reproduction script, not a machine-independent launcher.

Before running it:

1. Replace the literal `/{output_path}` and `/{expand_output_path}` placeholders with local output roots.
2. Review batch sizes for the available GPU memory.
3. Comment out any experiment blocks that should not run. All four examples are active in the supplied script.
4. Run from the release directory:

```bash
bash train.sh
```

For a single custom run, call the Python module directly. For example:

```bash
python -m train.train \
  --model-name-or-path Qwen/Qwen3-4B-Instruct-2507 \
  --data-dir sft_data_expand \
  --output-dir /path/to/models/Qwen3-4B-SSRFT-bf16 \
  --val-ratio 0.1 \
  --num-train-epochs 10 \
  --per-device-train-batch-size 32 \
  --gradient-accumulation-steps 2 \
  --per-device-eval-batch-size 16 \
  --learning-rate 1e-5 \
  --eval-strategy epoch \
  --save-strategy epoch \
  --load-best-model-at-end \
  --bf16 \
  --save-as-bf16 \
  --gradient-checkpointing \
  --early-stopping-patience 1
```

`train/train-lora.py` accepts the same core arguments and additionally supports `--lora-r`, `--lora-alpha`, `--lora-dropout`, and `--use-dora`.

## Evaluation

The four shell files are editable examples. Set each model path and result directory for the local environment, then keep only the desired function calls active.

```bash
bash run_personality_eval.sh
bash run_refusal_eval.sh
bash run_safety_eval.sh
bash run_perf_eval.sh
```

The expected output names are:

| Evaluation | Default result root | Output consumed downstream |
|---|---|---|
| Personality (BFI-44) | `eval/personality/<MODEL_NAME>` | `bfi_results.json` |
| Harmless-query refusal | `eval/refusal/<MODEL_NAME>` | `xstest_safe.json` |
| General performance | `eval/perf/<MODEL_NAME>` | lm-eval `results_*.json` files |
| Safety | `eval/safety/<MODEL_NAME>` | See below |

Safety evaluation writes the following dataset-level files:

- `forbidden_question_set.json`
- `forbidden_question_set_with_prompts.json`
- `JailBench-seed.json`
- `JailBench.json`
- `AdvBench.json`

MD-Judge v0.2 outputs are stored under `v2/`; prefilling-attack outputs are stored under `prefilling_attack/`, with v0.2 results under `prefilling_attack/v2/`.

The v0.2 functions first try to reuse target-model responses from the corresponding v0.1 result through `--resp-log-path`. If that file is absent or cannot be loaded, `eval.safety_eval` falls back to normal target-model generation. Therefore v0.2 can either follow v0.1 to avoid duplicate generation or run independently.

### Important: filter safety results by the official splits

> **IMPORTANT:** Raw results must be filtered with the provided dataset splits
> before computing accurate test/ID/OOD metrics.
> The unfiltered `overall_stats` field is not a held-out test result unless
> the input CSV has already been restricted to the intended split.

## Reproducibility notes

- `MODEL_NAME` determines the evaluation result directory. Keep its spelling and capitalization identical across training, evaluation, and analysis.
- Separate expanded and non-expanded results when their model basenames are identical; otherwise later runs can overwrite earlier results.
- Training currently loads models with `device_map="auto"`; the supplied setup targets the single-process experiment environment. Adjust model loading before using DDP, FSDP, or DeepSpeed.
- Checkpoints are saved with `save_only_model=True`, so optimizer and scheduler states are not available for exact training resumption.
- Hugging Face loading uses `trust_remote_code=True`. Review third-party model repositories before running their code.
- Dataset and model redistribution remains subject to the licenses and terms of their respective upstream sources.
