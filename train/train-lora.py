import argparse
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from datasets import concatenate_datasets, load_dataset, load_from_disk
from transformers import (
	AutoModelForCausalLM,
	AutoTokenizer,
	DataCollatorForSeq2Seq,
	Trainer,
	TrainingArguments,
	EarlyStoppingCallback,
)
from peft import LoraConfig, get_peft_model, TaskType

def load_mixed_dataset(data_dir: str, column_name: str = "messages"):
	path = Path(data_dir)
	if not path.exists():
		raise FileNotFoundError(f"Data directory {data_dir} does not exist")

	datasets_list = []

	# 1. Collect and load all valid JSON files (excluding metadata)
	json_files = sorted(path.glob("*.json"))
	# Exclude HF metadata files if present at root
	json_files = [p for p in json_files if p.name not in ["dataset_info.json", "dataset_dict.json", "state.json"]]
	
	if json_files:
		logging.info("Found %d json files in %s. Loading individually...", len(json_files), data_dir)
		for json_file in json_files:
			try:
				ds = load_dataset("json", data_files=str(json_file), split="train")
				# Align columns: only keep the target column to allow concatenation
				if column_name in ds.column_names:
					ds = ds.select_columns([column_name])
					datasets_list.append(ds)
				else:
					logging.warning("JSON file %s loaded but '%s' column missing. Skipping.", json_file.name, column_name)
			except Exception as e:
				logging.error("Failed to load json file %s: %s", json_file.name, e)

	# 2. Iterate over subdirectories and try to load them as datasets from disk
	for p in sorted(path.iterdir()):
		if p.is_dir():
			# Heuristic: Check for common HF dataset indicator files
			if (p / "dataset_info.json").exists() or (p / "state.json").exists():
				try:
					logging.info("Loading dataset folder: %s", p.name)
					ds = load_from_disk(str(p))
					
					# Handle DatasetDict (e.g. if saved with train/test splits) - prefer 'train'
					if hasattr(ds, "keys") and "train" in ds:
						ds = ds["train"]
					
					if column_name in ds.column_names:
						ds = ds.select_columns([column_name])
						datasets_list.append(ds)
					else:
						logging.warning("Dataset folder %s loaded but '%s' column missing. Skipping.", p.name, column_name)
				except Exception as e:
					logging.warning("Failed to load dataset folder %s: %s", p.name, e)

	if not datasets_list:
		raise FileNotFoundError(f"No valid SFT/PT data (json files or dataset folders) found in {data_dir}")

	if len(datasets_list) == 1:
		return datasets_list[0]
	
	logging.info("Concatenating %d datasets...", len(datasets_list))
	return concatenate_datasets(datasets_list)

def tokenize_pt_example(
	example: Dict[str, str],
	tokenizer,
	max_length: int,
	text_column: str = "text"
) -> Dict[str, List[int]]:
	text = example.get(text_column, "")
	if not text:
		return {"input_ids": [], "labels": [], "attention_mask": []}

	text = text + tokenizer.eos_token
	
	enc = tokenizer(
		text,
		truncation=True,
		max_length=max_length,
		add_special_tokens=True,
	)

	enc["labels"] = enc["input_ids"].copy()
	return enc


def tokenize_example(
	example: Dict[str, List[Dict[str, str]]],
	tokenizer,
	max_length: int,
) -> Dict[str, List[int]]:
	messages = example["messages"]
	if not isinstance(messages, list) or not messages:
		raise ValueError("Each example must have a non-empty 'messages' list")

	# 1. Generate full input_ids
	# Use tokenize=True directly to get IDs
	full_input_ids = tokenizer.apply_chat_template(
		messages, 
		tokenize=True, 
		add_generation_prompt=False
	)

	# 2. Truncate if necessary
	if len(full_input_ids) > max_length:
		full_input_ids = full_input_ids[:max_length]
	
	# 3. Create labels (initially all -100)
	labels = [-100] * len(full_input_ids)
	
	# 4. Iteratively find user/assistant boundaries to set labels
	current_len = 0
	for i, msg in enumerate(messages):
		# If we already exceeded max_length, break
		if current_len >= len(full_input_ids):
			break
			
		# Get tokens for valid prefix
		prefix_ids = tokenizer.apply_chat_template(
			messages[:i+1], 
			tokenize=True, 
			add_generation_prompt=False
		)
		
		next_len = len(prefix_ids)
		
		# If this segment starts after our truncation point, stop
		if current_len >= len(full_input_ids):
			break
			
		# If this segment ends after truncation point, clamp it
		if next_len > len(full_input_ids):
			next_len = len(full_input_ids)
			
		# If role is assistant, we train on these tokens
		if msg["role"] == "assistant":
			# Set labels to input_ids for this segment
			labels[current_len:next_len] = full_input_ids[current_len:next_len]
			
		current_len = next_len

	return {
		"input_ids": full_input_ids,
		"labels": labels,
		"attention_mask": [1] * len(full_input_ids)
	}


def main():
	args = parse_args()
	logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
	logging.info("Arguments: %s", args)

	tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=True, trust_remote_code=True)
	if tokenizer.pad_token is None:
		tokenizer.pad_token = tokenizer.eos_token
		logging.info("No pad_token found in tokenizer. Setting pad_token to eos_token: %s", tokenizer.eos_token)
	tokenizer.padding_side = "right"

	# 1. SFT Dataset Loading & Tokenization
	raw_sft = load_mixed_dataset(args.data_dir, column_name="messages")
	def _tokenize_sft(example):
		return tokenize_example(example, tokenizer, args.max_seq_length)

	tokenized_sft = raw_sft.map(
		_tokenize_sft,
		remove_columns=raw_sft.column_names,
		num_proc=args.num_proc,
		desc="Tokenizing SFT data"
	)

	# 2. PT Dataset Loading & Tokenization
	tokenized_pt = None
	if args.pt_data_dir:
		raw_pt = load_mixed_dataset(args.pt_data_dir, column_name=args.pt_data_column)
		if raw_pt:
			def _tokenize_pt(example):
				return tokenize_pt_example(example, tokenizer, args.max_seq_length, args.pt_data_column)
			
			tokenized_pt = raw_pt.map(
				_tokenize_pt,
				remove_columns=raw_pt.column_names,
				num_proc=args.num_proc,
				desc="Tokenizing PT data"
			)

	# 3. Split & Combine
	tokenized_train_list = []
	tokenized_eval_list = []

	def _split_ds(ds, ratio, seed):
		if ratio > 0 and len(ds) > 0:
			split = ds.train_test_split(test_size=ratio, seed=seed)
			return split["train"], split["test"]
		return ds, None

	# Split SFT
	sft_train, sft_eval = _split_ds(tokenized_sft, args.val_ratio, args.seed)
	tokenized_train_list.append(sft_train)
	if sft_eval:
		tokenized_eval_list.append(sft_eval)

	# Split PT
	if tokenized_pt:
		pt_train, pt_eval = _split_ds(tokenized_pt, args.val_ratio, args.seed)
		tokenized_train_list.append(pt_train)
		if pt_eval:
			tokenized_eval_list.append(pt_eval)

	# Combine Train
	if len(tokenized_train_list) == 1:
		tokenized_train = tokenized_train_list[0]
	else:
		logging.info("Concatenating SFT and PT train datasets...")
		tokenized_train = concatenate_datasets(tokenized_train_list)
	
	tokenized_train = tokenized_train.shuffle(seed=args.seed)

	# Combine Eval
	tokenized_eval = None
	if tokenized_eval_list:
		if len(tokenized_eval_list) == 1:
			tokenized_eval = tokenized_eval_list[0]
		else:
			logging.info("Concatenating SFT and PT eval datasets...")
			tokenized_eval = concatenate_datasets(tokenized_eval_list)
		tokenized_eval = tokenized_eval.shuffle(seed=args.seed)

	if tokenized_eval is None and args.eval_strategy != "no":
		logging.warning("No validation split is available; disabling evaluation.")
		args.eval_strategy = "no"
	if tokenized_eval is None and args.load_best_model_at_end:
		raise ValueError("--load-best-model-at-end requires --val-ratio greater than 0.")
	if tokenized_eval is None and args.early_stopping_patience is not None:
		raise ValueError("--early-stopping-patience requires --val-ratio greater than 0.")
	if args.eval_strategy == "no" and args.load_best_model_at_end:
		raise ValueError("--load-best-model-at-end requires an enabled evaluation strategy.")
	if args.eval_strategy == "no" and args.early_stopping_patience is not None:
		raise ValueError("--early-stopping-patience requires an enabled evaluation strategy.")

	model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path,device_map="auto", trust_remote_code=True,dtype='auto')
	if args.gradient_checkpointing:
		model.gradient_checkpointing_enable()
		model.config.use_cache = False

	# PEFT / LoRA / DoRA Configuration
	lora_config = LoraConfig(
		task_type=TaskType.CAUSAL_LM,
		r=args.lora_r,
		lora_alpha=args.lora_alpha,
		lora_dropout=args.lora_dropout,
		target_modules="all-linear",
		use_dora=args.use_dora,
	)
	model = get_peft_model(model, lora_config)
	model.print_trainable_parameters()

	training_args = TrainingArguments(
		output_dir=args.output_dir,
		per_device_train_batch_size=args.per_device_train_batch_size,
		per_device_eval_batch_size=args.per_device_eval_batch_size,
		gradient_accumulation_steps=args.gradient_accumulation_steps,
		num_train_epochs=args.num_train_epochs,
		learning_rate=args.learning_rate,
		weight_decay=args.weight_decay,
		warmup_ratio=args.warmup_ratio,
		logging_steps=args.logging_steps,
		save_steps=args.save_steps,
		eval_steps=args.eval_steps if tokenized_eval is not None else None,
		eval_strategy=args.eval_strategy,
		save_strategy=args.save_strategy,
		save_total_limit=1,
		bf16=args.bf16,
		fp16=args.fp16,
		seed=args.seed,
		dataloader_num_workers=2,
		report_to=["tensorboard"],
		load_best_model_at_end=args.load_best_model_at_end if hasattr(args, "load_best_model_at_end") else False,
		save_only_model=True,
		overwrite_output_dir=True,
		logging_dir=args.logging_dir
	)

	data_collator = DataCollatorForSeq2Seq(
		tokenizer=tokenizer,
		model=model,
		label_pad_token_id=-100,
		padding="longest",
	)

	callbacks = []
	if args.early_stopping_patience is not None:
		callbacks.append(EarlyStoppingCallback(early_stopping_patience=args.early_stopping_patience))
	trainer = Trainer(
		model=model,
		args=training_args,
		train_dataset=tokenized_train,
		eval_dataset=tokenized_eval,
		data_collator=data_collator,
		callbacks=callbacks,
	)

	trainer.train()
	if args.save_as_bf16:
		if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
			logging.info("Converting model to bfloat16 before saving...")
			trainer.model.to(torch.bfloat16)
		else:
			logging.warning("BF16 is not supported on this device. Cannot save in bfloat16. Saving in default format.")
	trainer.save_model(args.output_dir)
	tokenizer.save_pretrained(args.output_dir)

def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Supervised fine-tuning on sft_data")
	parser.add_argument("--model-name-or-path", type=str, required=True, help="Base model to fine-tune")
	parser.add_argument("--data-dir", type=str, default="sft_data", help="Directory with *.json training files")
	parser.add_argument("--output-dir", type=str, default="./outputs", help="Where to store checkpoints")
	parser.add_argument("--max-seq-length", type=int, default=2048, help="Max sequence length")
	parser.add_argument("--per-device-train-batch-size", type=int, default=16)
	parser.add_argument("--per-device-eval-batch-size", type=int, default=None)
	parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
	parser.add_argument("--learning-rate", type=float, default=2e-5)
	parser.add_argument("--weight-decay", type=float, default=0.01)
	parser.add_argument("--num-train-epochs", type=float, default=1.0)
	parser.add_argument("--warmup-ratio", type=float, default=0.03)
	parser.add_argument("--logging-steps", type=int, default=10)
	parser.add_argument("--eval-strategy", type=str, default="no", choices=["no", "steps", "epoch"], help="Evaluation strategy")
	parser.add_argument("--save-strategy", type=str, default="steps", help="Save strategy")
	parser.add_argument("--save-steps", type=int, default=500)
	parser.add_argument("--eval-steps", type=int, default=500)
	parser.add_argument("--val-ratio", type=float, default=0.0, help="Fraction of data for validation (0 to disable)")
	parser.add_argument("--seed", type=int, default=42)
	parser.add_argument("--num-proc", type=int, default=1, help="Number of processes for tokenization")
	parser.add_argument("--gradient-checkpointing", action="store_true")
	parser.add_argument("--bf16", action="store_true", help="Use bfloat16 if supported")
	parser.add_argument("--fp16", action="store_true", help="Use float16 if supported")
	parser.add_argument("--early-stopping-patience", type=int, default=None, help="Patience for early stopping")
	parser.add_argument("--load-best-model-at-end", action="store_true", help="Load best model at end of training")
	parser.add_argument("--save-as-bf16", action="store_true", help="Save model in bfloat16 format")
	parser.add_argument("--pt-data-dir", type=str, default=None, help="Directory with *.json pre-training files (raw text)")
	parser.add_argument("--pt-data-column", type=str, default="text", help="Column name for text in PT datasets")
	parser.add_argument("--logging_dir", type=str, default=None, help="Directory for logging")
	
	# LoRA / DoRA args
	parser.add_argument("--lora-r", type=int, default=8, help="LoRA rank")
	parser.add_argument("--lora-alpha", type=int, default=16, help="LoRA alpha")
	parser.add_argument("--lora-dropout", type=float, default=0.05, help="LoRA dropout")
	parser.add_argument("--use-dora", action="store_true", help="Use DoRA (Weight-Decomposed Low-Rank Adaptation)")
	
	args=parser.parse_args()
	if not 0.0 <= args.val_ratio < 1.0:
		parser.error("--val-ratio must be in the range [0, 1).")
	if args.per_device_eval_batch_size is None:
		args.per_device_eval_batch_size = args.per_device_train_batch_size
	return args
if __name__ == "__main__":
	main()
