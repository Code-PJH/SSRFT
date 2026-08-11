import argparse
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm
import sys
import os
import json

from utils.LLM_wrapper import build_client

def calculate_bfi_scores(responses):
    """
    Calculate Big Five Inventory (BFI) scores based on item responses.
    
    Args:
        responses (dict): A dictionary where keys are question IDs (int or str) and values are scores (1-5).
                          Example: {1: 4, 2: 2, ...} (or {'1': 4, '2': 2, ...})
    
    Returns:
        dict: A dictionary containing the average scores for each of the Big Five traits.
    """
    
    # Define the traits and their associated items.
    # Integers represent item IDs.
    # Tuples (ID, 'R') represent reverse-scored items.
    traits_config = {
        "Extraversion": [1, (6, 'R'), 11, 16, (21, 'R'), 26, (31, 'R'), 36],
        "Agreeableness": [(2, 'R'), 7, (12, 'R'), 17, 22, (27, 'R'), 32, (37, 'R'), 42],
        "Conscientiousness": [3, (8, 'R'), 13, (18, 'R'), (23, 'R'), 28, 33, 38, (43, 'R')],
        "Neuroticism": [4, (9, 'R'), 14, 19, (24, 'R'), 29, (34, 'R'), 39],
        "Openness": [5, 10, 15, 20, 25, 30, (35, 'R'), 40, (41, 'R'), 44]
    }
    
    results = {}
    
    for trait, items in traits_config.items():
        trait_scores = []
        for item in items:
            is_reverse = False
            item_id = item
            
            if isinstance(item, tuple):
                item_id = item[0]
                is_reverse = True
            
            # Retrieve score, handling both int and str keys
            score = responses.get(item_id)
            if score is None:
                score = responses.get(str(item_id))
            
            if score is not None:
                try:
                    score_val = float(score)
                    if is_reverse:
                        score_val = 6 - score_val
                    trait_scores.append(score_val)
                except ValueError:
                    pass
            else:
                pass
        
        # Calculate average
        if trait_scores:
            results[trait] = sum(trait_scores) / len(trait_scores)
        else:
            results[trait] = 0.0
            
    return results

def eval_bfi(model_path, data_path, output_path, device="cuda", batch_size=8, use_lora=False, lora_name_or_path=None):
    print(f"Loading model from {model_path}")
    try:
        client = build_client(
            backend="huggingface", 
            model=model_path, 
            device=device,
            model_kwargs={"trust_remote_code": True}, # Ensure custom models load correctly
            use_lora=use_lora,
            lora_name_or_path=lora_name_or_path
        )
        tokenizer = client.tokenizer
        model = client.model
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"Failed to initialize model with LLMWrapper: {e}")
        return

    model.eval()

    target_tokens_map = {}
    for i in range(1, 6):
        s = str(i)
        variants = [s, " " + s, s+' ']
        ids_list = []
        for v in variants:
            enc = tokenizer.encode(v, add_special_tokens=False)
            if len(enc) == 1:
                ids_list.append(enc[0])
            elif len(enc) > 1:
                pass
        
        ids_list = list(set(ids_list))
        if not ids_list:
            print(f"Warning: Could not find single token encoding for digit '{i}'")
        
        target_tokens_map[i] = ids_list
        
    print(f"Target token IDs map: {target_tokens_map}")
    
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is not None:
             tokenizer.pad_token_id = tokenizer.eos_token_id
        else:
             raise ValueError("Tokenizer does not have a pad_token_id or eos_token_id. Please set one for proper batching.")
    
    tokenizer.padding_side = 'left'

    df = pd.read_csv(data_path)
    
    responses = {}
    entropies = []
    
    results = []

    print(f"Starting evaluation with batch size {batch_size}...")
    
    for i in tqdm(range(0, len(df), batch_size)):
        batch_df = df.iloc[i:i+batch_size]
        batch_prompts = []
        batch_ids = []
        batch_questions = []

        for _, row in batch_df.iterrows():
            q_id = row['q_id']
            question = row['question']
            
            if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
                messages = [{"role": "user", "content": question}]
                prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            else:
                raise ValueError("Tokenizer does not support chat templates. Please ensure the tokenizer has a chat_template attribute and apply_chat_template method.")
            
            batch_prompts.append(prompt_text)
            batch_ids.append(q_id)
            batch_questions.append(question)
            
        inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True).to(model.device)
        
        with torch.no_grad():
            outputs = model(**inputs)
            last_token_logits = outputs.logits[:, -1, :] 
            
        for idx in range(len(batch_prompts)):
            logits = last_token_logits[idx]
            q_id = batch_ids[idx]
            question = batch_questions[idx]
            
            target_logits = []
            
            for score_val in range(1, 6):
                ids = target_tokens_map.get(score_val, [])
                if not ids:
                    target_logits.append(-float('inf'))
                    continue
                
                # Get logits for all token variants of this score (e.g. "1", " 1")
                # and take the maximum logit
                token_logits = logits[ids]
                best_logit = torch.max(token_logits).item()
                target_logits.append(best_logit)
                
            target_logits_tensor = torch.tensor(target_logits)
            probs = F.softmax(target_logits_tensor, dim=0)
            
            entropy = -torch.sum(probs * torch.log(probs + 1e-9)).item()
            entropies.append(entropy)
            
            best_idx = torch.argmax(probs).item()
            selected_score = best_idx + 1 # 1-based
            
            responses[q_id] = selected_score
            
            results.append({
                "q_id": int(q_id) if isinstance(q_id, (int, float)) else str(q_id),
                "question": question,
                "selected_score": selected_score,
                "entropy": entropy,
                "logits": target_logits, # list of floats
                "probs": probs.tolist()
            })
        
    bfi_scores = calculate_bfi_scores(responses)
    avg_entropy = sum(entropies) / len(entropies) if entropies else 0
    
    print("\nEvaluation Results:")
    print("-" * 30)
    print(f"Average Entropy: {avg_entropy:.4f}")
    print("Personality Traits Scores:")
    for trait, score in bfi_scores.items():
        print(f"  {trait}: {score:.4f}")
        
    final_output = {
        "model_path": model_path,
        "average_entropy": avg_entropy,
        "bfi_scores": bfi_scores,
        "details": results
    }
    
    if output_path:
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(final_output, f, indent=4, ensure_ascii=False)
        print(f"Detailed results saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate BFI-44 Personality Traits on LLM")
    parser.add_argument("--target-model", type=str, required=True, help="Path to local model")
    parser.add_argument("--input", type=str, default="test_datasets/BFI-44.csv", help="Path to BFI-44 CSV file")
    parser.add_argument("--output", type=str, default="eval_results/bfi_results.json", help="Path to save output JSON")
    parser.add_argument("--device", type=str, default="auto", help="Device to run evaluation on 'cuda'/'cpu'/'auto'(default)")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size for evaluation")
    parser.add_argument("--use-lora", action="store_true", help="Whether to use LoRA adapter")
    parser.add_argument("--lora-name-or-path", type=str, default=None, help="Path or name of the LoRA adapter")
    args = parser.parse_args()
    
    output_dir = os.path.dirname(args.output) if args.output else ""
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    eval_bfi(args.target_model, args.input, args.output, args.device, args.batch_size, args.use_lora, args.lora_name_or_path)
