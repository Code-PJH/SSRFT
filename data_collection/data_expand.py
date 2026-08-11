import argparse
import tqdm
import json
import re
import time
import os
import logging
from typing import Optional

from .template import role_description, expand_question_template
from .llm_eval import evaluate_expanded_questions
from utils.LLM_wrapper import LLMWrapper

def _extract_answer(content: str, role_desc: Optional[str]) -> str:
    content = content.strip()
    if role_desc is None:
        return content

    match = re.match(r"^\s*\{.*?\}\s*(.*)", content, re.DOTALL)
    return match.group(1).strip() if match else ""


def expand_question_QA_data_collection(llm:LLMWrapper, QA_data, QA_template, output_path="sft_data_expand/expanded_QA_data.json", role_description:Optional[str]=None, expand_template=expand_question_template, threshold=6,enable_thinking=False):
    output_dir = os.path.dirname(os.path.abspath(output_path))
    os.makedirs(output_dir, exist_ok=True)

    expanded_QA_data = []
    failed_QA_data = []

    # Detailed usage statistics
    usage_stats = {
        "generate_question": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        "answer_question": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        "evaluate": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        "total": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    }
    
    def update_usage(phase, usage_obj):
        if not usage_obj: return
        # Handle object or dict
        u_dict = {}
        if hasattr(usage_obj, 'prompt_tokens'):
            u_dict['prompt_tokens'] = usage_obj.prompt_tokens
            u_dict['completion_tokens'] = usage_obj.completion_tokens
            u_dict['total_tokens'] = usage_obj.total_tokens
        elif isinstance(usage_obj, dict):
            u_dict = usage_obj
        else:
            return
            
        for k in ["prompt_tokens", "completion_tokens", "total_tokens"]:
            val = u_dict.get(k, 0)
            usage_stats[phase][k] += val
            usage_stats["total"][k] += val
    
    pbar = tqdm.tqdm(QA_data)
    for item in pbar:
        try:
            # Extract original question and answer
            messages = item.get("messages", [])
            user_msg = next((m for m in messages if m["role"] == "user"), None)
            asst_msg = next((m for m in messages if m["role"] == "assistant"), None)
            
            if not user_msg or not asst_msg:
                continue
            
            original_question = user_msg["content"]
            original_answer = asst_msg["content"]
            
            max_question_retries = 5
            item_completed = False
            
            for q_attempt in range(max_question_retries):
                # --- Step 1: Expand the Question --- 
                retry_delay = 2
                expanded_question = ""
                
                expand_prompt = expand_template.substitute(role_description=role_description, question=original_question)
                
                try:
                    response, _ = llm.chat(expand_prompt, temperature=0.7,extra_body={"enable_thinking": enable_thinking})
                    
                    if 'raw_response' in response and hasattr(response['raw_response'], 'usage'):
                         update_usage("generate_question", response['raw_response'].usage)

                    content = ""
                    if isinstance(response, dict):
                        content = response.get('content', '') or ''
                    else:
                        content = str(response)

                    match = re.search(r"\{(.*?)\}", content, re.DOTALL)
                    if match:
                        expanded_question = match.group(1).strip()
                    else:
                        raise ValueError("Could not find expanded question in braces")
                    
                    if not expanded_question:
                        raise ValueError("Empty expanded question extracted")
                        
                except Exception as e:
                    # print(f"Generate Question Error: {e}")
                    time.sleep(retry_delay)
                    continue # Retry generating question (inner generation loop omitted for simplicity, relying on outer q_attempt loop)

                if not expanded_question:
                    continue

                # --- Step 2: Answer the Expanded Question & Validate ---
                max_answer_retries = 3
                
                for a_attempt in range(max_answer_retries):
                    try:
                        answer_prompt = QA_template.substitute(role_description=role_description, question=expanded_question)
                        response, _ = llm.chat(answer_prompt, temperature=0.7,extra_body={"enable_thinking": enable_thinking})
                        
                        if 'raw_response' in response and hasattr(response['raw_response'], 'usage'):
                            update_usage("answer_question", response['raw_response'].usage)

                        content = ""
                        if isinstance(response, dict):
                            content = response.get('content', '') or ''
                        else:
                            content = str(response)
                        
                        current_answer = _extract_answer(content, role_description)
                        
                        if not current_answer:
                             raise ValueError("Could not extract answer")
                        
                        # Evaluate
                        scores, analysis, usage = evaluate_expanded_questions(
                            original_question, original_answer, 
                            expanded_question, current_answer, 
                            llm, role_desc=role_description,
                            enable_thinking=enable_thinking
                        )
                        update_usage("evaluate", usage)

                        if not scores:
                            raise ValueError("Evaluation returned None")
                        
                        quality_score = scores.get('quality_score', 0)
                        consistency_score = scores.get('consistency_score', 0)
                        expanded_answer_score = scores.get('expanded_answer_score', 0)
                        
                        # 2: Retry, based on scores
                        if expanded_answer_score < threshold:
                            # Bad Answer quality - retry answering same question
                            logging.warning(f"Low Answer Score: {expanded_answer_score} Retrying A...")
                            continue # Will loop a_attempt
                        
                        if quality_score < threshold or consistency_score < threshold:
                            # Bad Question quality - needs impossible fix here, so break to generate new question
                            logging.warning(f"Low Quality/Consist: Q={quality_score}, C={consistency_score} Retrying Q...")
                            break 
                        
                        
                        # If here, Success!
                        entry = {
                            "messages": [
                                {"role": "user", "content": expanded_question},
                                {"role": "assistant", "content": current_answer}
                            ],
                            "original_question": original_question,
                            "original_answer": original_answer,
                            "llm_scores": scores,
                            "llm_analysis": analysis
                        }
                        expanded_QA_data.append(entry)
                        item_completed = True
                        break # Break answer loop
                        
                    except Exception as e:
                        # print(f"Answer/Eval Error: {e}")
                        logging.error(f"Answer/Eval Error: {e}")
                        time.sleep(retry_delay)
                
                if item_completed:
                    break # Break question loop
            
            if not item_completed:
                failed_entry = {
                    "original_question": original_question,
                    "original_answer": original_answer,
                    "expanded_question": None,
                    "expanded_answer": None
                }
                failed_QA_data.append(failed_entry)

            # Real-time save
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(expanded_QA_data, f, ensure_ascii=False, indent=4)
            
            # Save failed data real-time as well
            if len(failed_QA_data) > 0 and len(failed_QA_data) % 5 == 0:
                # determine failed filename
                dir_name = os.path.dirname(output_path)
                file_name = os.path.basename(output_path)
                failed_path = os.path.join(dir_name, f"failed_{file_name}")
                with open(failed_path, 'w', encoding='utf-8') as f:
                    json.dump(failed_QA_data, f, ensure_ascii=False, indent=4)

            pbar.set_postfix(tokens=usage_stats['total']['total_tokens'])
            
        except Exception as e:
            print(f"Error processing item: {e}")
            continue

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(expanded_QA_data, f, ensure_ascii=False, indent=4)
    
    # Save final failed data
    if len(failed_QA_data) > 0:
        dir_name = os.path.dirname(output_path)
        file_name = os.path.basename(output_path)
        failed_path = os.path.join(dir_name, f"failed_{file_name}")
        with open(failed_path, 'w', encoding='utf-8') as f:
            json.dump(failed_QA_data, f, ensure_ascii=False, indent=4)
        print(f"Saved {len(failed_QA_data)} failed entries to {failed_path}")
        
    log_dir = "logs"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    base_name = os.path.basename(output_path)
    usage_file_name = f"usage_{base_name}"
    usage_path = os.path.join(log_dir, usage_file_name)
    
    with open(usage_path, 'w', encoding='utf-8') as f:
        json.dump(usage_stats, f, ensure_ascii=False, indent=4)    

    print(f"Expansion complete. Saved {len(expanded_QA_data)} entries to {output_path}")
    print(f"Total token usage saved to {usage_path}")


def process_manual_failed_samples(llm: LLMWrapper, failed_file_path, output_path, question_template, role_description: Optional[str] = None, expand_template=expand_question_template,enable_thinking=False):
    """
    Manually process failed samples.
    Attempts to generate expansion using LLM, then asks human for evaluation/edit.
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(failed_file_path)), exist_ok=True)

    if not os.path.exists(failed_file_path):
        print(f"Failed file {failed_file_path} not found.")
        return

    try:
        with open(failed_file_path, 'r', encoding='utf-8') as f:
            failed_data = json.load(f)
    except json.JSONDecodeError:
        print(f"Error decoding JSON from {failed_file_path}")
        return

    if not failed_data:
        print("No failed samples found.")
        return

    # Load existing successful data
    expanded_QA_data = []
    if os.path.exists(output_path):
        try:
            with open(output_path, 'r', encoding='utf-8') as f:
                expanded_QA_data = json.load(f)
        except json.JSONDecodeError:
            print(f"Warning: Could not read existing output file {output_path}. Starting fresh.")

    print(f"Processing {len(failed_data)} failed samples...")

    remaining_failed = []
    
    for i, item in enumerate(failed_data):
        print(f"\n{'='*20} Item {i+1}/{len(failed_data)} {'='*20}")
        original_question = item.get("original_question")
        original_answer = item.get("original_answer")
        
        print(f"Original Q: {original_question}")
        print(f"Original A: {original_answer}")
        print("-" * 50)

        # 1. Try to generate expansion automatically first
        retry_delay = 2
        expanded_question = ""
        current_answer = ""
        
        # Expand Q
        expand_prompt = expand_template.substitute(role_description=role_description, question=original_question)
        try:
            print("Generating expansion...")
            response, _ = llm.chat(expand_prompt, temperature=0.7,extra_body={"enable_thinking": enable_thinking})
            # Safely get content
            if isinstance(response, dict):
                content = response.get('content', '') or ''
            else:
                try:
                    content = str(response)
                except Exception:
                    content = ''
            
            match = re.search(r"\{(.*?)\}", content, re.DOTALL)
            if match:
                expanded_question = match.group(1).strip()
        except Exception as e:
            print(f"Error generating question: {e}")

        # Expand A (if Q exists)
        if expanded_question:
            answer_prompt = question_template.substitute(role_description=role_description, question=expanded_question)
            try:
                print("Generating answer...")
                response, _ = llm.chat(answer_prompt, temperature=0.7,extra_body={"enable_thinking": enable_thinking})
                 # Safely get content again
                if isinstance(response, dict):
                    content = response.get('content', '') or ''
                else:
                    try:
                        content = str(response)
                    except Exception:
                         content = ''
                    
                current_answer = _extract_answer(content, role_description)

            except Exception as e:
                print(f"Error generating answer: {e}")

        # 2. Show result and ask for manual evaluation/edit
        accepted = False
        while True:
            # Evaluate current candidate if both exists
            scores = {}
            scores_display = "N/A"
            analysis_display = "N/A"
            if expanded_question and current_answer:
                print("Evaluating current candidate...")
                try:
                    scores, analysis, _ = evaluate_expanded_questions(
                        original_question, original_answer, 
                        expanded_question, current_answer, 
                        llm, role_desc=role_description,
                        enable_thinking=enable_thinking
                    )
                    if scores:
                        scores_display = (f"Ans:{scores.get('expanded_answer_score')} | "
                                        f"Qual:{scores.get('quality_score')} | "
                                        f"Consist:{scores.get('consistency_score')}")
                        analysis_display = (f"Q Analysis: {analysis.get('expanded_question_quality_analysis', 'N/A')}\n"
                                            f"A Analysis: {analysis.get('expanded_answer_analysis', 'N/A')}\n"
                                            f"Consistency Analysis: {analysis.get('answer_consistency_analysis', 'N/A')}")
                except Exception as e:
                    scores_display = f"Error: {e}"

            # Display current state
            print(f"\n[Expansion Candidate]")
            print(f"Expanded Q: {expanded_question}")
            print(f"Expanded A: {current_answer}")
            print(f"Scores: {scores_display}")
            print(f"Analysis: {analysis_display}")
            print("-" * 30)
            
            raw_choice = input("Action? [a]ccept / [e]dit / [req]new question / [rea]new answer / [s]kip: ").lower().strip()
            
            if raw_choice.startswith('a'):
                # Accept
                entry = {
                    "messages": [
                        {"role": "user", "content": expanded_question},
                        {"role": "assistant", "content": current_answer}
                    ],
                    "original_question": original_question,
                    "original_answer": original_answer,
                    "llm_scores": {
                        "expanded_answer_score": 0,
                        "quality_score": 0,
                        "consistency_score": 0
                    },
                    "llm_analysis":{
                        "expanded_answer_analysis": "Manual Evaluation Passed",
                        "expanded_question_quality_analysis":"Manual Evaluation Passed",
                        "answer_consistency_analysis": "Manual Evaluation Passed"
                    }
                }
                expanded_QA_data.append(entry)
                
                # Save immediately to ensure data safety
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(expanded_QA_data, f, ensure_ascii=False, indent=4)
                print("Saved to output.")
                accepted = True
                break
                
            elif raw_choice.startswith('e'):
                # Manual Edit
                print("\n----- Editing Mode -----")
                new_q = input(f"New Question (Press Enter to keep current):\n").strip()
                if new_q: 
                    expanded_question = new_q
                
                new_a = input(f"New Answer (Press Enter to keep current):\n").strip()
                if new_a: 
                    current_answer = new_a
                print("Updated local candidate. Please review again.")
                
            elif raw_choice.startswith('req'):
                # Retry Question Generation
                print("Regenerating QUESTION via LLM...")
                try:
                     response, _ = llm.chat(expand_prompt, temperature=0.9) # Higher temp
                     if isinstance(response, dict):
                        content = response.get('content', '') or ''
                     else:
                        content = str(response)
                     
                     match = re.search(r"\{(.*?)\}", content, re.DOTALL)
                     if match: 
                         expanded_question = match.group(1).strip()
                         print("New Question Generated.")
                         # Clear answer as it might not match
                         current_answer = "" 
                     else:
                         print("Failed to extract question from regeneration.")
                except Exception as e:
                    print(f"Error: {e}")

            elif raw_choice.startswith('rea'):
                # Retry Answer Generation
                if not expanded_question:
                    print("Cannot generate answer without a question.")
                    continue
                print("Regenerating ANSWER via LLM...")
                try:
                     answer_prompt = question_template.substitute(role_description=role_description, question=expanded_question)
                     response, _ = llm.chat(answer_prompt, temperature=0.7)
                     if isinstance(response, dict):
                        content = response.get('content', '') or ''
                     else:
                        content = str(response)

                     current_answer = _extract_answer(content, role_description)
                     print("New Answer Generated.")
                except Exception as e:
                    print(f"Error: {e}")
                
            elif raw_choice.startswith('s'):
                # Skip
                print("Skipped item.")
                remaining_failed.append(item)
                break
            else:
                print("Invalid command. Please try again.")
        
        # End while loop for one item

    # Save remaining failed
    with open(failed_file_path, 'w', encoding='utf-8') as f:
        json.dump(remaining_failed, f, ensure_ascii=False, indent=4)
    print(f"\nAll Done. {len(remaining_failed)} items remain in failed file.")

def parse_args():
    parser = argparse.ArgumentParser(description="Expand one selected SFT dataset.")
    parser.add_argument(
        "--dataset",
        required=True,
        choices=("mbti", "wvs", "ipip", "all"),
        help="Dataset to expand. Use 'all' only when all configured expansions are intended.",
    )
    parser.add_argument("--threshold", type=float, default=8)
    parser.add_argument("--enable-thinking", action="store_true")
    parser.add_argument(
        "--process-failures",
        action="store_true",
        help="Interactively process failed samples after expansion.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    from utils.data import load_json
    from dotenv import load_dotenv
    load_dotenv(override=True)
    
    from utils.LLM_wrapper import OpenAIWrapper
    
    base_url = os.getenv("OPENAI_API_BASE_URL") or 'https://dashscope.aliyuncs.com/compatible-mode/v1'
    api_key = os.getenv("OPENAI_API_KEY")
    model = os.getenv("OPENAI_MODEL", 'qwen3-max')
    
    if not api_key:
        print("Warning: OPENAI_API_KEY environment variable not set. Please set it in .env file.")

    llm = OpenAIWrapper(api_key=api_key, model=model, base_url=base_url,max_new_tokens=1024)

    from .template import (
        expand_question_template,
        expand_question_template_cn,
        general_question_template,
        general_question_template_cn,
    )

    datasets = {
        "mbti": {
            "label": "MBTI",
            "input": "sft_data/mbti_QA.json",
            "output": "sft_data_expand/Expanded_mbti_QA.json",
            "question_template": general_question_template_cn,
            "expand_template": expand_question_template_cn,
        },
        "wvs": {
            "label": "WVS-7",
            "input": "sft_data/WVS_7_QA.json",
            "output": "sft_data_expand/Expanded_WVS_7_QA.json",
            "question_template": general_question_template,
            "expand_template": expand_question_template,
        },
        "ipip": {
            "label": "IPIP_NEO",
            "input": "sft_data/IPIP_NEO_QA.json",
            "output": "sft_data_expand/Expanded_IPIP_NEO_QA.json",
            "question_template": general_question_template,
            "expand_template": expand_question_template,
        },
    }

    selected = list(datasets) if args.dataset == "all" else [args.dataset]
    for dataset_name in selected:
        config = datasets[dataset_name]
        print(f"Starting {config['label']} QA expansion...")
        original_QA_data = load_json(config["input"])
        expand_question_QA_data_collection(
            llm,
            original_QA_data,
            config["question_template"],
            output_path=config["output"],
            role_description=role_description,
            expand_template=config["expand_template"],
            threshold=args.threshold,
            enable_thinking=args.enable_thinking,
        )

        if args.process_failures:
            output_dir = os.path.dirname(config["output"])
            failed_path = os.path.join(
                output_dir,
                f"failed_{os.path.basename(config['output'])}",
            )
            process_manual_failed_samples(
                llm,
                failed_path,
                config["output"],
                config["question_template"],
                role_description=role_description,
                expand_template=config["expand_template"],
                enable_thinking=args.enable_thinking,
            )


if __name__ == "__main__":
    main()
