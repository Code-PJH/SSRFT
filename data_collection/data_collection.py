import argparse
import tqdm
import json
import re
import time
import os

from typing import Optional

from .template import role_description
from .llm_eval import evaluate_response
from utils.LLM_wrapper import LLMWrapper
from utils.data import get_questions

def _extract_answer(content: str, role_desc: Optional[str]) -> str:
    content = content.strip()
    if role_desc is None:
        return content

    match = re.match(r"^\s*\{.*?\}\s*(.*)", content, re.DOTALL)
    return match.group(1).strip() if match else ""


def QA_data_collection(llm:LLMWrapper,questions,QA_template,output_path = "sft_data/sft_data.json",role_description:Optional[str]=None,threshold=6,enable_thinking=False):
    output_dir = os.path.dirname(os.path.abspath(output_path))
    os.makedirs(output_dir, exist_ok=True)

    QA_data = []
    total_tokens_used = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    pbar = tqdm.tqdm(questions)
    for question in pbar:
        try:
            # Call LLM with retry if answer is empty
            max_retries = 10
            retry_delay = 5
            answer = ""
            prompt = QA_template.substitute(question=question, role_description=role_description)
            for attempt in range(1, max_retries + 1):
                try:
                    response, _ = llm.chat(prompt, temperature=0.7,extra_body={"enable_thinking": enable_thinking})

                    # Count tokens
                    if 'raw_response' in response and hasattr(response['raw_response'], 'usage') and response['raw_response'].usage:
                        u = response['raw_response'].usage
                        total_tokens_used['prompt_tokens'] += u.prompt_tokens
                        total_tokens_used['completion_tokens'] += u.completion_tokens
                        total_tokens_used['total_tokens'] += u.total_tokens

                    content = ""
                    # Safely get content from response
                    if isinstance(response, dict):
                        content = response.get('content', '') or ''
                    else:
                        try:
                            content = str(response)
                        except Exception:
                            content = ''
                    
                    assert content, "Empty content received from LLM"

                    answer = _extract_answer(content, role_description)
                    
                    assert answer, "Failed to extract answer (format mismatch or empty)"

                    # Evaluate the answer if role_description is provided
                    if role_description is not None:
                        score, analysis, usage = evaluate_response(question, answer, llm, role_desc=role_description, enable_thinking=enable_thinking)
                        if usage:
                            total_tokens_used['prompt_tokens'] += usage['prompt_tokens']
                            total_tokens_used['completion_tokens'] += usage['completion_tokens']
                            total_tokens_used['total_tokens'] += usage['total_tokens']
                        
                        assert score is not None and score >= threshold, f"Low score ({score}) for answer"

                    else:
                        # Skip evaluation if no role_description
                        score = None
                        analysis = "Evaluation skipped due to missing role_description"
                    
                    # If we reached here, success
                    break

                except AssertionError as e:
                    # print(f"\nRetry {attempt}/{max_retries}: {e}")
                    pbar.set_description(f"Retry {attempt}/{max_retries}: {e}")
                    answer = ""  # Ensure answer is reset if we fail
                    time.sleep(retry_delay)
                except Exception as e:
                    print(f"\nError on attempt {attempt}/{max_retries}: {e}")
                    answer = ""
                    time.sleep(retry_delay)

            if not answer:
                print(f"Failed after {max_retries} attempts for question: {question}")
                continue

            # Format as SFT data (QA pair: question -> answer)
            
            entry = {
                "messages": [
                    {"role": "user", "content": question},
                    {"role": "assistant", "content": answer}
                ],
                "llm_score": score,
                "llm_analysis": analysis
            }
            QA_data.append(entry)

            # Real-time save
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(QA_data, f, ensure_ascii=False, indent=4)
            
            pbar.set_postfix(tokens=total_tokens_used['total_tokens'])
            
        except Exception as e:
            print(f"Error processing question: {question}. Error: {e}")
            continue

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(QA_data, f, ensure_ascii=False, indent=4)
    print(f"Data collection complete. Saved {len(QA_data)} entries to {output_path}")
    print(f"Total token usage: {total_tokens_used}")

def parse_args():
    parser = argparse.ArgumentParser(description="Collect one selected SFT dataset.")
    parser.add_argument(
        "--dataset",
        required=True,
        choices=(
            "mbti",
            "wvs",
            "ipip",
            "jailbreak",
            "jailbench",
            "jailbreak_no_character",
            "jailbench_no_character",
            "all",
        ),
        help="Dataset to collect. Use 'all' only when all configured collections are intended.",
    )
    parser.add_argument("--threshold", type=float, default=6)
    parser.add_argument("--enable-thinking", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()

    from dotenv import load_dotenv
    load_dotenv(override=True)
    
    from utils.LLM_wrapper import OpenAIWrapper
    
    base_url = os.getenv("OPENAI_API_BASE_URL") or 'https://dashscope.aliyuncs.com/compatible-mode/v1'
    api_key = os.getenv("OPENAI_API_KEY")
    model = os.getenv("OPENAI_MODEL", 'qwen3-max')
    
    if not api_key:
        print("Warning: OPENAI_API_KEY environment variable not set. Please set it in .env file.")

    llm = OpenAIWrapper(api_key=api_key, model=model, base_url=base_url)

    from .template import (
        IPIP_NEO_question_template,
        WVS_7_question_template,
        jailbreak_question_template,
        jailbreak_question_template_cn,
        jailbreak_question_template_cn_no_character,
        jailbreak_question_template_no_character,
        mbti_question_template,
    )

    datasets = {
        "mbti": {
            "label": "MBTI",
            "questions": "data_collection/questions/MBTI_questions.json",
            "template": mbti_question_template,
            "output": "sft_data/mbti_QA.json",
            "role": role_description,
        },
        "wvs": {
            "label": "WVS-7",
            "questions": "data_collection/questions/WVS-7_questions.json",
            "template": WVS_7_question_template,
            "output": "sft_data/WVS_7_QA.json",
            "role": role_description,
        },
        "ipip": {
            "label": "IPIP_NEO",
            "questions": "data_collection/questions/IPIP_NEO_questions.json",
            "template": IPIP_NEO_question_template,
            "output": "sft_data/IPIP_NEO_QA.json",
            "role": role_description,
        },
        "jailbreak": {
            "label": "jailbreak",
            "questions": "data_collection/questions/jailbreak_llms_questions.json",
            "template": jailbreak_question_template,
            "output": "sft_data/jailbreak_llms_QA.json",
            "role": role_description,
        },
        "jailbench": {
            "label": "JailBench",
            "questions": "data_collection/questions/JailBench_questions.json",
            "template": jailbreak_question_template_cn,
            "output": "sft_data/JailBench_QA.json",
            "role": role_description,
        },
        "jailbreak_no_character": {
            "label": "jailbreak (no character)",
            "questions": "data_collection/questions/jailbreak_llms_questions.json",
            "template": jailbreak_question_template_no_character,
            "output": "sft_data_no_character/jailbreak_llms_QA.json",
            "role": None,
        },
        "jailbench_no_character": {
            "label": "JailBench (no character)",
            "questions": "data_collection/questions/JailBench_questions.json",
            "template": jailbreak_question_template_cn_no_character,
            "output": "sft_data_no_character/JailBench_QA.json",
            "role": None,
        },
    }

    selected = list(datasets) if args.dataset == "all" else [args.dataset]
    for dataset_name in selected:
        config = datasets[dataset_name]
        print(f"Starting {config['label']} question answering data collection...")
        questions = get_questions(config["questions"])
        if not questions:
            raise ValueError(f"No questions loaded from {config['questions']}")
        QA_data_collection(
            llm,
            questions,
            config["template"],
            output_path=config["output"],
            role_description=config["role"],
            threshold=args.threshold,
            enable_thinking=args.enable_thinking,
        )


if __name__ == "__main__":
    main()
