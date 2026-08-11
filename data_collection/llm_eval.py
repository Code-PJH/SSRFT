import json
import re
import time
import logging
from tqdm import tqdm
from scipy import stats
from .template import eval_template, role_description, eval_expand_template
from utils.LLM_wrapper import OpenAIWrapper
from utils.decorator import retry_evaluation



def calculate_metrics(manual_scores, llm_scores):
    if len(manual_scores) != len(llm_scores) or len(manual_scores) < 2:
        return None

    metrics = {}
    
    try:
        pearson_corr, _ = stats.pearsonr(manual_scores, llm_scores)
        metrics['Pearson'] = pearson_corr
    except Exception as e:
        metrics['Pearson'] = float('nan')
        print(f"Pearson calculation failed: {e}")

    try:
        spearman_corr, _ = stats.spearmanr(manual_scores, llm_scores)
        metrics['Spearman'] = spearman_corr
    except Exception as e:
        metrics['Spearman'] = float('nan')
        print(f"Spearman calculation failed: {e}")

    try:
        kendall_corr, _ = stats.kendalltau(manual_scores, llm_scores)
        metrics['Kendall'] = kendall_corr
    except Exception as e:
        metrics['Kendall'] = float('nan')
        print(f"Kendall calculation failed: {e}")

    return metrics

@retry_evaluation()
def evaluate_response(question, answer, llm_wrapper, role_desc=role_description, ans=None,prompt_template=None,enable_thinking=False):
    """
    Evaluates a single response using the LLM.
    
    Args:
        question (str): The question asked.
        answer (str): The answer given.
        llm_wrapper (LLMWrapper): The LLM wrapper instance.
        role_desc (str): The role description for context.
        
    Returns:
        tuple: (score (float), analysis (str))
    """
    if prompt_template is None:
        prompt_template = eval_template

    prompt = prompt_template.substitute(
        role_description=role_desc,
        question=question, 
        answer=answer
    )
    if ans:
        prompt += f"\n注意：该问题的参考答案为：{ans}\n"

    # We use a fresh conversation for each evaluation
    response_dict = llm_wrapper.call([{"role": "user", "content": prompt}],temperature=0.7,extra_body={"enable_thinking": enable_thinking})
    llm_output = response_dict['content'].strip()
    
    usage = {"prompt_tokens":0, "completion_tokens":0, "total_tokens":0}
    if 'raw_response' in response_dict and hasattr(response_dict['raw_response'], 'usage') and response_dict['raw_response'].usage:
        u = response_dict['raw_response'].usage
        usage['prompt_tokens'] = u.prompt_tokens
        usage['completion_tokens'] = u.completion_tokens
        usage['total_tokens'] = u.total_tokens

    score = None
    analysis = llm_output

    # Extract score
    score_match = re.search(r'分数：\s*(\d+(\.\d+)?)', llm_output)
    if score_match:
        score = float(score_match.group(1))
    else:
        # Fallback: try to find any number if the format isn't exact
        match = re.search(r'\d+(\.\d+)?', llm_output)
        if match:
            score = float(match.group())
        else:
            print(f"\n Could not parse score. Output: {llm_output}")

    # Extract analysis
    analysis_match = re.search(r'分析：(.*?)(?:分数：|$)', llm_output, re.DOTALL)
    if analysis_match:
        analysis = analysis_match.group(1).strip()
    elif score_match:
            # If "分析：" tag is missing but "分数：" is present, take everything before score
            analysis = llm_output[:score_match.start()].replace("分析：", "").strip()
    
    return score, analysis, usage

@retry_evaluation()
def evaluate_expanded_questions(original_question, original_answer, expanded_question, expanded_answer, llm_wrapper, role_desc=role_description,prompt_template=None,enable_thinking=False):
    """
    Evaluates expanded questions using the LLM.
    
    Args:
        original_question (str): The original question.
        original_answer (str): The answer to the original question.
        expanded_question (str): The expanded question.
        expanded_answer (str): The answer to the expanded question.
        llm_wrapper (LLMWrapper): The LLM wrapper instance.
        role_desc (str): The role description for context.
        
    Returns:
        tuple: (scores (dict), analysis (dict), usage (dict))
    """
    if prompt_template is None:
        prompt_template = eval_expand_template

    prompt = prompt_template.substitute(
        role_description=role_desc,
        question=original_question, 
        answer=original_answer,
        expanded_question=expanded_question,
        expanded_answer=expanded_answer
    )
    
    # We use a fresh conversation for each evaluation
    response_dict = llm_wrapper.call([{"role": "user", "content": prompt}],temperature=0.7,extra_body={"enable_thinking": enable_thinking})
    llm_output = response_dict['content'].strip()
    
    usage = {"prompt_tokens":0, "completion_tokens":0, "total_tokens":0}
    if 'raw_response' in response_dict and hasattr(response_dict['raw_response'], 'usage') and response_dict['raw_response'].usage:
        u = response_dict['raw_response'].usage
        usage['prompt_tokens'] = u.prompt_tokens
        usage['completion_tokens'] = u.completion_tokens
        usage['total_tokens'] = u.total_tokens

    scores = {}
    analysis = llm_output

    # Extract scores
    # 扩展答案评分
    score_match = re.search(r'扩展答案评分：\s*(\d+(\.\d+)?)', llm_output)
    if score_match:
        scores['expanded_answer_score'] = float(score_match.group(1))
    
    # 扩展质量评分
    quality_match = re.search(r'扩展质量评分：\s*(\d+(\.\d+)?)', llm_output)
    if quality_match:
        scores['quality_score'] = float(quality_match.group(1))

    # 答案一致性评分
    consistency_match = re.search(r'答案一致性评分：\s*(\d+(\.\d+)?)', llm_output)
    if consistency_match:
        scores['consistency_score'] = float(consistency_match.group(1))

    try:
        analysis = {}
        
        # 扩展答案分析
        match1 = re.search(r'扩展答案分析：(.*?)扩展答案评分：', llm_output, re.DOTALL)
        if match1:
            analysis['expanded_answer_analysis'] = match1.group(1).strip()
        
        # 扩展质量分析
        match2 = re.search(r'扩展质量分析：(.*?)扩展质量评分：', llm_output, re.DOTALL)
        if match2:
            analysis['expanded_question_quality_analysis'] = match2.group(1).strip()
            
        # 答案一致性分析
        match3 = re.search(r'答案一致性分析：(.*?)答案一致性评分：', llm_output, re.DOTALL)
        if match3:
            analysis['answer_consistency_analysis'] = match3.group(1).strip()
        
        assert analysis['answer_consistency_analysis'] and analysis['expanded_question_quality_analysis'] and analysis['expanded_answer_analysis'], f"Failed to extract analyses. Output: {llm_output}"
    except AssertionError as e:
        logging.error(str(e))
        analysis = llm_output
    
    return scores, analysis, usage


if __name__ == "__main__":
    import os
    from dotenv import load_dotenv
    load_dotenv(override=True)
    base_url = os.getenv("OPENAI_API_BASE_URL") or 'https://dashscope.aliyuncs.com/compatible-mode/v1'
    api_key = os.getenv("OPENAI_API_KEY")
    model = os.getenv("OPENAI_MODEL", 'qwen3-max')
    llm = OpenAIWrapper(api_key=api_key, model=model, base_url=base_url)
    input_file = 'scored_ai_gen_QA_50.json'
    output_file = 'llm_scored_ai_gen_QA.json'
    
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    llm_scores = []
    manual_scores = []
    processed_data = []

    print(f"Starting LLM evaluation for {len(data)} items...")

    for i, item in enumerate(tqdm(data, desc="Evaluating")):
        messages = item.get('messages', [])
        user_content = next((m['content'] for m in messages if m['role'] == 'user'), "")
        assistant_content = next((m['content'] for m in messages if m['role'] == 'assistant'), "")
        
        if not user_content or not assistant_content:
            tqdm.write(f"Skipping item {i}: Missing content")
            continue

        llm_score = None
        llm_analysis = ""
        max_retries = 3
        
        for attempt in range(max_retries):
            llm_score, llm_analysis, _ = evaluate_response(user_content, assistant_content, llm)
            if llm_score is not None:
                break
            
            if attempt < max_retries - 1:
                tqdm.write(f"Retry {attempt+1}/{max_retries} for item {i}...")
                time.sleep(2)
        
        if llm_score is None:
            llm_score = 0.0

        item['llm_score'] = llm_score
        item['llm_analysis'] = llm_analysis
        processed_data.append(item)
        
        # Collect scores for correlation

        manual_score = item.get('manual_score')
        if manual_score is not None:
            try:
                manual_scores.append(float(manual_score))
                llm_scores.append(llm_score)
            except ValueError:
                pass
        
        tqdm.write(f"Item {i+1}: Manual={manual_score}, LLM={llm_score}")

    # Calculate correlation
    metrics = None
    if len(llm_scores) > 1:
        metrics = calculate_metrics(manual_scores, llm_scores)
        if metrics:
            print("\n" + "="*30)
            print("Evaluation Metrics:")
            print(f"Pearson Correlation:  {metrics['Pearson']:.4f} (Linear)")
            print(f"Spearman Correlation: {metrics['Spearman']:.4f} (Rank-based)")
            print(f"Kendall's Tau:        {metrics['Kendall']:.4f} (Robust to ties)")
            print("="*30 + "\n")
            
            if metrics['Pearson'] > 0.9 or metrics['Spearman'] > 0.9:
                print("Note: High correlation detected. However, due to score concentration (9-10),")
                print("consider checking the variance or using Cohen's Kappa for agreement analysis.")
    else:
        print("\nNot enough data to calculate correlation.")

    # Save results
    output_data = {
        "metrics": metrics,
        "data": processed_data
    }

    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=4)
        print(f"\nSaved LLM evaluation results to {output_file}")
    except Exception as e:
        print(f"Error saving results: {e}")

