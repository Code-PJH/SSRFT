import json
import random
import os

def main():
    input_file = os.path.join('sft_data', 'ai_gen_QA.json')
    output_file = 'scored_ai_gen_QA_50.json'
    
    # Load data
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: File {input_file} not found.")
        return

    # Sample 50 items
    if len(data) < 50:
        print(f"Warning: Only {len(data)} items found. Using all of them.")
        sampled_data = data
    else:
        sampled_data = random.sample(data, 50)

    scored_data = []

    print(f"Starting manual scoring for {len(sampled_data)} items.")
    print("Please enter a score for each item.")
    
    for i, item in enumerate(sampled_data):
        print(f"\n{'='*20} Item {i+1}/{len(sampled_data)} {'='*20}")
        messages = item.get('messages', [])
        user_content = next((m['content'] for m in messages if m['role'] == 'user'), "No user content")
        assistant_content = next((m['content'] for m in messages if m['role'] == 'assistant'), "No assistant content")
        
        print(f"Question: {user_content}")
        print(f"Answer:   {assistant_content}")
        print("-" * 50)
        
        while True:
            score_input = input("Enter score: ")
            if score_input.strip(): # Ensure not empty
                # Try to convert to number if possible, otherwise keep as string or ask user preference?
                # User didn't specify type, but usually scores are numbers.
                try:
                    score = float(score_input)
                    break
                except ValueError:
                    print("Invalid input. Please enter a numeric score.")
            else:
                print("Input cannot be empty.")
        
        # Create a copy to avoid modifying original if we were referencing it (though we loaded fresh)
        # and add the score
        item_with_score = item.copy()
        item_with_score['manual_score'] = score
        scored_data.append(item_with_score)

    # Save to file
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(scored_data, f, ensure_ascii=False, indent=4)
        print(f"\nScoring complete. Data saved to {output_file}")
    except Exception as e:
        print(f"Error saving file: {e}")

if __name__ == "__main__":
    main()
