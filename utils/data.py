import csv
import json
from typing import List, Dict

def load_csv_questions(path: str) -> List[Dict[str, str]]:
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return list(reader)
    
def get_questions(question_path):
    try:
        with open(question_path, 'r', encoding='utf-8') as f:
            questions = json.load(f).get("questions", [])
            return questions
    except FileNotFoundError:
        print(f"File {question_path} not found.")
        return []

def load_json(path: str):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)