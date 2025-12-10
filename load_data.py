"""
Data loading utilities for MathQA dataset
"""
import json
import os
from typing import List, Dict, Tuple


def load_jsonl(file_path: str) -> List[Dict]:
    """Load JSONL file and return list of dictionaries."""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def format_qa_pair(question: str, answer: str) -> str:
    """Format question and answer as a training prompt."""
    # Format as a simple Q&A pair
    prompt = f"Question: {question}\nAnswer: {answer}"
    return prompt


def prepare_dataset(file_path: str) -> List[str]:
    """Load and format the dataset for training."""
    data = load_jsonl(file_path)
    formatted_data = []
    
    for item in data:
        question = item.get('question', '')
        answer = item.get('answer', '')
        # Also check for 'solution' field (MATH dataset format)
        if not answer:
            answer = item.get('solution', '')
        
        if question and answer:
            formatted_data.append(format_qa_pair(question, answer))
    
    return formatted_data


def prepare_multiple_datasets(file_paths: List[str]) -> List[str]:
    """Load and combine multiple datasets."""
    all_data = []
    for file_path in file_paths:
        if os.path.exists(file_path):
            data = prepare_dataset(file_path)
            all_data.extend(data)
            print(f"Loaded {len(data)} examples from {file_path}")
        else:
            print(f"Warning: {file_path} not found, skipping...")
    
    return all_data


if __name__ == "__main__":
    # Test loading
    data = load_jsonl("mathqa_questionanswer.jsonl")
    print(f"Loaded {len(data)} examples")
    print("\nFirst example:")
    print(format_qa_pair(data[0]['question'], data[0]['answer']))
