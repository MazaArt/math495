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
    prompt = f"Question: {question}\nAnswer: {answer}"
    return prompt


def _passes_level_filter(item: Dict, allowed_levels) -> bool:
    if not allowed_levels:
        return True
    level_value = item.get("level") or item.get("type")
    if level_value is None:
        return True  # Keep examples without level metadata (e.g., MathQA)
    return str(level_value).lower() in allowed_levels


def prepare_dataset(file_path: str, allowed_levels=None, max_examples: int = None) -> List[str]:
    """Load and format the dataset for training."""
    data = load_jsonl(file_path)
    formatted_data = []
    allowed_levels_normalized = {lvl.lower() for lvl in allowed_levels} if allowed_levels else None
    
    for item in data:
        if max_examples and len(formatted_data) >= max_examples:
            break
        
        if not _passes_level_filter(item, allowed_levels_normalized):
            continue
        
        question = item.get('question', '')
        answer = item.get('answer', '')
        if not answer:
            answer = item.get('solution', '')
        
        if question and answer:
            formatted_data.append(format_qa_pair(question, answer))
    
    return formatted_data


def prepare_multiple_datasets(
    file_paths: List[str],
    allowed_levels=None,
    max_examples_per_dataset: int = None,
    total_max_examples: int = None,
) -> List[str]:
    """Load and combine multiple datasets with optional filtering and deduplication."""
    all_data = []
    seen_prompts = set()
    for file_path in file_paths:
        if not os.path.exists(file_path):
            print(f"Warning: {file_path} not found, skipping...")
            continue
        
        data = prepare_dataset(
            file_path,
            allowed_levels=allowed_levels,
            max_examples=max_examples_per_dataset
        )
        before_len = len(all_data)
        for prompt in data:
            if prompt in seen_prompts:
                continue
            if total_max_examples and len(all_data) >= total_max_examples:
                break
            seen_prompts.add(prompt)
            all_data.append(prompt)
        print(f"Loaded {len(data)} examples from {file_path} ({len(all_data) - before_len} kept after dedup)")
        
        if total_max_examples and len(all_data) >= total_max_examples:
            break
    
    return all_data


if __name__ == "__main__":
    # Test loading
    data = load_jsonl("mathqa_questionanswer.jsonl")
    print(f"Loaded {len(data)} examples")
    print("\nFirst example:")
    print(format_qa_pair(data[0]['question'], data[0]['answer']))
