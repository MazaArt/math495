"""
Download and convert MATH dataset to our format
Based on: https://github.com/hendrycks/math
"""
import json
import os
import requests
import zipfile
from pathlib import Path
from typing import List, Dict


def _passes_level_filter(level_value, allowed_levels):
    if not allowed_levels:
        return True
    if level_value is None:
        return False
    return str(level_value).lower() in {lvl.lower() for lvl in allowed_levels}


def download_math_dataset(
    output_dir: str = "./math_data",
    split: str = "train",
    levels: List[str] = None,
    max_examples: int = None,
):
    """Download MATH dataset from Hugging Face or GitHub."""
    print("Downloading MATH dataset...")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Try downloading from Hugging Face datasets (easier)
    try:
        from datasets import load_dataset
        print("Downloading from Hugging Face...")
        # Try different possible dataset names
        dataset_names = [
            "EleutherAI/hendrycks_math",
            "lighteval/MATH",
            "hendrycks/competition_math", 
            "math_dataset"
        ]
        
        datasets_to_merge = []
        for name in dataset_names:
            try:
                print(f"Trying {name} ({split})...")
                # Special handling for EleutherAI/hendrycks_math which requires configs
                if name == "EleutherAI/hendrycks_math":
                    configs = ['algebra', 'counting_and_probability', 'geometry', 'intermediate_algebra', 'number_theory', 'prealgebra', 'precalculus']
                    all_config_datasets = []
                    for config in configs:
                        try:
                            if split == "all":
                                train_ds = load_dataset(name, config, split="train")
                                test_ds = load_dataset(name, config, split="test")
                                all_config_datasets.extend([train_ds, test_ds])
                            else:
                                all_config_datasets.append(load_dataset(name, config, split=split))
                        except Exception as config_e:
                            print(f"  Warning: Failed to load config {config}: {config_e}")
                            continue
                    if all_config_datasets:
                        datasets_to_merge = all_config_datasets
                        print(f"Successfully loaded {name} with {len(all_config_datasets)} configs!")
                        break
                else:
                    if split == "all":
                        train_ds = load_dataset(name, split="train")
                        test_ds = load_dataset(name, split="test")
                        datasets_to_merge = [train_ds, test_ds]
                    else:
                        datasets_to_merge = [load_dataset(name, split=split)]
                    print(f"Successfully loaded {name}!")
                    break
            except Exception as e:
                print(f"Failed {name}: {e}")
                datasets_to_merge = []
                continue
        
        if not datasets_to_merge:
            raise Exception("Could not find dataset on Hugging Face")
        
        # Convert to our format with optional filtering
        converted_data = []
        allowed_levels = set(levels) if levels else None
        for ds in datasets_to_merge:
            for item in ds:
                problem = item.get("problem", "")
                solution = item.get("solution", "")
                level_value = item.get("level") or item.get("type")
                
                if not _passes_level_filter(level_value, allowed_levels):
                    continue
                
                if problem and solution:
                    converted_data.append({
                        "question": problem,
                        "answer": solution,
                        "level": level_value
                    })
                
                if max_examples and len(converted_data) >= max_examples:
                    break
            if max_examples and len(converted_data) >= max_examples:
                break
        
        # Save as JSONL
        output_file = os.path.join(output_dir, "math_dataset.jsonl")
        with open(output_file, 'w', encoding='utf-8') as f:
            for item in converted_data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        
        print(f"Downloaded {len(converted_data)} examples from Hugging Face")
        print(f"Saved to {output_file}")
        return output_file
        
    except Exception as e:
        print(f"Hugging Face download failed: {e}")
        print("Trying alternative method...")
        
        # Alternative: Download from GitHub releases or use local clone
        print("\nPlease download the MATH dataset manually:")
        print("1. Clone: git clone https://github.com/hendrycks/math.git")
        print("2. Or download from: https://people.eecs.berkeley.edu/~hendrycks/MATH.tar")
        print("3. Then run: python download_math_dataset.py --convert <path_to_math_data>")
        return None


def convert_math_json_to_jsonl(
    math_data_dir: str,
    output_file: str,
    levels: List[str] = None,
    max_examples: int = None,
):
    """Convert MATH dataset JSON files to JSONL format."""
    print(f"Converting MATH dataset from {math_data_dir}...")
    
    converted_data = []
    
    # Walk through all JSON files in the data directory
    data_path = Path(math_data_dir)
    if not data_path.exists():
        print(f"Error: {math_data_dir} does not exist")
        return None
    
    # Look for JSON files in subdirectories
    json_files = list(data_path.rglob("*.json"))
    
    if not json_files:
        # Try common structure: math/data/train/*.json or math/data/test/*.json
        for split in ["train", "test"]:
            split_dir = data_path / split
            if split_dir.exists():
                json_files.extend(list(split_dir.rglob("*.json")))
    
    print(f"Found {len(json_files)} JSON files")
    allowed_levels = set(levels) if levels else None
    
    for json_file in json_files:
        if max_examples and len(converted_data) >= max_examples:
            break
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                problem_data = json.load(f)
                
                # Extract problem and solution
                problem = problem_data.get("problem", "")
                solution = problem_data.get("solution", "")
                level_value = problem_data.get("level") or problem_data.get("type")
                
                if not _passes_level_filter(level_value, allowed_levels):
                    continue
                
                if problem and solution:
                    converted_data.append({
                        "question": problem,
                        "answer": solution,
                        "level": level_value
                    })
        except Exception as e:
            print(f"Error processing {json_file}: {e}")
            continue
    
    # Save as JSONL
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in converted_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    print(f"Converted {len(converted_data)} examples")
    print(f"Saved to {output_file}")
    return output_file


def combine_datasets(files: List[str], output_file: str):
    """Combine multiple JSONL datasets into one."""
    print(f"Combining {len(files)} datasets...")
    
    all_data = []
    for file in files:
        if not os.path.exists(file):
            print(f"Warning: {file} not found, skipping...")
            continue
        
        with open(file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    all_data.append(json.loads(line))
    
    # Save combined dataset
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in all_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    print(f"Combined {len(all_data)} examples into {output_file}")
    return output_file


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Download and convert MATH dataset")
    parser.add_argument("--convert", type=str, help="Path to MATH dataset directory to convert")
    parser.add_argument("--combine", action="store_true", help="Combine with existing mathqa_questionanswer.jsonl")
    parser.add_argument("--output", type=str, default="combined_dataset.jsonl", help="Output file name")
    parser.add_argument("--split", type=str, default="train", choices=["train", "test", "all"], help="Which split to download when using Hugging Face")
    parser.add_argument("--levels", type=str, help="Comma-separated difficulty levels to include (e.g., level1,level2)")
    parser.add_argument("--max-examples", type=int, help="Cap the number of examples to save")
    
    args = parser.parse_args()
    levels = [lvl.strip() for lvl in args.levels.split(",")] if args.levels else None
    
    if args.convert:
        # Convert existing MATH dataset
        output_file = args.output
        convert_math_json_to_jsonl(args.convert, output_file, levels=levels, max_examples=args.max_examples)
        
        if args.combine:
            # Combine with existing dataset
            files = ["math_data/mathqa_questionanswer.jsonl", output_file]
            combine_datasets(files, args.output)
    else:
        # Try to download
        math_file = download_math_dataset(
            split=args.split,
            levels=levels,
            max_examples=args.max_examples
        )
        
        if math_file and args.combine:
            # Combine with existing dataset
            files = ["math_data/mathqa_questionanswer.jsonl", math_file]
            combine_datasets(files, args.output)

