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


def download_math_dataset(output_dir: str = "./math_data"):
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
            "lighteval/MATH",
            "hendrycks/competition_math", 
            "math_dataset"
        ]
        
        dataset = None
        for name in dataset_names:
            try:
                print(f"Trying {name}...")
                dataset = load_dataset(name, split="train")
                print(f"Successfully loaded {name}!")
                break
            except:
                continue
        
        if dataset is None:
            raise Exception("Could not find dataset on Hugging Face")
        
        # Convert to our format
        converted_data = []
        for item in dataset:
            problem = item.get("problem", "")
            solution = item.get("solution", "")
            if problem and solution:
                converted_data.append({
                    "question": problem,
                    "answer": solution
                })
        
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


def convert_math_json_to_jsonl(math_data_dir: str, output_file: str):
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
    
    for json_file in json_files:
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                problem_data = json.load(f)
                
                # Extract problem and solution
                problem = problem_data.get("problem", "")
                solution = problem_data.get("solution", "")
                
                if problem and solution:
                    converted_data.append({
                        "question": problem,
                        "answer": solution
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
    
    args = parser.parse_args()
    
    if args.convert:
        # Convert existing MATH dataset
        output_file = args.output
        convert_math_json_to_jsonl(args.convert, output_file)
        
        if args.combine:
            # Combine with existing dataset
            files = ["mathqa_questionanswer.jsonl", output_file]
            combine_datasets(files, args.output)
    else:
        # Try to download
        math_file = download_math_dataset()
        
        if math_file and args.combine:
            # Combine with existing dataset
            files = ["mathqa_questionanswer.jsonl", math_file]
            combine_datasets(files, args.output)

