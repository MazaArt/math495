"""
Training script for MathQA LLM
"""
import os
import argparse
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from datasets import Dataset
from load_data import prepare_dataset, prepare_multiple_datasets


def parse_args():
    parser = argparse.ArgumentParser(description="Train Math LLM on MathQA + MATH")
    parser.add_argument("--model-name", type=str, default="google/flan-t5-small", help="Hugging Face model name")
    parser.add_argument("--output-dir", type=str, default="./mathqa_model", help="Where to save checkpoints")
    parser.add_argument("--mathqa", action=argparse.BooleanOptionalAction, default=True, help="Include MathQA dataset")
    parser.add_argument("--math", action=argparse.BooleanOptionalAction, default=True, help="Include MATH dataset (downloaded subset)")
    parser.add_argument("--mathqa-path", type=str, default="math_data/mathqa_questionanswer.jsonl", help="Path to MathQA jsonl")
    parser.add_argument("--math-path", type=str, default="math_data/math_dataset.jsonl", help="Path to MATH subset jsonl")
    parser.add_argument("--combined-path", type=str, default="combined_dataset.jsonl", help="Optional precombined jsonl path")
    parser.add_argument("--math-levels", type=str, help="Comma-separated difficulty levels to include from MATH")
    parser.add_argument("--max-examples", type=int, help="Total max examples after merge")
    parser.add_argument("--max-examples-per-dataset", type=int, help="Max examples to keep per dataset before merge")
    parser.add_argument("--num-train-epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--per-device-train-batch-size", type=int, default=4, help="Training batch size per device")
    parser.add_argument("--per-device-eval-batch-size", type=int, default=4, help="Evaluation batch size per device")
    return parser.parse_args()


def main():
    args = parse_args()
    model_name = args.model_name
    output_dir = args.output_dir
    
    math_levels = [lvl.strip() for lvl in args.math_levels.split(",")] if args.math_levels else None
    
    candidate_files = []
    if args.math and os.path.exists(args.math_path):
        candidate_files.append(args.math_path)
        print(f"Including MATH subset: {args.math_path}")
    elif args.math:
        print(f"Requested MATH subset but missing file: {args.math_path}")
    
    if args.mathqa and os.path.exists(args.mathqa_path):
        candidate_files.append(args.mathqa_path)
        print(f"Including MathQA: {args.mathqa_path}")
    elif args.mathqa:
        print(f"Requested MathQA but missing file: {args.mathqa_path}")
    
    # Allow optional precombined file
    if os.path.exists(args.combined_path):
        print(f"Found precombined dataset: {args.combined_path} (will be merged as well)")
        candidate_files.append(args.combined_path)
    
    available_files = candidate_files
    
    if not available_files:
        print("Error: No dataset files found!")
        print("To download MATH dataset, run:")
        print("  python3 download_math_dataset.py --split train --levels level1,level2")
        print("Or to combine datasets:")
        print("  python3 download_math_dataset.py --combine --output combined_dataset.jsonl")
        return
    
    # Load and prepare data
    print("Loading dataset...")
    print(f"Datasets to load: {available_files}")
    if len(available_files) > 1:
        formatted_data = prepare_multiple_datasets(
            available_files,
            allowed_levels=math_levels,
            max_examples_per_dataset=args.max_examples_per_dataset,
            total_max_examples=args.max_examples,
        )
    else:
        formatted_data = prepare_dataset(
            available_files[0],
            allowed_levels=math_levels,
            max_examples=args.max_examples,
        )
    
    # Split into train/validation (80/20)
    split_idx = int(len(formatted_data) * 0.8)
    train_data = formatted_data[:split_idx]
    val_data = formatted_data[split_idx:]
    
    print(f"Train examples: {len(train_data)}")
    print(f"Validation examples: {len(val_data)}")
    
    # Load tokenizer and model
    print(f"Loading model: {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Check if it's a T5/FLAN model (encoder-decoder) or GPT-style (decoder-only)
    if "t5" in model_name.lower() or "flan" in model_name.lower():
        from transformers import AutoModelForSeq2SeqLM
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        model_type = "seq2seq"
    else:
        model = AutoModelForCausalLM.from_pretrained(model_name)
        model_type = "causal"
    
    # Add padding token if it doesn't exist
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Tokenize data
    def tokenize_function(examples):
        if model_type == "seq2seq":
            # For T5/FLAN: split into input (question) and target (answer)
            inputs = []
            targets = []
            for text in examples["text"]:
                if "Question:" in text and "Answer:" in text:
                    parts = text.split("Answer:")
                    inputs.append(parts[0].replace("Question:", "").strip())
                    targets.append(parts[1].strip() if len(parts) > 1 else "")
                else:
                    inputs.append(text)
                    targets.append("")
            
            model_inputs = tokenizer(
                inputs,
                truncation=True,
                max_length=256,
                padding="max_length"
            )
            labels = tokenizer(
                targets,
                truncation=True,
                max_length=256,
                padding="max_length"
            )
            # For T5 models, set padding tokens in labels to -100 (ignored in loss)
            labels["input_ids"] = [
                [(l if l != tokenizer.pad_token_id else -100) for l in label]
                for label in labels["input_ids"]
            ]
            model_inputs["labels"] = labels["input_ids"]
            return model_inputs
        else:
            # For GPT-style models: use full text
            return tokenizer(
                examples["text"],
                truncation=True,
                max_length=512,
                padding="max_length"
            )
    
    # Create datasets
    train_dataset = Dataset.from_dict({"text": train_data})
    val_dataset = Dataset.from_dict({"text": val_data})
    
    train_dataset = train_dataset.map(tokenize_function, batched=True)
    val_dataset = val_dataset.map(tokenize_function, batched=True)
    
    # Data collator
    if model_type == "seq2seq":
        from transformers import DataCollatorForSeq2Seq
        data_collator = DataCollatorForSeq2Seq(
            tokenizer=tokenizer,
            padding=True
        )
    else:
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False  # Causal LM, not masked LM
        )
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        warmup_steps=500,  # Increased for larger dataset
        logging_steps=50,
        eval_steps=200,  # Less frequent for larger dataset
        save_steps=200,  # Must be multiple of eval_steps for load_best_model_at_end
        eval_strategy="steps",
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        learning_rate=3e-5,  # Slightly lower for more stable training with large dataset
        fp16=False,
        gradient_accumulation_steps=2,  # Effective batch size = 4 * 2 = 8
        dataloader_num_workers=2,  # Speed up data loading
    )
    
    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
    )
    
    # Train
    print("Starting training...")
    trainer.train()
    
    # Save final model
    print(f"Saving model to {output_dir}...")
    trainer.save_model()
    tokenizer.save_pretrained(output_dir)
    
    print("Training complete!")


if __name__ == "__main__":
    main()

