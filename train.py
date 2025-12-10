"""
Training script for MathQA LLM
"""
import os
import json
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from datasets import Dataset
from load_data import prepare_dataset, prepare_multiple_datasets


def main():
    # Configuration
    # Model options (all from Hugging Face):
    # - "gpt2": Small, fast, good for testing
    # - "distilgpt2": Even smaller/faster version
    # - "google/flan-t5-small": Better for Q&A tasks (encoder-decoder)
    # - "google/flan-t5-base": Better performance, larger
    # - "microsoft/DialoGPT-small": Good for conversational Q&A
    # - "EleutherAI/gpt-neo-125M": Alternative GPT architecture
    # - "t5-small": Original T5, good for text-to-text tasks
    
    model_name = "google/flan-t5-small"  # Better for Q&A than GPT-2
    # Alternative: "gpt2" for causal LM, or "t5-small" for encoder-decoder
    output_dir = "./mathqa_model"
    
    # Dataset files - can specify multiple files to combine
    data_files = [
        "mathqa_questionanswer.jsonl",  # Original dataset
        "combined_dataset.jsonl",  # Combined dataset (if created)
        "math_data/math_dataset.jsonl",  # MATH dataset (if downloaded)
    ]
    
    # Use first available dataset file, or combine multiple
    available_files = [f for f in data_files if os.path.exists(f)]
    
    if not available_files:
        print("Error: No dataset files found!")
        print(f"Looking for: {data_files}")
        print("\nTo download MATH dataset, run:")
        print("  python3 download_math_dataset.py")
        print("Or to combine datasets:")
        print("  python3 download_math_dataset.py --combine --output combined_dataset.jsonl")
        return
    
    # Load and prepare data
    print("Loading dataset...")
    if len(available_files) > 1:
        formatted_data = prepare_multiple_datasets(available_files)
    else:
        formatted_data = prepare_dataset(available_files[0])
    
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
        num_train_epochs=3,  # Can increase to 5-10 for larger datasets
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
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

