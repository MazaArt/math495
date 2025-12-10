"""
Inference script for trained MathQA model
"""
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM
import torch


def load_model(model_path: str):
    """Load trained model and tokenizer."""
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    # Try to detect model type
    try:
        model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
        model_type = "seq2seq"
    except:
        model = AutoModelForCausalLM.from_pretrained(model_path)
        model_type = "causal"
    
    return tokenizer, model, model_type


def generate_answer(question: str, tokenizer, model, model_type: str, max_length=200):
    """Generate answer for a given question."""
    
    if model_type == "seq2seq":
        # For T5/FLAN models: use task prefix and question (matching training format)
        # During training, we removed "Question:" prefix, so input is just the question text
        # But FLAN-T5 works better with a task instruction
        prompt = f"solve: {question}"
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=256, padding=True)
        # Ensure attention mask is set
        if "attention_mask" not in inputs:
            inputs["attention_mask"] = (inputs["input_ids"] != tokenizer.pad_token_id).long()
    else:
        # For GPT-style models: use full prompt
        prompt = f"Question: {question}\nAnswer:"
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=256, padding=True)
        if "attention_mask" not in inputs:
            inputs["attention_mask"] = (inputs["input_ids"] != tokenizer.pad_token_id).long()
    
    # Generate with better parameters
    with torch.no_grad():
        if model_type == "seq2seq":
            outputs = model.generate(
                inputs["input_ids"],
                attention_mask=inputs.get("attention_mask"),
                max_length=max_length,
                min_length=10,  # Ensure minimum answer length
                num_return_sequences=1,
                temperature=0.3,  # Lower temperature for more focused answers
                do_sample=True,
                top_p=0.9,  # Nucleus sampling
                repetition_penalty=1.2,  # Reduce repetition
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
        else:
            outputs = model.generate(
                inputs,
                max_length=max_length,
                min_length=10,
                num_return_sequences=1,
                temperature=0.7,
                do_sample=True,
                top_p=0.9,
                repetition_penalty=1.2,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
    
    # Decode
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract answer part for causal models
    if model_type == "causal" and "Answer:" in generated_text:
        answer = generated_text.split("Answer:")[-1].strip()
    elif model_type == "causal":
        answer = generated_text[len(prompt):].strip()
    else:
        # For seq2seq, the output is already the answer
        answer = generated_text.strip()
    
    return answer


def main():
    model_path = "./mathqa_model"
    
    print("Loading model...")
    try:
        tokenizer, model, model_type = load_model(model_path)
        print(f"Model loaded successfully! (Type: {model_type})")
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Make sure you've trained the model first using train.py")
        return
    
    # Interactive loop
    print("\nMathQA Inference - Type 'quit' to exit\n")
    while True:
        question = input("Question: ")
        if question.lower() in ['quit', 'exit', 'q']:
            break
        
        answer = generate_answer(question, tokenizer, model, model_type)
        print(f"Answer: {answer}\n")


if __name__ == "__main__":
    main()

