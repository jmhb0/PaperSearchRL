import transformers
import torch
import pandas as pd
from datasets import load_dataset
import os
import re
from tqdm import tqdm
from verl.utils.reward_score.qa_em import compute_score_em
import ipdb

# Create results directory
os.makedirs("results/eval_direct_inference", exist_ok=True)

# Model ID and device setup (same as infer.py)
model_id = 'Qwen/Qwen2.5-3B-Instruct'
# checkpoint_path = "data/verl_checkpoints/20250604_grpo_bioasqv0_fullcorpus_qwenit_bm25/actor/global_step_100/"
checkpoint_path = None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Template for factoid questions
QUESTION_TEMPLATE_FACTOID = "{question}\nThink step by step. Your final answer should be a short 'factoid' with only a few words maximum. Put that answer inside <answer>...</answer>"

def load_model_and_tokenizer(model_id, checkpoint_path=None):
    """Load model and tokenizer, optionally from a checkpoint."""
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_id)

    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f"Loading model from checkpoint: {checkpoint_path}")
        model = transformers.AutoModelForCausalLM.from_pretrained(
            checkpoint_path, torch_dtype=torch.bfloat16, device_map="auto")
    else:
        print(f"Loading model from HuggingFace: {model_id}")
        model = transformers.AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=torch.bfloat16, device_map="auto")

    return tokenizer, model

def extract_answer(text):
    """Extract answer from <answer>...</answer> tags."""
    pattern = re.compile(r"<answer>(.*?)</answer>", re.DOTALL | re.IGNORECASE)
    matches = pattern.findall(text)
    if matches:
        return matches[-1].strip()
    else:
        return ""

def direct_inference(question, tokenizer, model):
    """Run direct inference for a single question without search."""
    # Format the question using the template
    prompt = QUESTION_TEMPLATE_FACTOID.format(question=question.strip())
    
    # Apply chat template if available
    if tokenizer.chat_template:
        prompt = tokenizer.apply_chat_template([{
            "role": "user",
            "content": prompt
        }], add_generation_prompt=True, tokenize=False)
    
    # Encode and move to device
    input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
    attention_mask = torch.ones_like(input_ids)
    
    # Generate response
    with torch.no_grad():
        outputs = model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=512,
            pad_token_id=tokenizer.eos_token_id,
            do_sample=True,
            temperature=0.7
        )
    
    # Decode the generated tokens
    generated_tokens = outputs[0][input_ids.shape[1]:]
    output_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    
    return output_text

def main():
    print("Loading model and tokenizer...")
    tokenizer, model = load_model_and_tokenizer(model_id, checkpoint_path)
    
    print("Loading dataset...")
    # Load the test dataset
    dataset = load_dataset("jmhb/bioasq_trainv0_n1609_test100", split="test")
    
    # Convert to DataFrame
    df = dataset.to_pandas()
    print(f"Loaded {len(df)} test examples")
    
    # Initialize results
    results = []
    
    print("Running direct inference...")
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing questions"):
        question = row['question']
        golden_answers = row['golden_answers']
        
        # Run direct inference
        output_text = direct_inference(question, tokenizer, model)
        
        # Extract answer
        extracted_answer = extract_answer(output_text)
        
        # Format ground truth for compute_score_em function
        # The function expects a dict with 'target' key containing the list of golden answers
        ground_truth = {'target': golden_answers}
        
        # Compute correctness score
        score = compute_score_em(
            solution_str=output_text,  # Pass the full output text for extraction
            ground_truth=ground_truth,
            method='strict',
            format_score=0.0,
            score=1.0
        )
        
        # Store results
        result = {
            'question': question,
            'golden_answers': golden_answers,
            'generated_text': output_text,
            'extracted_answer': extracted_answer,
            'correctness_score': score
        }
        results.append(result)
    
    # Create results DataFrame
    results_df = pd.DataFrame(results)
    
    # Calculate and print mean correctness score
    mean_score = results_df['correctness_score'].mean()
    print(f"\nMean correctness score: {mean_score:.4f}")
    print(f"Total correct answers: {(results_df['correctness_score'] > 0).sum()}/{len(results_df)}")
    
    # Save results to CSV
    output_path = "results/eval_direct_inference/results.csv"
    results_df.to_csv(output_path, index=False)
    print(f"Results saved to: {output_path}")
    
    # Save summary statistics
    summary_stats = {
        'mean_correctness_score': mean_score,
        'total_questions': len(results_df),
        'correct_answers': (results_df['correctness_score'] > 0).sum(),
        'accuracy_percentage': (results_df['correctness_score'] > 0).mean() * 100
    }
    
    summary_df = pd.DataFrame([summary_stats])
    summary_path = "results/eval_direct_inference/summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"Summary statistics saved to: {summary_path}")
    ipdb.set_trace()
    pass

if __name__ == "__main__":

    main() 