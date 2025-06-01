"""
BioASQ Inference Script
Loads BioASQ dataset, filters for factoid questions, and runs LLM inference.
"""
import sys
import os
import pandas as pd
import ipdb
import re
from datasets import load_dataset

# Add parent directory to path to import api.py
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from api import call_llm_batch

# Global template for questions
QUESTION_TEMPLATE_FACTOID = "{question}\nThink step by step. Your final answer should be a short 'factoid' with only a few words maximum. Put that answer inside <answer>...</answer>"

JUDGE_TEMPLATE = """\
# LLM Judge Prompt for Factoid Answer Evaluation

You are an expert judge evaluating factoid answers. Determine if a predicted answer matches the ground truth.

**Ground Truth Answers:** {ground_truth_answers}
**Predicted Answer:** {predicted_answer}

## Scoring:
- **Score 1**: Semantically equivalent (same entity, synonyms, abbreviations, minor wording differences)
- **Score 0.5**: Less specific but correct (predicted answer is a broader category that contains the ground truth)
- **Score 0**: Incorrect (different entity, factually wrong, or unrelated)

## Examples:
- GT: "toe", Pred: "foot" → **Score 0.5** (less specific but correct)
- GT: "FSHD", Pred: "Facioscapulohumeral muscular dystrophy" → **Score 1** (equivalent)
- GT: "Paris", Pred: "London" → **Score 0** (incorrect)

## Output Format:
<score>[0, 0.5, or 1]</score>
<reasoning>[Brief explanation]</reasoning>
"""


def parse_answer_from_response(response: str) -> str:
    """
    Parse the answer from <answer>...</answer> tags in the LLM response.
    
    Args:
        response: The full LLM response string
        
    Returns:
        The extracted answer string, or "ERROR" if formatting is incorrect
    """
    if not isinstance(response, str):
        return "ERROR"
    
    # Use regex to find content between <answer> and </answer> tags
    pattern = r'<answer>(.*?)</answer>'
    matches = re.findall(pattern, response, re.DOTALL | re.IGNORECASE)
    
    if len(matches) == 1:
        # Return the answer, stripped of leading/trailing whitespace
        return matches[0].strip()
    elif len(matches) == 0:
        # No answer tags found
        return "ERROR"
    else:
        # Multiple answer tags found - ambiguous
        return "ERROR"


def parse_judge_score_from_response(response: str) -> float:
    """
    Parse the score from the judge LLM response.
    
    Args:
        response: The full judge LLM response string
        
    Returns:
        The extracted score as float (0, 0.5, or 1), or -1 if parsing failed
    """
    if not isinstance(response, str):
        return -1
    
    # Use regex to find content between <score> and </score> tags
    pattern = r'<score>(.*?)</score>'
    matches = re.findall(pattern, response, re.DOTALL | re.IGNORECASE)
    
    if len(matches) == 1:
        try:
            score_text = matches[0].strip()
            score = float(score_text)
            # Validate that score is one of the expected values
            if score in [0, 0.5, 1]:
                return score
            else:
                return -1
        except ValueError:
            return -1
    else:
        return -1


def main(num_questions: int = 100):
    """
    Main function to run BioASQ inference.
    
    Args:
        num_questions: Number of questions to process (default: 20)
    """
    print("Loading BioASQ dataset from Hugging Face...")
    
    # Load the dataset
    dataset = load_dataset("jmhb/BioASQ-taskb")
    
    # Convert to pandas DataFrame for easier manipulation
    df = pd.DataFrame(dataset['train'])
    
    print(f"Original dataset size: {len(df)}")
    print(f"Dataset columns: {df.columns.tolist()}")
    
    # Filter for factoid questions
    factoid_df = df[df['type'] == 'factoid'].copy()
    print(f"Factoid questions: {len(factoid_df)}")
    
    # Shuffle the DataFrame with random seed 0
    factoid_df = factoid_df.sample(frac=1, random_state=0).reset_index(drop=True)
    
    # Filter for first num_questions questions
    factoid_df = factoid_df.head(num_questions).copy()
    print(f"Using first {num_questions} factoid questions: {len(factoid_df)}")
    
    
    # Create a new DataFrame with just question and answer columns
    inference_df = factoid_df[['question', 'answer']].copy()
    
    # List of models to test
    model_names = ['openai/gpt-4o-mini']  # Note: using gpt-4o-mini as gpt-4.1-mini doesn't exist
    model_names = ['qwen/qwen3-8b']  # Note: using gpt-4o-mini as gpt-4.1-mini doesn't exist
    
    # Loop over model names
    for model_name in model_names:
        print(f"\nRunning inference with model: {model_name}")
        
        # Construct questions using the template
        questions = []
        for question in inference_df['question']:
            questions.append(QUESTION_TEMPLATE_FACTOID.format(question=question))
        
        print(f"Running batch inference on {len(questions)} questions...")
        
        # Use call_llm_batch to run all questions
        try:
            responses, costs = call_llm_batch(
                prompts=questions,
                model_name=model_name,
                use_cache=True,
                max_tokens=500,
                temperature=0.1,  # Lower temperature for more consistent factual answers
                include_cost=True,
                max_concurrent=2
            )
            
            # Verify we got the same number of responses
            assert len(responses) == len(inference_df), f"Expected {len(inference_df)} responses, got {len(responses)}"
            
            # Create column names
            model_name_clean = model_name.replace('/', '_').replace('-', '_')
            cot_column_name = f"cot_{model_name_clean}"
            pred_column_name = f"pred_{model_name_clean}"
            
            # Add full LLM responses (chain of thought)
            inference_df[cot_column_name] = responses
            
            # Parse answers from responses
            parsed_answers = [parse_answer_from_response(response) for response in responses]
            inference_df[pred_column_name] = parsed_answers
            
            print(f"✅ Successfully added predictions to columns: {cot_column_name}, {pred_column_name}")
            print(f"Total cost: ${sum(costs):.8f} USD")
            
        except Exception as e:
            print(f"❌ Error running inference with {model_name}: {str(e)}")
            # Add error columns with error messages
            model_name_clean = model_name.replace('/', '_').replace('-', '_')
            cot_column_name = f"cot_{model_name_clean}"
            pred_column_name = f"pred_{model_name_clean}"
            inference_df[cot_column_name] = [f"ERROR: {str(e)}"] * len(inference_df)
            inference_df[pred_column_name] = ["ERROR"] * len(inference_df)
    
    # Now run LLM judge evaluation for each model's predictions
    print("\n" + "="*80)
    print("RUNNING LLM JUDGE EVALUATION")
    print("="*80)
    
    judge_model = 'openai/gpt-4o-mini'  # Use same model for judging
    
    for col in inference_df.columns:
        if col.startswith('pred_'):
            model_name_from_col = col.replace('pred_', '')
            score_column_name = f"score_{model_name_from_col}"
            judge_column_name = f"judge_{model_name_from_col}"
            
            print(f"\nEvaluating predictions from column: {col}")
            
            # Create judge prompts
            judge_prompts = []
            for i, row in inference_df.iterrows():
                ground_truth_answers = row['answer']
                predicted_answer = row[col]
                
                # Skip if prediction is ERROR
                if predicted_answer == "ERROR":
                    judge_prompts.append("SKIP")
                    continue
                
                judge_prompt = JUDGE_TEMPLATE.format(
                    ground_truth_answers=ground_truth_answers,
                    predicted_answer=predicted_answer
                )
                judge_prompts.append(judge_prompt)
            
            # Filter out SKIP prompts and keep track of indices
            valid_prompts = []
            valid_indices = []
            for i, prompt in enumerate(judge_prompts):
                if prompt != "SKIP":
                    valid_prompts.append(prompt)
                    valid_indices.append(i)
            
            print(f"Running judge evaluation on {len(valid_prompts)} valid predictions...")
            
            try:
                if len(valid_prompts) > 0:
                    judge_responses, judge_costs = call_llm_batch(
                        prompts=valid_prompts,
                        model_name=judge_model,
                        use_cache=True,
                        max_tokens=200,
                        temperature=0.0,  # Deterministic judging
                        include_cost=True,
                        max_concurrent=100
                    )
                    
                    # Parse scores from judge responses
                    scores = [parse_judge_score_from_response(response) for response in judge_responses]
                    
                    # Create full scores array with -1 for ERROR predictions
                    full_scores = [-1] * len(inference_df)
                    full_judge_responses = ["ERROR: No prediction to judge"] * len(inference_df)
                    
                    for i, (score, response) in enumerate(zip(scores, judge_responses)):
                        full_scores[valid_indices[i]] = score
                        full_judge_responses[valid_indices[i]] = response
                    
                    # Add scores and judge responses to dataframe
                    inference_df[score_column_name] = full_scores
                    inference_df[judge_column_name] = full_judge_responses
                    
                    print(f"✅ Successfully added judge scores to column: {score_column_name}")
                    print(f"✅ Successfully added judge responses to column: {judge_column_name}")
                    print(f"Judge cost: ${sum(judge_costs):.8f} USD")
                    
                    # Print score distribution
                    score_counts = pd.Series(full_scores).value_counts().sort_index()
                    print(f"Score distribution: {score_counts.to_dict()}")
                    
                else:
                    # All predictions were ERROR
                    inference_df[score_column_name] = [-1] * len(inference_df)
                    inference_df[judge_column_name] = ["ERROR: No prediction to judge"] * len(inference_df)
                    print(f"⚠️ All predictions were ERROR for {col}")
                    
            except Exception as e:
                print(f"❌ Error running judge evaluation for {col}: {str(e)}")
                inference_df[score_column_name] = [-1] * len(inference_df)
                inference_df[judge_column_name] = [f"ERROR: {str(e)}"] * len(inference_df)
    
    # Display some results
    print("\n" + "="*80)
    print("INFERENCE RESULTS SUMMARY")
    print("="*80)
    print(f"Total questions processed: {len(inference_df)}")
    print(f"Columns in final DataFrame: {inference_df.columns.tolist()}")

    
    # Show first few examples
    print("\nFirst 3 examples:")
    for i in range(min(3, len(inference_df))):
        print(f"\n--- Example {i+1} ---")
        print(f"Question: {inference_df.iloc[i]['question']}")
        print(f"Ground Truth: {inference_df.iloc[i]['answer']}")
        for col in inference_df.columns:
            if col.startswith('pred_'):
                print(f"Prediction ({col}): {inference_df.iloc[i][col]}")
            elif col.startswith('cot_'):
                print(f"Chain of Thought ({col}): {inference_df.iloc[i][col][:100]}...")
            elif col.startswith('score_'):
                print(f"Judge Score ({col}): {inference_df.iloc[i][col]}")
            elif col.startswith('judge_'):
                print(f"Judge Response ({col}): {inference_df.iloc[i][col][:200]}...")
                if len(inference_df.iloc[i][col]) > 200:
                    print(f"  [Response truncated - full response saved in CSV]")
    
    # Show evaluation summary
    print("\n" + "="*80)
    print("EVALUATION SUMMARY")
    print("="*80)
    
    for col in inference_df.columns:
        if col.startswith('score_'):
            model_name = col.replace('score_', '')
            scores = inference_df[col]
            
            # Calculate metrics
            valid_scores = [s for s in scores if s >= 0]  # Exclude -1 (ERROR) scores
            if len(valid_scores) > 0:
                avg_score = sum(valid_scores) / len(valid_scores)
                perfect_matches = sum(1 for s in valid_scores if s == 1.0)
                partial_matches = sum(1 for s in valid_scores if s == 0.5)
                incorrect = sum(1 for s in valid_scores if s == 0.0)
                
                print(f"\nModel: {model_name}")
                print(f"  Valid evaluations: {len(valid_scores)}/{len(scores)}")
                print(f"  Average score: {avg_score:.3f}")
                print(f"  Perfect matches (1.0): {perfect_matches} ({perfect_matches/len(valid_scores)*100:.1f}%)")
                print(f"  Partial matches (0.5): {partial_matches} ({partial_matches/len(valid_scores)*100:.1f}%)")
                print(f"  Incorrect (0.0): {incorrect} ({incorrect/len(valid_scores)*100:.1f}%)")
            else:
                print(f"\nModel: {model_name}")
                print(f"  No valid evaluations (all predictions were ERROR)")
    
    # Save results
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)
    output_file = os.path.join(results_dir, "bioasq_inference_results.csv")
    inference_df.to_csv(output_file, index=False)
    print(f"\n💾 Results saved to: {output_file}")
    ipdb.set_trace()
    
    return inference_df


if __name__ == "__main__":
    results_df = main() 
