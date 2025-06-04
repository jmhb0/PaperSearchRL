"""
python data_gen/inference_qwen_local.py
BioASQ Local Inference Script with vLLM

Main functionality:
- Loads BioASQ or custom datasets from Hugging Face
- Filters for factoid questions (if applicable)
- Runs local LLM inference using vLLM with multiple Qwen models
- Evaluates predictions using LLM judge (API or local)
- Computes exact match scores when golden_answers are available
- Supports two dataset options:
  * jmhb/BioASQ-taskb (default)
  * jmhb/bioasq_trainv0_n1609
- Default processes 50 questions with configurable judge backend
"""
import sys
import os
import pandas as pd
import ipdb
import re
from datasets import load_dataset
from vllm import LLM, SamplingParams
from typing import List, Tuple, Optional
import torch
from api import call_llm_batch

# Global template for questions (same as bioasq_inference.py)
QUESTION_TEMPLATE_FACTOID = "{question}\nThink step by step. Your final answer should be a short 'factoid' with only a few words maximum. Put that answer inside <answer>...</answer>"

QUESTION_TEMPLATE_RAG = """Based on the following context, answer the question below.

Context:
{context}

Question: {question}

Think step by step using the provided context. Your final answer should be a short 'factoid' with only a few words maximum.

Put that answer inside <answer>...</answer>"""

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


def run_vllm_inference(model_name: str,
                       prompts: List[str],
                       max_tokens: int = 500,
                       temperature: float = 0.1) -> List[str]:
    """
    Run inference using vLLM on a list of prompts.
    
    Args:
        model_name: The model name/path for vLLM
        prompts: List of prompt strings
        max_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        
    Returns:
        List of generated responses
    """
    print(f"Loading model {model_name} with vLLM...")

    # Initialize the LLM
    llm = LLM(
        model=model_name,
        tensor_parallel_size=1,  # Adjust based on your GPU setup
        gpu_memory_utilization=0.8,  # Adjust based on your GPU memory
        trust_remote_code=True,
    )

    # Set up sampling parameters
    sampling_params = SamplingParams(
        temperature=temperature,
        max_tokens=max_tokens,
        stop=None,  # Let the model decide when to stop
    )

    print(f"Running inference on {len(prompts)} prompts...")

    # Generate responses
    outputs = llm.generate(prompts, sampling_params)

    # Extract the generated text from outputs
    responses = []
    for output in outputs:
        generated_text = output.outputs[0].text
        responses.append(generated_text)

    # Clean up GPU memory
    del llm
    torch.cuda.empty_cache()

    return responses


def compute_exact_match(predicted_answer: str,
                        golden_answers: List[str]) -> bool:
    """
    Compute exact match between predicted answer and golden answers.
    
    Args:
        predicted_answer: The predicted answer string
        golden_answers: List of golden answer strings
        
    Returns:
        True if any golden answer is a case-insensitive substring of the predicted answer
    """
    if predicted_answer == "ERROR" or not isinstance(predicted_answer, str):
        return False

    if not isinstance(golden_answers, list):
        return False

    # Convert predicted answer to lowercase for case-insensitive comparison
    predicted_lower = predicted_answer.lower()

    # Check if any golden answer is a substring of the predicted answer
    for golden_answer in golden_answers:
        if isinstance(golden_answer,
                      str) and golden_answer.lower() in predicted_lower:
            return True

    return False


def load_and_filter_data(
        num_questions: int = 50,
        dataset_name: str = "jmhb/BioASQ-taskb") -> pd.DataFrame:
    """
    Load BioASQ dataset and filter for factoid questions.
    
    Args:
        num_questions: Number of questions to process
        dataset_name: Name of the Hugging Face dataset to load
        
    Returns:
        DataFrame with filtered questions
    """
    print(f"Loading {dataset_name} dataset from Hugging Face...")

    # Load the dataset
    dataset = load_dataset(dataset_name)

    # Convert to pandas DataFrame for easier manipulation
    df = pd.DataFrame(dataset['train'])

    print(f"Original dataset size: {len(df)}")
    print(f"Dataset columns: {df.columns.tolist()}")

    # Filter for factoid questions if type column exists
    if 'type' in df.columns:
        factoid_df = df[df['type'] == 'factoid'].copy()
        print(f"Factoid questions: {len(factoid_df)}")
    else:
        factoid_df = df.copy()
        print(
            f"No 'type' column found, using all questions: {len(factoid_df)}")

    # Shuffle the DataFrame with random seed 0
    factoid_df = factoid_df.sample(frac=1,
                                   random_state=0).reset_index(drop=True)

    # Filter for first num_questions questions
    factoid_df = factoid_df.head(num_questions).copy()
    print(f"Using first {num_questions} questions: {len(factoid_df)}")

    # Determine which columns to keep
    columns_to_keep = ['question', 'answer']
    if 'golden_answers' in factoid_df.columns:
        columns_to_keep.append('golden_answers')
        print(
            "Found 'golden_answers' column - will compute exact match scores")

    # Create a new DataFrame with the selected columns
    inference_df = factoid_df[columns_to_keep].copy()

    return inference_df


def run_judge_evaluation_flexible(inference_df: pd.DataFrame,
                                  judge_inference_fn) -> pd.DataFrame:
    """
    Run LLM judge evaluation for model predictions using a flexible inference function.
    
    Args:
        inference_df: DataFrame with predictions
        judge_inference_fn: Function that takes a list of prompts and returns responses
        
    Returns:
        DataFrame with judge scores added
    """
    print("\n" + "=" * 80)
    print("RUNNING LLM JUDGE EVALUATION")
    print("=" * 80)

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
                    predicted_answer=predicted_answer)
                judge_prompts.append(judge_prompt)

            # Filter out SKIP prompts and keep track of indices
            valid_prompts = []
            valid_indices = []
            for i, prompt in enumerate(judge_prompts):
                if prompt != "SKIP":
                    valid_prompts.append(prompt)
                    valid_indices.append(i)

            print(
                f"Running judge evaluation on {len(valid_prompts)} valid predictions..."
            )

            try:
                if len(valid_prompts) > 0:
                    # Use the flexible inference function
                    judge_responses = judge_inference_fn(valid_prompts)

                    # Parse scores from judge responses
                    scores = [
                        parse_judge_score_from_response(response)
                        for response in judge_responses
                    ]

                    # Create full scores array with -1 for ERROR predictions
                    full_scores = [-1] * len(inference_df)
                    full_judge_responses = ["ERROR: No prediction to judge"
                                            ] * len(inference_df)

                    for i, (score,
                            response) in enumerate(zip(scores,
                                                       judge_responses)):
                        full_scores[valid_indices[i]] = score
                        full_judge_responses[valid_indices[i]] = response

                    # Add scores and judge responses to dataframe
                    inference_df[score_column_name] = full_scores
                    inference_df[judge_column_name] = full_judge_responses

                    print(
                        f"✅ Successfully added judge scores to column: {score_column_name}"
                    )
                    print(
                        f"✅ Successfully added judge responses to column: {judge_column_name}"
                    )

                    # Print score distribution
                    score_counts = pd.Series(
                        full_scores).value_counts().sort_index()
                    print(f"Score distribution: {score_counts.to_dict()}")

                else:
                    # All predictions were ERROR
                    inference_df[score_column_name] = [-1] * len(inference_df)
                    inference_df[judge_column_name] = [
                        "ERROR: No prediction to judge"
                    ] * len(inference_df)
                    print(f"⚠️ All predictions were ERROR for {col}")

            except Exception as e:
                print(f"❌ Error running judge evaluation for {col}: {str(e)}")
                inference_df[score_column_name] = [-1] * len(inference_df)
                inference_df[judge_column_name] = [f"ERROR: {str(e)}"
                                                   ] * len(inference_df)

    return inference_df


def main(num_questions: int = 50,
         judge_backend: str = 'api',
         dataset_name: str = "jmhb/BioASQ-taskb"):
    """
    Main function to run BioASQ inference with local vLLM models.
    
    Args:
        num_questions: Number of questions to process (default: 50)
        judge_backend: Backend to use for judge evaluation ('api' or 'local')
        dataset_name: Name of the Hugging Face dataset to load. Options:
                     - "jmhb/BioASQ-taskb" (default)
                     - "jmhb/bioasq_trainv0_n1609"
    """
    # Load and filter data
    inference_df = load_and_filter_data(num_questions, dataset_name)

    # Check if we have golden_answers column for exact match computation
    has_golden_answers = 'golden_answers' in inference_df.columns

    # List of Qwen models to test
    model_name = 'Qwen/Qwen2.5-3B-Instruct'
    # 'Qwen/Qwen2.5-3B',
    # 'Qwen/Qwen2.5-7B-Instruct',
    # 'Qwen/Qwen2.5-7B'

    # Loop over model names
    print(f"\nRunning inference with model: {model_name}")

    # Construct questions using the template
    questions = []
    for question in inference_df['question']:
        questions.append(QUESTION_TEMPLATE_FACTOID.format(question=question))

    print(f"Running vLLM inference on {len(questions)} questions...")

    try:
        # Run vLLM inference
        responses = run_vllm_inference(model_name=model_name,
                                       prompts=questions,
                                       max_tokens=500,
                                       temperature=0.1)

        # Verify we got the same number of responses
        assert len(responses) == len(
            inference_df
        ), f"Expected {len(inference_df)} responses, got {len(responses)}"

        # Create column names
        model_name_clean = model_name.replace('/', '_').replace('-',
                                                                '_').replace(
                                                                    '.', '_')
        cot_column_name = f"cot_{model_name_clean}"
        pred_column_name = f"pred_{model_name_clean}"
        exact_match_column_name = f"exact_match_{model_name_clean}"

        # Add full LLM responses (chain of thought)
        inference_df[cot_column_name] = responses

        # Parse answers from responses
        parsed_answers = [
            parse_answer_from_response(response) for response in responses
        ]
        inference_df[pred_column_name] = parsed_answers

        # Compute exact match scores if golden_answers are available
        if has_golden_answers:
            exact_matches = [
                compute_exact_match(pred_answer, golden_answers)
                for pred_answer, golden_answers in zip(
                    parsed_answers, inference_df['golden_answers'])
            ]
            inference_df[exact_match_column_name] = exact_matches
            print(f"✅ Successfully computed exact match scores")

        print(
            f"✅ Successfully added predictions to columns: {cot_column_name}, {pred_column_name}"
        )

    except Exception as e:
        print(f"❌ Error running inference with {model_name}: {str(e)}")
        # Add error columns with error messages
        model_name_clean = model_name.replace('/', '_').replace('-',
                                                                '_').replace(
                                                                    '.', '_')
        cot_column_name = f"cot_{model_name_clean}"
        pred_column_name = f"pred_{model_name_clean}"
        exact_match_column_name = f"exact_match_{model_name_clean}"
        inference_df[cot_column_name] = [f"ERROR: {str(e)}"
                                         ] * len(inference_df)
        inference_df[pred_column_name] = ["ERROR"] * len(inference_df)
        if has_golden_answers:
            inference_df[exact_match_column_name] = [False] * len(inference_df)

    # Choose judge model and backend
    if judge_backend == 'api':
        judge_model = 'openai/gpt-4o-mini'
        judge_inference_fn = lambda prompts: call_llm_batch(
            prompts=prompts,
            model_name=judge_model,
            use_cache=True,
            max_tokens=200,
            temperature=0.0,
            include_cost=True)[0]  # Just return responses, not costs
    else:
        judge_model = 'Qwen/Qwen2.5-7B-Instruct'
        judge_inference_fn = lambda prompts: run_vllm_inference(
            judge_model, prompts, max_tokens=200, temperature=0.0)

    inference_df = run_judge_evaluation_flexible(inference_df,
                                                 judge_inference_fn)

    # Display some results
    print("\n" + "=" * 80)
    print("INFERENCE RESULTS SUMMARY")
    print("=" * 80)
    print(f"Total questions processed: {len(inference_df)}")
    print(f"Columns in final DataFrame: {inference_df.columns.tolist()}")

    # Show first few examples
    print("\nFirst 3 examples:")
    for i in range(min(3, len(inference_df))):
        print(f"\n--- Example {i+1} ---")
        print(f"Question: {inference_df.iloc[i]['question']}")
        print(f"Ground Truth: {inference_df.iloc[i]['answer']}")
        if has_golden_answers:
            print(f"Golden Answers: {inference_df.iloc[i]['golden_answers']}")
        for col in inference_df.columns:
            if col.startswith('pred_'):
                print(f"Prediction ({col}): {inference_df.iloc[i][col]}")
            elif col.startswith('exact_match_'):
                print(f"Exact Match ({col}): {inference_df.iloc[i][col]}")
            elif col.startswith('cot_'):
                print(
                    f"Chain of Thought ({col}): {inference_df.iloc[i][col][:100]}..."
                )
            elif col.startswith('score_'):
                print(f"Judge Score ({col}): {inference_df.iloc[i][col]}")
            elif col.startswith('judge_'):
                print(
                    f"Judge Response ({col}): {inference_df.iloc[i][col][:200]}..."
                )
                if len(inference_df.iloc[i][col]) > 200:
                    print(
                        f"  [Response truncated - full response saved in CSV]")

    # Show evaluation summary
    print("\n" + "=" * 80)
    print("EVALUATION SUMMARY")
    print("=" * 80)

    for col in inference_df.columns:
        if col.startswith('score_'):
            model_name = col.replace('score_', '')
            scores = inference_df[col]

            # Calculate metrics
            valid_scores = [s for s in scores
                            if s >= 0]  # Exclude -1 (ERROR) scores
            if len(valid_scores) > 0:
                avg_score = sum(valid_scores) / len(valid_scores)
                perfect_matches = sum(1 for s in valid_scores if s == 1.0)
                partial_matches = sum(1 for s in valid_scores if s == 0.5)
                incorrect = sum(1 for s in valid_scores if s == 0.0)

                print(f"\nModel: {model_name}")
                print(
                    f"  Valid evaluations: {len(valid_scores)}/{len(scores)}")
                print(f"  Average score: {avg_score:.3f}")
                print(
                    f"  Perfect matches (1.0): {perfect_matches} ({perfect_matches/len(valid_scores)*100:.1f}%)"
                )
                print(
                    f"  Partial matches (0.5): {partial_matches} ({partial_matches/len(valid_scores)*100:.1f}%)"
                )
                print(
                    f"  Incorrect (0.0): {incorrect} ({incorrect/len(valid_scores)*100:.1f}%)"
                )
            else:
                print(f"\nModel: {model_name}")
                print(f"  No valid evaluations (all predictions were ERROR)")

    # Show exact match summary if available
    if has_golden_answers:
        print("\n" + "=" * 40)
        print("EXACT MATCH SUMMARY")
        print("=" * 40)

        for col in inference_df.columns:
            if col.startswith('exact_match_'):
                model_name = col.replace('exact_match_', '')
                exact_matches = inference_df[col]

                # Calculate exact match metrics
                total_valid = len(
                    [em for em in exact_matches if isinstance(em, bool)])
                if total_valid > 0:
                    exact_match_count = sum(exact_matches)
                    exact_match_rate = exact_match_count / total_valid * 100

                    print(f"\nModel: {model_name}")
                    print(
                        f"  Exact matches: {exact_match_count}/{total_valid} ({exact_match_rate:.1f}%)"
                    )
                else:
                    print(f"\nModel: {model_name}")
                    print(f"  No valid exact match evaluations")

    # Save results
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)

    # Create filename with model name included
    model_name_clean = model_name.replace('/',
                                          '_').replace('-',
                                                       '_').replace('.', '_')
    output_file = os.path.join(
        results_dir, f"bioasq_{model_name_clean}_inference_results.csv")
    inference_df.to_csv(output_file, index=False)
    print(f"\n💾 Results saved to: {output_file}")

    ipdb.set_trace()

    return inference_df


def interactive_llm_session(model_name: str = 'Qwen/Qwen2.5-3B-Instruct'):
    """
    Load an LLM instance and provide an interactive session for testing.
    
    Args:
        model_name: Model to load for interactive testing
    """
    print(f"Loading {model_name} for interactive session...")

    # Load the model
    llm = LLM(
        model=model_name,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.8,
        trust_remote_code=True,
    )

    # Set up sampling parameters
    sampling_params = SamplingParams(
        temperature=0.1,
        max_tokens=500,
        stop=None,
    )

    def generate(prompt: str, temp: float = 0.1, max_tok: int = 500):
        """Helper function to generate responses easily."""
        params = SamplingParams(temperature=temp,
                                max_tokens=max_tok,
                                stop=None)
        outputs = llm.generate([prompt], params)
        return outputs[0].outputs[0].text

    def generate_batch(prompts: List[str],
                       temp: float = 0.1,
                       max_tok: int = 500):
        """Helper function to generate batch responses easily."""
        params = SamplingParams(temperature=temp,
                                max_tokens=max_tok,
                                stop=None)
        outputs = llm.generate(prompts, params)
        return [output.outputs[0].text for output in outputs]

    print("✅ Model loaded successfully!")
    print("Available functions:")
    print("  - generate(prompt, temp=0.1, max_tok=500) - single prompt")
    print("  - generate_batch(prompts, temp=0.1, max_tok=500) - batch prompts")
    print("  - llm - the LLM instance")
    print("  - sampling_params - default sampling parameters")
    print("\nExample usage:")
    print("  response = generate('What is the capital of France?')")
    print("  responses = generate_batch(['Hello', 'How are you?'])")

    # Single prompt
    response = generate("What is the capital of France?")
    ipdb.set_trace()
    pass

    # With different temperature
    response = generate("Tell me a joke", temp=0.8)

    # Batch prompts
    responses = generate_batch(
        ["What is 2+2?", "Name a color", "What is the sky?"])

    # Use the raw LLM instance if you need more control
    outputs = llm.generate(["Custom prompt"], sampling_params)

    # Test your specific prompt template
    test_prompt = QUESTION_TEMPLATE_FACTOID.format(
        question="What causes diabetes?")
    response = generate(test_prompt)
    parsed = parse_answer_from_response(response)
    print(f"Parsed answer: {parsed}")

    # Set breakpoint for interactive use
    ipdb.set_trace()

    # Cleanup when done
    del llm
    torch.cuda.empty_cache()
    print("🧹 Model cleaned up and GPU memory freed")


if __name__ == "__main__":
    # Uncomment the line you want to run:

    # For interactive LLM session (testing/debugging)
    # interactive_llm_session()

    # For full evaluation pipeline
    num_questions = 1609
    dataset_name = "jmhb/bioasq_trainv0_n1609"
    results_df = main(num_questions=num_questions, dataset_name=dataset_name)
    ipdb.set_trace()
