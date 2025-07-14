"""
python -m ipdb data_gen/filter_dataset.py 
python -m ipdb data_gen/filter_dataset.py --dataset_name jmhb/PaperSearchRL_v4_gv2_n3000_test500_parav1pcnt50 --key 1
python -m ipdb data_gen/filter_dataset.py --dataset_name jmhb/PaperSearchRL_v4_gv2_n3000_test500
"""
import os
import sys
import pdb
import datasets
import pandas as pd
import numpy as np
import ipdb
import json
from argparse import ArgumentParser

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from eval.run_inference import run_inference, InferenceConfig, _extract_model_name
from data_gen.api import call_llm_batch
from verl.utils.reward_score.qa_em import compute_score_em


def fixed_llm_judge_batch(questions,
                          ground_truths,
                          predictions,
                          judge_model="openai/gpt-4o-mini"):
    """
    Fixed version of LLM judge batch function with corrected prompt and error handling.
    """
    # Create prompts for batch processing (fixed typo)
    judge_prompts = []
    for question, ground_truth, prediction in zip(questions, ground_truths,
                                                  predictions):
        judge_prompt = f"""Please evaluate the following answer to a question.

Question: {question}

Ground Truth Answer: {ground_truth}

Predicted Answer: {prediction}

Please provide:
1. A correctness score either 0 or 1. 
2. A brief explanation of your evaluation
3. Whether the prediction is "correct" or "incorrect"

The answer should be a single entity. 

- The answer is correct **only if it matches all key aspects, specificity, and detail of the ground truth, even if differently phrased (e.g., synonym or rewording that does not lose or broaden meaning).**
- If the prediction omits, generalizes, or adds information not present in the ground truth, it should be considered **incorrect**.
- **Even if the answer is a valid synonym, ensure it covers the same detail and scope.**
- For example, if the ground truth is "heart muscle," correct answers include "cardiac muscle" or "muscle of the heart," but NOT just "muscle" or "organ."
- **Any answer that is more vague, less specific, or encompasses a broader/narrower category than the ground truth should be marked as incorrect.**

Respond in JSON format:
{{
    "score": <score>,
    "explanation": "<explanation>",
    "judgment": "<correct/incorrect>"
}}"""
        judge_prompts.append(judge_prompt)

    try:
        # Fixed: Remove asyncio.run since call_llm_batch is already synchronous
        responses, _ = call_llm_batch(prompts=judge_prompts,
                                      model_name=judge_model,
                                      max_tokens=500,
                                      temperature=0.1,
                                      json_mode=True,
                                      max_concurrent=10)

        # Parse responses
        judgments = []
        for i, response in enumerate(responses):
            try:
                if response.startswith("[ERROR:"):
                    judgments.append({
                        "score": 0,
                        "explanation": f"Judge failed: {response}",
                        "judgment": "error"
                    })
                else:
                    # Try to parse JSON response
                    judgment = json.loads(response)
                    # Ensure required keys exist
                    if not all(
                            key in judgment
                            for key in ['score', 'explanation', 'judgment']):
                        judgments.append({
                            "score": 0,
                            "explanation":
                            f"Missing required keys in response: {response[:100]}...",
                            "judgment": "error"
                        })
                    else:
                        judgments.append(judgment)
            except json.JSONDecodeError as e:
                print(f"Failed to parse judge response {i}: {e}")
                print(f"Response was: {response[:200]}...")
                judgments.append({
                    "score": 0,
                    "explanation": f"JSON parse failed: {response[:100]}...",
                    "judgment": "error"
                })
            except Exception as e:
                print(f"Unexpected error parsing response {i}: {e}")
                judgments.append({
                    "score": 0,
                    "explanation": f"Unexpected error: {str(e)}",
                    "judgment": "error"
                })
        # ipdb.set_trace()

        return judgments

    except Exception as e:
        print(f"LLM judge batch failed: {e}")
        # Return error judgments for all items
        return [{
            "score": 0,
            "explanation": f"Batch judge failed: {str(e)}",
            "judgment": "error"
        } for _ in questions]


def run_and_load_inference(method,
                           dataset_name,
                           model_id,
                           results_dir,
                           split="test",
                           overwrite_cache=False):
    """
    Runs inference using the specified method, or loads existing results.
    This function leverages `run_inference` from `eval.run_inference` to perform
    the actual inference, evaluation (judge and EM), and saving of results.
    """
    config = InferenceConfig(method=method,
                             model_id=model_id,
                             dataset_id=dataset_name,
                             batch_size=1,
                             first_n=10000)

    model_name = _extract_model_name(config.model_id)
    output_filename = f"{method}_{model_name}_{split}.csv"
    output_path = os.path.join(results_dir, output_filename)

    if os.path.exists(output_path) and not overwrite_cache:
        print("\n" + "*" * 80)
        print(
            f"Loading existing {method.upper()} {split} results from: {output_path}"
        )
        print("*" * 80 + "\n")
        return pd.read_csv(output_path)

    if overwrite_cache and os.path.exists(output_path):
        print(f"\nOverwriting cached results for {method.upper()} {split}...")

    print(
        f"\nRunning {method.upper()} inference on {split} split, this may take a while...\n"
    )

    # Use the existing run_inference function but with custom handling for splits
    from eval.run_inference import InferenceEngine
    from datasets import load_dataset
    from tqdm import tqdm

    print(f"Starting inference with method: {config.method}")
    print(f"Dataset: {config.dataset_id} ({split} split)")
    print(f"Model: {config.model_id}")
    print(f"Batch size: {config.batch_size}")
    print(f"First N examples: {config.first_n}")

    # Load dataset with specified split
    try:
        dataset = load_dataset(config.dataset_id, split=split)
        df = dataset.to_pandas()
        print(f"Loaded {len(df)} examples from {split} split")

        # Limit to first N examples
        if config.first_n > 0 and len(df) > config.first_n:
            df = df.head(config.first_n)
            print(f"Limited to first {config.first_n} examples")

    except Exception as e:
        print(f"Failed to load dataset: {e}")
        return pd.DataFrame()

    # Initialize inference engine
    engine = InferenceEngine(config)

    # Extract questions - dataset has 'question' and 'answer' columns, no context
    questions = []
    for _, row in df.iterrows():
        question = row.get('question', '')
        questions.append(question)

    # For most methods, contexts are empty since dataset doesn't have context
    # For RAG, contexts will be retrieved dynamically during inference
    contexts = [""] * len(questions)

    # Run inference based on method
    if config.method == "direct":
        print("Running batch direct inference...")
        all_results = engine.direct_inference_batch(questions, contexts)

    elif config.method == "cot":
        print("Running batch CoT inference...")
        all_results = engine.cot_inference_batch(questions, contexts)

    elif config.method == "rag":
        print("Running batch RAG inference...")
        all_results = engine.rag_inference_batch(questions, contexts)

    elif config.method in ["searchr1", "papersearchr1"]:
        print(f"Running {config.method} inference (sequential)...")
        # SearchR1/PaperSearchR1 methods process individually
        inference_fn = (engine.searchr1_inference if config.method
                        == "searchr1" else engine.papersearchr1_inference)

        all_results = []
        for question, context in tqdm(zip(questions, contexts),
                                      total=len(questions),
                                      desc="Processing"):
            try:
                response, prediction = inference_fn(question, context)
                all_results.append((response, prediction))
            except Exception as e:
                print(f"Error processing question: {e}")
                all_results.append((f"[ERROR: {str(e)}]", "[ERROR]"))

    else:
        raise ValueError(f"Unknown method: {config.method}")

    # Create results DataFrame
    results_data = []
    for i, (response, prediction) in enumerate(all_results):
        result_row = df.iloc[i].to_dict()
        result_row['model_output'] = response
        result_row['prediction'] = prediction
        result_row['method'] = config.method
        result_row['model_id'] = config.model_id
        results_data.append(result_row)

    results_df = pd.DataFrame(results_data)

    print(f"Inference complete! Processed {len(results_df)} examples")

    # Run LLM judge with fixed function
    print("Running LLM judge evaluation...")

    # Prepare data for batch processing
    questions = []
    ground_truths = []
    predictions = []
    golden_answers = []

    for idx, row in results_df.iterrows():
        questions.append(row['question'])
        ground_truths.append(row['answer'])
        predictions.append(row['prediction'])
        # Use golden_answers column if available, otherwise fallback to answer
        if 'golden_answers' in row:
            golden_answers.append(row['golden_answers'])
        else:
            golden_answers.append(row['answer'])

    # Run batch judgment with fixed function
    judgments = fixed_llm_judge_batch(questions,
                                      ground_truths,
                                      predictions,
                                      judge_model="openai/gpt-4o-mini")

    # Add judgment results to DataFrame
    for key in ['score', 'explanation', 'judgment']:
        results_df[f'judge_{key}'] = [j.get(key, None) for j in judgments]

    # Compute exact match scores using golden_answers
    print("Computing exact match scores...")
    em_scores = []
    for pred, gold in zip(predictions, golden_answers):
        clean_gold_list = {"target": list(gold)}
        # em_score func expects the answer to be in a very long answer trace with these answer tags:
        pred_str = f"<answer></answer><answer>{pred}</answer>"
        em_score = compute_score_em(pred_str, clean_gold_list)
        em_scores.append(em_score)

    results_df['em_score'] = em_scores

    # Print some debugging info
    judge_scores = results_df['judge_score'].tolist()
    em_scores_list = results_df['em_score'].tolist()
    print(f"Judge scores: {np.mean(judge_scores)}")
    print(f"EM scores: {np.mean(em_scores_list)}")

    # Save results
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    results_df.to_csv(output_path, index=False)
    print(f"Results saved to: {output_path}")

    return results_df


def apply_filtering(df, key):
    """
    Applies filtering to the DataFrame based on rag_judge_score and direct_judge_score.
    The filtering rules are defined by the key.

    Args:
        df (pd.DataFrame): The DataFrame to filter.
        key (str): The filtering configuration key.

    Returns:
        pd.DataFrame: The filtered DataFrame.
    """
    # Ensure scores are present
    if 'rag_judge_score' not in df.columns or 'direct_judge_score' not in df.columns:
        print(
            "Warning: 'rag_judge_score' or 'direct_judge_score' not in DataFrame. "
            "Skipping filtering.")
        return df

    # Define filter percentages for each key. (s_rag, s_dir) -> percentage to drop
    # s_rag = df['rag_judge_score'], s_dir = df['direct_judge_score']
    filter_configs = {
        "0": {
            (0, 0): 0.0,
            (0, 1): 0.0,
            (1, 0): 0.0,
            (1, 1): 0.0
        },
        "1": {
            (0, 0): 0.0,
            (0, 1): 0.5,
            (1, 0): 0.8,
            (1, 1): 0.8
        },
    }

    if key not in filter_configs:
        print(f"Warning: Key '{key}' not found in filter_configs. "
              "No filtering will be applied.")
        return df

    config = filter_configs[key]
    print(f"\nApplying filtering for key='{key}' with config: {config}")

    s_rag = df['rag_judge_score']
    s_dir = df['direct_judge_score']

    initial_count = len(df)
    indices_to_drop = []

    for group, drop_pcnt in config.items():
        if drop_pcnt > 0:
            s_rag_val, s_dir_val = group
            group_indices = df[(s_rag == s_rag_val)
                               & (s_dir == s_dir_val)].index

            num_to_drop = int(len(group_indices) * drop_pcnt)
            if num_to_drop > 0:
                # Use random.choice for sampling without replacement
                drop_indices = np.random.choice(group_indices,
                                                size=num_to_drop,
                                                replace=False)
                indices_to_drop.extend(drop_indices)

    filtered_df = df.drop(indices_to_drop)

    print(f"Original size: {initial_count}")
    print(f"Filtered size: {len(filtered_df)}")
    print(f"Dropped: {len(indices_to_drop)} samples.\n")

    return filtered_df


def process_split(dataset_name, model_id, results_dir, key, split,
                  overwrite_cache):
    """Process a single split (train or test)."""
    print(f"\n{'='*60}")
    print(f"Processing {split.upper()} split")
    print('=' * 60)

    # Run inference for both methods
    rag_results_df = run_and_load_inference('rag', dataset_name, model_id,
                                            results_dir, split,
                                            overwrite_cache)
    direct_results_df = run_and_load_inference('direct', dataset_name,
                                               model_id, results_dir, split,
                                               overwrite_cache)

    # Load original dataset split
    dataset = datasets.load_dataset(dataset_name, split=split)
    df = dataset.to_pandas()

    first_n = 10000
    if first_n > 0 and len(df) > first_n:
        df = df.head(first_n)

    # Safely add score columns
    if 'em_score' in rag_results_df.columns and 'judge_score' in rag_results_df.columns:
        df['rag_em_score'] = rag_results_df['em_score']
        df['rag_judge_score'] = rag_results_df['judge_score']
    else:
        print(
            f"Warning: RAG results for {split} split loaded from cache are missing 'em_score' or 'judge_score'. You may need to delete the cache file and re-run."
        )

    if 'em_score' in direct_results_df.columns and 'judge_score' in direct_results_df.columns:
        df['direct_em_score'] = direct_results_df['em_score']
        df['direct_judge_score'] = direct_results_df['judge_score']
    else:
        print(
            f"Warning: Direct results for {split} split loaded from cache are missing 'em_score' or 'judge_score'. You may need to delete the cache file and re-run."
        )

    # Apply filtering
    df = apply_filtering(df, key)

    return df


def main():
    parser = ArgumentParser()
    parser.add_argument(
        "--dataset_name",
        type=str,
        # default="jmhb/PaperSearchRL_v1_n10000_test500",
        # default="jmhb/PaperSearchRL_v4_gv2_n3000_test500",
        default="jmhb/PaperSearchRL_v5_gv3_n3000_test300",
        # default="jmhb/PaperSearchRL_v5_gv3_n3000_test300_parav1pcnt50",
        help="Name of the huggingface dataset")
    parser.add_argument("--key",
                        type=str,
                        default="1",
                        help="Configuration key for naming the output dataset")
    parser.add_argument("--model_id",
                        type=str,
                        default="Qwen/Qwen2.5-7B-Instruct",
                        help="Model to use for inference")
    parser.add_argument(
        "--overwrite_cache",
        action="store_true",
        help="Overwrite cached inference results and re-run inference")
    args = parser.parse_args()

    # Create new dataset name
    new_dataset_name = f"{args.dataset_name}_filterk{args.key}"
    print(f"Will generate new dataset named: {new_dataset_name}")

    # Create results directory
    results_dir_name = args.dataset_name.replace("/", "-")
    results_dir = f"results/filter_dataset_{results_dir_name}"
    os.makedirs(results_dir, exist_ok=True)
    print(f"Results will be saved in: {results_dir}")

    # Process both train and test splits
    splits_data = {}
    for split in ["train", "test"]:
        try:
            filtered_df = process_split(args.dataset_name, args.model_id,
                                        results_dir, args.key, split,
                                        args.overwrite_cache)
            splits_data[split] = datasets.Dataset.from_pandas(filtered_df)
            print(
                f"Successfully processed {split} split: {len(filtered_df)} samples"
            )
        except Exception as e:
            print(f"Error processing {split} split: {e}")
            print(f"Skipping {split} split...")
            continue

    if not splits_data:
        print("Error: No splits were successfully processed. Exiting.")
        return

    # Create DatasetDict with available splits
    print(f"\nCreating DatasetDict with splits: {list(splits_data.keys())}")
    # ipdb.set_trace()
    filtered_dataset_dict = datasets.DatasetDict(splits_data)

    # Push to hub
    print(
        f"\nPushing filtered dataset to Hugging Face Hub: {new_dataset_name}")
    # Note: You may need to log in to Hugging Face CLI first.
    # Run `huggingface-cli login` and provide a token with write permissions.
    filtered_dataset_dict.push_to_hub(new_dataset_name)
    print("\nDataset successfully pushed to the Hub.")


if __name__ == "__main__":
    main()
