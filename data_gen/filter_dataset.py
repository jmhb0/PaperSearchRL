"""
python -m ipdb data_gen/filter_dataset.py 
print(df.groupby(['rag_judge_score', 'direct_judge_score'])['rag_judge_score'].count() / len(df))
"""
import os
import sys
import pdb
import datasets
import pandas as pd
import numpy as np
import ipdb
from argparse import ArgumentParser

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from eval.run_inference import run_inference, InferenceConfig, _extract_model_name


def run_and_load_inference(method, dataset_name, model_id, results_dir):
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
    output_filename = f"{method}_{model_name}.csv"
    output_path = os.path.join(results_dir, output_filename)

    if os.path.exists(output_path):
        print("\n" + "*" * 80)
        print(f"Loading existing {method.upper()} results from: {output_path}")
        print("*" * 80 + "\n")
        return pd.read_csv(output_path)

    print(f"\nRunning {method.upper()} inference, this may take a while...\n")
    # run_inference will save the txt file and the csv with em_score and judge_score
    results_df = run_inference(config, output_path=output_path, run_judge=True)

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


def main():
    parser = ArgumentParser()
    parser.add_argument("--dataset_name",
                        type=str,
                        default="jmhb/PaperSearchRL_v1_n10000_test500",
                        help="Name of the huggingface dataset")
    parser.add_argument("--key",
                        type=str,
                        default="1",
                        help="Configuration key for naming the output dataset")
    parser.add_argument("--model_id",
                        type=str,
                        default="Qwen/Qwen2.5-7B-Instruct",
                        help="Model to use for inference")
    args = parser.parse_args()

    # Create new dataset name
    new_dataset_name = f"{args.dataset_name}_filterk{args.key}"
    print(f"Will generate new dataset named: {new_dataset_name}")

    # Create results directory
    results_dir_name = args.dataset_name.replace("/", "-")
    results_dir = f"results/filter_dataset_{results_dir_name}"
    os.makedirs(results_dir, exist_ok=True)
    print(f"Results will be saved in: {results_dir}")

    cot_results_df = run_and_load_inference('cot', args.dataset_name,
                                            args.model_id, results_dir)
    rag_results_df = run_and_load_inference('rag', args.dataset_name,
                                            args.model_id, results_dir)
    direct_results_df = run_and_load_inference('direct', args.dataset_name,
                                               args.model_id, results_dir)

    dataset = datasets.load_dataset(args.dataset_name, split="test")
    df = dataset.to_pandas()

    first_n = 10000
    if first_n > 0 and len(df) > first_n:
        df = df.head(first_n)

    # Safely add score columns
    if 'em_score' in cot_results_df.columns and 'judge_score' in cot_results_df.columns:
        df['cot_em_score'] = cot_results_df['em_score']
        df['cot_judge_score'] = cot_results_df['judge_score']
    else:
        print(
            "Warning: COT results loaded from cache are missing 'em_score' or 'judge_score'. You may need to delete the cache file and re-run."
        )

    if 'em_score' in rag_results_df.columns and 'judge_score' in rag_results_df.columns:
        df['rag_em_score'] = rag_results_df['em_score']
        df['rag_judge_score'] = rag_results_df['judge_score']
    else:
        print(
            "Warning: RAG results loaded from cache are missing 'em_score' or 'judge_score'. You may need to delete the cache file and re-run."
        )

    if 'em_score' in direct_results_df.columns and 'judge_score' in direct_results_df.columns:
        df['direct_em_score'] = direct_results_df['em_score']
        df['direct_judge_score'] = direct_results_df['judge_score']
    else:
        print(
            "Warning: Direct results loaded from cache are missing 'em_score' or 'judge_score'. You may need to delete the cache file and re-run."
        )

    ipdb.set_trace()
    df = apply_filtering(df, args.key)
    ipdb.set_trace()

    print("Added scores to the dataset. Entering debug mode.")
    ipdb.set_trace()
    pass


if __name__ == "__main__":
    main()
