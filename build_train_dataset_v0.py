#!/usr/bin/env python3
"""
python -m ipdb build_train_dataset_v0.py --n_samples 10000 --n_test 100
Script to build training dataset v0 for biological question answering.
Loads BioASQ-taskb dataset, filters for factoid questions, and generates
synonym lists using GPT-4o mini.
"""

import argparse
import ast
import json
import pandas as pd
from datasets import load_dataset, Dataset, DatasetDict
from huggingface_hub import HfApi
import os
from typing import List, Dict, Any, Union
from data_gen.api import call_llm_batch
import ipdb
import numpy as np


def parse_answer_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    Parse the 'answer' column from string representation to actual lists.
    
    Args:
        df: DataFrame with 'answer' column containing string representations of lists
        
    Returns:
        DataFrame with 'answer' column converted to actual lists
    """

    def safe_literal_eval(x):
        try:
            return ast.literal_eval(x)
        except (ValueError, SyntaxError):
            # If literal_eval fails, try to handle as a simple string
            return [x] if isinstance(x, str) else x

    df['answer'] = df['answer'].apply(safe_literal_eval)
    return df


def generate_synonyms_prompt(answers: List[str]) -> str:
    """
    Generate the prompt for synonym generation.
    
    Args:
        answers: List of answer strings
        
    Returns:
        Formatted prompt string
    """
    answers_str = str(answers)

    prompt = f"""I am generating a dataset for biological question answering.

They are questions that have simple factoid answers - meaning the answer is a single entity.

Below I'll give the target answer or answers.

However the same entity might have synonyms that are equivalent.

Your task is to return a python list of synonyms for each unique entity mentioned in the answers.

For example, if the target answer was "c-Jun NH2-terminal kinase", then you should return a list of synonyms ["JNK", "c-Jun N-terminal kinase", "c-Jun amino-terminal kinase", "c-Jun NH2-terminal kinase"]

Return a python list of all the common synonyms. Return as many as you can think of. Include the original answer(s) in the list as well.

You must respond with valid JSON containing only the list of synonyms.

ANSWERS
{answers_str}"""

    return prompt


def process_gpt_response(response: str) -> List[str]:
    """
    Process GPT response to extract the python list of synonyms.
    
    Args:
        response: Raw response from GPT
        
    Returns:
        List of synonym strings
    """
    try:
        response = response.strip()

        # First try to parse as JSON (since we're using json_mode)
        try:
            parsed = json.loads(response)
            if isinstance(parsed, list):
                return parsed
            elif isinstance(parsed, dict) and 'synonyms' in parsed:
                # Handle cases where GPT wraps the list in an object
                return parsed['synonyms']
            elif isinstance(parsed, dict):
                # Try to find any list value in the JSON object
                for value in parsed.values():
                    if isinstance(value, list):
                        return value
        except json.JSONDecodeError:
            pass

        # Fallback: try to parse as Python literal (ast.literal_eval)
        try:
            if response.startswith('[') and response.endswith(']'):
                return ast.literal_eval(response)

            # Try to find a list within the response
            lines = response.split('\n')
            for line in lines:
                line = line.strip()
                if line.startswith('[') and line.endswith(']'):
                    return ast.literal_eval(line)
        except (ValueError, SyntaxError):
            pass

        # Final fallback: return the response as a single-item list
        print(f"Warning: Could not parse response as list: {response}")
        return [response]

    except Exception as e:
        print(f"Error processing GPT response: {e}")
        print(f"Response was: {response}")
        return ["[ERROR_PARSING_RESPONSE]"]


def generate_golden_answers(df: pd.DataFrame,
                            batch_size: int = 50) -> pd.DataFrame:
    """
    Generate golden_answers column using GPT-4o mini in batch mode.
    
    Args:
        df: DataFrame with 'answer' column
        batch_size: Batch size for API calls
        
    Returns:
        DataFrame with new 'golden_answers' column.
    """
    print(f"Generating synonyms for {len(df)} samples...")

    # Prepare prompts for all rows
    prompts = []
    for _, row in df.iterrows():
        prompt = generate_synonyms_prompt(row['answer'])
        prompts.append(prompt)

    print(f"Making batch API calls with {len(prompts)} prompts...")

    # Call GPT-4o mini in batch mode
    responses, costs = call_llm_batch(
        prompts=prompts,
        model_name="openai/gpt-4o-mini",
        max_tokens=500,
        temperature=0.1,  # Low temperature for more consistent outputs
        max_concurrent=10,  # Conservative concurrent limit
        json_mode=True  # Enable JSON mode for structured output
    )

    print("Processing responses...")

    # Process responses to extract synonym lists
    golden_answers = []
    for i, response in enumerate(responses):
        if response.startswith('[ERROR:'):
            print(f"API error for row {i}: {response}")
            # Fallback to original answers
            golden_answers.append(df.iloc[i]['answer'])
        else:
            synonyms = process_gpt_response(response)
            golden_answers.append(synonyms)

    # Add the golden_answers column right after the answer column
    df = df.copy()
    answer_col_idx = df.columns.get_loc('answer')
    df.insert(answer_col_idx + 1, 'golden_answers', golden_answers)

    return df


def upload_to_huggingface(data: Union[pd.DataFrame, Dict[str, pd.DataFrame]],
                          dataset_name: str) -> None:
    """
    Upload the processed dataset to HuggingFace as a private dataset.
    
    Args:
        data: Either a DataFrame for single split or dict of DataFrames for multiple splits
        dataset_name: Name for the dataset on HuggingFace
    """
    print(f"Uploading dataset to HuggingFace as '{dataset_name}'...")

    if isinstance(data, dict):
        # Multiple splits
        dataset_dict = {}
        for split_name, df in data.items():
            dataset_dict[split_name] = Dataset.from_pandas(df)
        dataset = DatasetDict(dataset_dict)
    else:
        # Single DataFrame
        dataset = Dataset.from_pandas(data)

    # Upload to HuggingFace Hub
    dataset.push_to_hub(
        dataset_name,
        private=False,
        token=os.getenv("HF_TOKEN")  # Make sure HF_TOKEN is set
    )
    print(
        f"✅ Successfully uploaded dataset to: https://huggingface.co/datasets/{dataset_name}"
    )


def main():
    """Main function to process the dataset."""
    parser = argparse.ArgumentParser(
        description="Build training dataset v0 for BioASQ")
    parser.add_argument("--n_samples",
                        type=int,
                        required=True,
                        default=100,
                        help="Number of samples to process")
    parser.add_argument("--batch_size",
                        type=int,
                        default=50,
                        help="Batch size for API calls (default: 50)")
    parser.add_argument(
        "--n_test",
        type=int,
        default=None,
        help="Number of samples to use as test set (default: None)")

    args = parser.parse_args()

    # Set random seed for reproducibility
    np.random.seed(0)

    print(f"Building training dataset with {args.n_samples} samples...")
    if args.n_test is not None:
        print(
            f"Will split into {args.n_samples - args.n_test} train and {args.n_test} test samples"
        )

    # Load dataset from HuggingFace
    print("Loading dataset from HuggingFace...")
    dataset = load_dataset("jmhb/BioASQ-taskb")

    # Convert to DataFrame
    df = dataset['train'].to_pandas()

    # Drop the unwanted index column if it exists
    if '__index_level_0__' in df.columns:
        df = df.drop(columns=['__index_level_0__'])

    print(f"Loaded {len(df)} total samples")

    # Filter for factoid questions
    print("Filtering for factoid questions...")
    df_factoid = df[df['type'] == 'factoid'].copy()
    print(f"Found {len(df_factoid)} factoid questions")

    if len(df_factoid) < args.n_samples:
        print(f"Warning: Only {len(df_factoid)} factoid questions available, "
              f"but {args.n_samples} requested")
        args.n_samples = len(df_factoid)

    # Validate n_test if provided
    if args.n_test is not None:
        if args.n_test >= args.n_samples:
            raise ValueError(
                f"n_test ({args.n_test}) must be less than n_samples ({args.n_samples})"
            )

    # Parse answer column
    print("Parsing answer column...")
    df_factoid = parse_answer_column(df_factoid)

    # Randomly sample n_samples from factoid questions
    print(
        f"Randomly sampling {args.n_samples} from {len(df_factoid)} factoid questions..."
    )
    df_subset = df_factoid.sample(n=args.n_samples, random_state=0).copy()
    print(f"Processing {len(df_subset)} samples")

    # Generate golden_answers using GPT-4o mini
    df_processed = generate_golden_answers(df_subset,
                                           batch_size=args.batch_size)

    # put the 'answer' column back to a string type bc the types are not consistenbt and it breaks dataset conversion
    df_processed['answer'] = [str(a) for a in df_processed['answer']]

    # Split into train/test if n_test is provided
    if args.n_test is not None:
        print(f"Splitting dataset into train/test sets...")

        # Reset index to ensure we have consecutive indices for sampling
        df_processed = df_processed.reset_index(drop=True)

        # Randomly sample test indices
        test_indices = np.random.choice(len(df_processed),
                                        size=args.n_test,
                                        replace=False)

        # Create boolean mask for test samples
        test_mask = np.zeros(len(df_processed), dtype=bool)
        test_mask[test_indices] = True

        df_test = df_processed[test_mask].copy()
        df_train = df_processed[~test_mask].copy()

        print(f"Train set: {len(df_train)} samples")
        print(f"Test set: {len(df_test)} samples")

        # Validate that we have the expected number of samples
        assert len(
            df_test
        ) == args.n_test, f"Expected {args.n_test} test samples, got {len(df_test)}"
        assert len(
            df_train
        ) == args.n_samples - args.n_test, f"Expected {args.n_samples - args.n_test} train samples, got {len(df_train)}"

        # Create dataset name and upload with splits
        dataset_name = f"jmhb/bioasq_trainv0_n{args.n_samples}_test{args.n_test}"

        # Upload as single dataset with train/test splits
        upload_to_huggingface({
            'train': df_train,
            'test': df_test
        }, dataset_name)

        print("✅ Dataset processing complete!")
        print(f"Dataset uploaded as: {dataset_name}")
        print(f"  - Train split: {len(df_train)} samples")
        print(f"  - Test split: {len(df_test)} samples")

        # Print sample of results for both sets
        print("\nSample train results:")
        for i in range(min(2, len(df_train))):
            row = df_train.iloc[i]
            print(f"\nTrain Sample {i+1}:")
            print(f"Question: {row['question'][:100]}...")
            print(f"Original answers: {row['answer']}")
            print(f"Golden answers: {row['golden_answers']}")

        print("\nSample test results:")
        for i in range(min(2, len(df_test))):
            row = df_test.iloc[i]
            print(f"\nTest Sample {i+1}:")
            print(f"Question: {row['question'][:100]}...")
            print(f"Original answers: {row['answer']}")
            print(f"Golden answers: {row['golden_answers']}")
    else:
        # Original behavior - single dataset
        dataset_name = f"jmhb/bioasq_trainv0_n{args.n_samples}"

        # Upload to HuggingFace
        upload_to_huggingface(df_processed, dataset_name)

        print("✅ Dataset processing complete!")
        print(f"Dataset uploaded as: {dataset_name}")

        # Print sample of results
        print("\nSample results:")
        for i in range(min(3, len(df_processed))):
            row = df_processed.iloc[i]
            print(f"\nSample {i+1}:")
            print(f"Question: {row['question'][:100]}...")
            print(f"Original answers: {row['answer']}")
            print(f"Golden answers: {row['golden_answers']}")


if __name__ == "__main__":
    main()
