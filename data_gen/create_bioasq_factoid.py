"""
Script to download jmhb/bioasq_trainv0_n1609 dataset and create jmhb/bioasq_factoid
with all samples in the test set.

This script:
1. Downloads the HuggingFace dataset jmhb/bioasq_trainv0_n1609
2. Combines all splits into a single dataset
3. Creates a new dataset jmhb/bioasq_factoid with all samples in the test set
4. Pushes the new dataset to HuggingFace Hub
"""

import os
from datasets import load_dataset, Dataset, DatasetDict
from typing import List, Dict, Any


def create_bioasq_factoid_dataset():
    """
    Download the bioasq_trainv0_n1609 dataset and create bioasq_factoid 
    with all samples in the test set.
    """
    print("Downloading dataset jmhb/bioasq_trainv0_n1609...")

    # Load the original dataset
    try:
        original_dataset = load_dataset("jmhb/bioasq_trainv0_n1609")
        print(
            f"Successfully loaded dataset. Splits: {list(original_dataset.keys())}"
        )

        # Print info about each split
        for split_name, split_data in original_dataset.items():
            print(f"  {split_name}: {len(split_data)} samples")
            if len(split_data) > 0:
                print(f"    Columns: {list(split_data.column_names)}")

    except Exception as e:
        print(f"Error loading dataset: {e}")
        return None

    # Combine all splits into a single list
    all_samples = []

    for split_name, split_data in original_dataset.items():
        print(f"Adding {len(split_data)} samples from {split_name} split...")
        # Convert to list of dictionaries
        split_samples = split_data.to_list()
        all_samples.extend(split_samples)

    print(f"Total combined samples: {len(all_samples)}")

    if not all_samples:
        print("No samples found in the dataset!")
        return None

    # Create new dataset with all samples in test set
    print("Creating new dataset with all samples in test set...")
    test_dataset = Dataset.from_list(all_samples)

    # Create DatasetDict with only test split
    new_dataset = DatasetDict({'test': test_dataset})

    print(f"New dataset created:")
    print(f"  test: {len(test_dataset)} samples")
    print(f"  Columns: {list(test_dataset.column_names)}")

    # Push to HuggingFace Hub
    new_hub_name = "jmhb/bioasq_factoid"
    print(f"Pushing dataset to HuggingFace Hub as: {new_hub_name}")

    try:
        new_dataset.push_to_hub(new_hub_name)
        print(f"Successfully pushed dataset to {new_hub_name}")

        # Print summary
        print("\nSummary:")
        print(f"  Source dataset: jmhb/bioasq_trainv0_n1609")
        print(f"  New dataset: {new_hub_name}")
        print(f"  Total samples moved to test set: {len(all_samples)}")

        return new_dataset

    except Exception as e:
        print(f"Error pushing dataset to hub: {e}")
        return None


if __name__ == "__main__":
    create_bioasq_factoid_dataset()
