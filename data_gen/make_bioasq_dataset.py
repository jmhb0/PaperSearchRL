import glob
import json
import os
import re
import ipdb
import pandas as pd
import numpy as np
from datasets import load_dataset, Dataset, DatasetDict
from tqdm import tqdm

def get_data(data_type='train'):
    # Find all matching folders and filter to only include directories
    if data_type == 'train':
        bioasq_folders = [f for f in glob.glob("data/BioASQ-training*b*") if os.path.isdir(f)]
        pattern = r'training(\d+)b'
    else:  # test
        bioasq_folders = [f for f in glob.glob("data/Task*BGolden*") if os.path.isdir(f)]
        pattern = r'Task(\d+)BGolden'

    # Extract numbers and load JSONs
    folder_data = {}
    folder_names = {}  # Add this to store folder names
    for folder in bioasq_folders:
        # Extract number using regex
        match = re.search(pattern, folder)
        if match:
            number = int(match.group(1))
            
            # Check for JSON files in folder
            json_files = glob.glob(os.path.join(folder, "*.json"))
            if data_type == 'train':
                assert len(json_files) == 1, f"Expected exactly one JSON file in {folder}, found {len(json_files)}"
                json_path = json_files[0]
            else:  # test
                # For test, we'll process all JSON files and combine their questions
                if not json_files:
                    print(f"Warning: No JSON files found in {folder}")
                    continue
            
            # Load the JSON file(s)
            if data_type == 'train':
                with open(json_path, 'r') as f:
                    data = json.load(f)
                folder_data[number] = data
            else:  # test
                # Combine questions from all JSON files
                all_questions = []
                for json_path in json_files:
                    with open(json_path, 'r') as f:
                        data = json.load(f)
                        all_questions.extend(data['questions'])
                folder_data[number] = {'questions': all_questions}
            
            folder_names[number] = os.path.basename(folder)  # Store folder name with its number

    # get all key names to make as dataframe columns
    cols = set()
    for k in folder_data.keys():
        questions = folder_data[k]['questions']
        for q in questions:
            cols_ = q.keys()
            cols = cols.union(cols_)

    # Create DataFrame with all columns
    df = pd.DataFrame(columns=['asq_challenge', 'folder_name'] + list(cols))

    # Populate DataFrame with question data
    for k in tqdm(folder_data.keys(), desc=f"Processing BioASQ {data_type} challenges"):
        questions = folder_data[k]['questions']
        folder_name = folder_names[k]  # Get folder name from our mapping
        for q in questions:
            # Create a row with the challenge number and all question data
            row = {'asq_challenge': k, 'folder_name': folder_name}
            row.update(q)
            df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)

    # Rename 'body' column to 'question'
    df = df.rename(columns={'body': 'question', 'exact_answer': 'answer'})

    # Define the desired column order
    primary_cols = ['type', 'question', 'answer', 'ideal_answer', 'documents', 'snippets']
    # Get remaining columns (excluding primary_cols, asq_challenge, and folder_name)
    remaining_cols = [col for col in df.columns if col not in primary_cols + ['asq_challenge', 'folder_name']]
    # Sort remaining columns to put _body and _type at the end
    remaining_cols.sort(key=lambda x: (not x.endswith('_body'), not x.endswith('_type')))

    # Reorder columns
    df = df[primary_cols + ['asq_challenge', 'folder_name'] + remaining_cols]

    # Drop columns ending with _body or _type
    cols_to_drop = [col for col in df.columns if col.endswith('_body') or col.endswith('_type') or col.endswith('statements')]
    df = df.drop(columns=cols_to_drop)

    for col in ['answer', 'ideal_answer']:
        df[col] = df[col].apply(lambda x: "" if (isinstance(x, float) and pd.isna(x)) else str(x))
    
    return df

# Process both training and test sets
train_df = get_data('train')
# train_df['split'] = 'train'
test_df = get_data('test')
# test_df['split'] = 'test'
df = pd.concat([train_df, test_df])
print("Dataset size before question-level dedeupe: ", len(df)) # returns 40747
df = df.drop_duplicates(subset=['question'], keep='first')
print("Dataset size after question-level dedeupe: ", len(df)) # returns 5404 

# Create datasets from DataFrames
dataset = Dataset.from_pandas(df)
hf_token = os.environ["HF_TOKEN"]
dataset.push_to_hub("jmhb/BioASQ-taskb", token=hf_token)

ipdb.set_trace()
pass 
