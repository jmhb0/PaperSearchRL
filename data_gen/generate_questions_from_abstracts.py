"""
Script to generate question-answer datasets from PubMed abstracts using GPT.

PREREQUISITES:
    Before running this script, you need to create the PubMed abstracts database:
    1. Run: python data_gen/allMesh_to_parquet.py
    2. This will create the required file: data/allMeSH_2022.parquet

Usage:
    python -m ipdb data_gen/generate_questions_from_abstracts.py

This script:
1. Loads PubMed abstracts from the parquet database
2. Samples n_samples abstracts randomly
3. Processes them in batches using GPT-4 to generate Q&A pairs
4. Generates golden_answers (synonyms) for each answer
5. Creates train/test split and pushes to HuggingFace Hub
"""

import os
import pandas as pd
import numpy as np
import csv
import ipdb
from typing import List, Dict, Any
import random
import json
from tqdm import tqdm
import re
from datasets import Dataset, DatasetDict

from data_gen.allMesh_to_parquet import return_indexer
from data_gen.api import call_llm, call_llm_batch

# Template definitions
TEMPLATE_1 = """
BACKGROUND
You are a domain-expert biomedical NLP assistant.
You are helping me to create an open-domain QA dataset. 
The downstream task will read a query and require an agent to search over Pubmed abstracts

--------
YOUR TASK 
I will provide you with title and abstract of a Pubmed article. 
Your task is to create a new question-answer pair. 

--------
TYPES OF QUESTIONS
The questions should be 'factoid based'. 
The answer should be a simple entity. 
It should not be ambiguous.
Don't be pretentious. 

--------
IMPORTANT NOTES
The question-answer pair will be used to evaluation question-answering systems with retrieval. Ths means the target system does not know which paper the question was sourced from. So an inappropriate question would be "What technology is used in this study to ...".
If the question contains acronyms that are not well known, then explain the acronym.

--------
EXAMPLE CATEGORIES 
Below are sample categories with sample questions. 

Category: 1 - Genetic inheritance & disease-linked mutations
question: What gene is mutated in Sickle Cell Anemia?
answer: HBB
question: Which ultraconserved element is associated with Embryonic Stem Cells (ESC) self-renewal?
answer: T-UCstem1
question: Is Huntington's disease caused by a dominate or recessive gene?
answer: dominant

Category: 2 - Therapeutics, indications & clinical evidence
question: What is the most effective drug for oxaliplatin-induced neuropathy?
answer: Duloxetine
question: Which cancer is the BCG vaccine used for?
answer: Non-muscle Invasive Bladder Cancer
question: How many injections of CLS-TA did the patients participating in the PEACHTREE trial receive?
answer: two

Category: 3 - Protein function, localization & signalling/enzymatic interactions
question: Which histone mark distinguishes active from inactive enhancers?
answer: H3K27ac
question: Which component of the Influenza A Virus affects mRNA transcription termination?
answer: NS1
question: Which is the main calcium binding protein of the sarcoplasmic reticulum?
answer: Calsequestrin

Category: 4 - Experimental & computational methods, resources & acronyms
question: Which algorithm has been proposed for efficient storage of WGS variant calls?
answer: SeqArray
question: What is an acceptable sequence coverage(depth) required for human whole-exome sequencing?
answer: 30x-60x

Category: 5 - Disease causation & pathogens
question: Which is the most common disease attributed to malfunction or absence of primary cilia?
answer: ['Polycystic kidney disease', 'PKD']
question: What organism causes scarlet fever also known as scarletina?
answer: ['Group A Streptococcus', 'Streptococcus pyogenes']
question: The pathogen Fusarium graminearum affects what type of plant species?
answer: cereal crops

Category: 6 - Biomarkers & diagnostic tests
question: Salivary Cortisol is a biomarker for what disease/syndrome/condition?
answer: stress
question: What is the gold standard for a diagnosis of narcolepsy?
answer: ['Sleep study', 'overnight polysomnography']

Category: 7 - Bioinformatics databases & curated resources
question: Which R/bioconductor package has been developed to aid in epigenomic analysis?
answer: DeepBlueR
question: Which database associates human noncoding SNPs with their three-dimensional interacting genes?
answer: 3DSNP
question: What is the RESID database?
question: Which is the literature-based database of phenotypes?
answer: PheneBank

Category: 8 - Clinical grading & diagnostic scales / classification systems
question: What can be predicted with the Wells criteria?
answer: pulmonary embolism
question: Symptoms of which disorder are evaluated with the Davidson Trauma Scale?
answer: ['post-traumatic stress disorder', 'PTSD']
question: Which value of nuchal translucency thickness is set as the threshold for high-risk for Down Syndrome?
answer: 3mm

Category: 9 - Anatomical / cellular structures & localisation
question: Where is corticosterone synthesized?
answer: Adrenal glands
question: Which is the chromosome area that the human gene coding for the dopamine transporter (DAT1) is located to?
answer: 5p15.3
question: Where is the respirasome located?
answer: inner mitochondrial membrane

--------

OUTPUT FORMAT
Return question inside tag `<question>...</question>`, answer inside `<answer>...</answer>`. 
If the QA corresponde to one of the above categories put its number in <cat_num>...</cat_num> and category description in <cat>...</cat>

--------
TITLE AND ABSTRACT
{title_abstract}
"""

# Golden answers template
GOLDEN_ANSWERS_TEMPLATE = """I am generating a dataset for biological question answering.
Below I'll give a question with its target answer.

They are questions that have simple factoid answers - meaning the answer is a single entity.
However the same entity might have synonyms that are equivalent.
Your task is to return a python list of synonyms for each unique entity mentioned in the answers.

For example, if the target answer was "c-Jun NH2-terminal kinase", then you should return a list of synonyms ["JNK", "c-Jun N-terminal kinase", "c-Jun amino-terminal kinase", "c-Jun NH2-terminal kinase"]
If the answer was "two" then you should return ["two","2"]

OTHER GUIDANCE
The answers are correct and you may not disagree with the answer. 
Upper and lower case letters are treated the same.

OUTPUT FORMAT
Return a python list of all the common synonyms. Return as many as you can think of. Include the original answer(s) in the list as well.

You must respond with valid JSON list containing only the list of synonyms.

QUESTION
{question}
ANSWER
{answer}"""

# Template mapping
TEMPLATES = {1: TEMPLATE_1}


def parse_llm_response(response: str) -> Dict[str, str]:
    """
    Parse LLM response that contains XML-like tags for question, answer, category, etc.
    
    Expected format:
    <question>...</question>
    <answer>...</answer>
    <cat_num>...</cat_num>
    <cat>...</cat>
    
    Returns:
        Dict with keys: question, answer, cat_num, cat
    """
    result = {'question': '', 'answer': '', 'cat_num': '', 'cat': ''}

    # Define regex patterns for each tag
    patterns = {
        'question': r'<question>(.*?)</question>',
        'answer': r'<answer>(.*?)</answer>',
        'cat_num': r'<cat_num>(.*?)</cat_num>',
        'cat': r'<cat>(.*?)</cat>'
    }

    # Extract content from each tag
    for key, pattern in patterns.items():
        match = re.search(pattern, response, re.DOTALL | re.IGNORECASE)
        if match:
            result[key] = match.group(1).strip()

    return result


def process_batch(batch_data: List[Dict], template: str) -> List[Dict]:
    """Process a batch of abstracts through GPT using batch mode"""
    if not batch_data:
        return []

    # First, generate all prompts for the batch
    prompts = []
    for item in batch_data:
        title = item.get('title', '')
        abstract = item.get('abstractText', '')
        pmid = item.get('pmid', '')
        year = item.get('year', '')

        # Inline the format_title_abstract function
        title_abstract = f"TITLE: {title}. ABSTRACT: {abstract}. YEAR: {year}"

        # Replace template variable
        prompt = template.replace("{title_abstract}", title_abstract)
        prompts.append(prompt)

    # Send entire batch to GPT
    try:
        responses, _ = call_llm_batch(prompts=prompts,
                                      model_name="openai/gpt-4.1",
                                      max_tokens=1000,
                                      temperature=0.7,
                                      use_cache=True)

    except Exception as e:
        print(f"Error calling GPT-4 batch: {e}")
        # Return empty results for the batch if API call fails
        return []

    # Process responses and match them back to original items
    results = []
    for i, (item, response) in enumerate(zip(batch_data, responses)):
        title = item.get('title', '')
        pmid = item.get('pmid', '')

        # Parse response using the new parsing function
        try:
            parsed = parse_llm_response(response)

            # Only include if we got at least a question
            if parsed['question']:
                results.append({
                    'question': parsed['question'],
                    'answer': parsed['answer'],
                    'cat_num': parsed['cat_num'],
                    'cat': parsed['cat'],
                    'pmid': pmid,
                    'paper_title': title,
                    'raw_response': response  # Keep raw response for debugging
                })
            else:
                print(
                    f"Warning: No question found in response for PMID {pmid}")

        except Exception as e:
            print(f"Error parsing response for PMID {pmid}: {e}")
            print(f"Raw response: {response[:200]}...")
            continue

    return results


def generate_golden_answers(qa_pairs: List[Dict]) -> List[Dict]:
    """Generate golden_answers (synonyms) for each Q&A pair in batches of 50"""
    print("Generating golden answers (synonyms)...")

    batch_size = 50
    updated_qa_pairs = []

    # Process in batches of 50
    for i in tqdm(range(0, len(qa_pairs), batch_size),
                  desc="Generating golden answers"):
        batch = qa_pairs[i:i + batch_size]

        # Create prompts for the entire batch
        prompts = []
        for qa in batch:
            prompt = GOLDEN_ANSWERS_TEMPLATE.format(question=qa['question'],
                                                    answer=qa['answer'])
            prompts.append(prompt)

        # Call LLM for the entire batch
        try:
            responses, _ = call_llm_batch(prompts=prompts,
                                          model_name="openai/gpt-4.1",
                                          max_tokens=500,
                                          temperature=0.3,
                                          use_cache=True)

            # Process each response in the batch
            for qa, response in zip(batch, responses):
                # Create a copy of the QA pair to avoid modifying original
                updated_qa = qa.copy()

                try:
                    # Try to parse the JSON response
                    response_clean = response.strip()
                    golden_answers = json.loads(response_clean)

                    # Validate that it's a list
                    if isinstance(golden_answers,
                                  list) and len(golden_answers) > 0:
                        updated_qa['golden_answers'] = golden_answers
                    else:
                        # Fallback to original answer
                        updated_qa['golden_answers'] = [qa['answer']]
                        print(
                            f"Warning: Empty or invalid golden_answers format for question: {qa['question'][:50]}..."
                        )

                except json.JSONDecodeError as e:
                    # Fallback to original answer if JSON parsing fails
                    updated_qa['golden_answers'] = [qa['answer']]
                    print(
                        f"Warning: JSON decode error for question: {qa['question'][:50]}... Error: {e}"
                    )
                    print(f"Raw response: {response[:100]}...")

                except Exception as e:
                    # Catch any other parsing errors
                    updated_qa['golden_answers'] = [qa['answer']]
                    print(
                        f"Warning: Unexpected error parsing golden_answers for question: {qa['question'][:50]}... Error: {e}"
                    )

                updated_qa_pairs.append(updated_qa)

        except Exception as e:
            print(f"Error calling LLM batch for golden answers: {e}")
            # Add fallback for entire batch if LLM call fails
            for qa in batch:
                updated_qa = qa.copy()
                updated_qa['golden_answers'] = [qa['answer']]
                updated_qa_pairs.append(updated_qa)

    print(f"Generated golden answers for {len(updated_qa_pairs)} Q&A pairs")
    return updated_qa_pairs


def create_and_push_dataset(qa_pairs: List[Dict],
                            n_test: int = 200,
                            hub_name: str = "PaperSearchRL_n1500_test200"):
    """Create train/test split and push to HuggingFace Hub"""
    print(
        f"Creating dataset with {len(qa_pairs)} examples, test size: {n_test}")

    # Split into train and test (last n_test examples as test)
    if len(qa_pairs) > n_test:
        train_data = qa_pairs[:-n_test]
        test_data = qa_pairs[-n_test:]
    else:
        print(
            f"Warning: Not enough data for test split. Using all data as train."
        )
        train_data = qa_pairs
        test_data = []

    print(f"Train size: {len(train_data)}, Test size: {len(test_data)}")

    # Create HuggingFace datasets
    dataset_dict = {}

    if train_data:
        train_dataset = Dataset.from_list(train_data)
        dataset_dict['train'] = train_dataset

    if test_data:
        test_dataset = Dataset.from_list(test_data)
        dataset_dict['test'] = test_dataset

    # Create DatasetDict
    dataset = DatasetDict(dataset_dict)

    # Push to hub
    print(f"Pushing dataset to HuggingFace Hub as: {hub_name}")
    dataset.push_to_hub(hub_name)

    return dataset


def generate_dataset_from_abstracts(key: int = 1,
                                    n_samples: int = 1000,
                                    n_test: int = 200,
                                    hub_name: str = None):
    """Main function to generate dataset from abstracts"""
    print(f"Starting dataset generation with key={key}, n_samples={n_samples}")

    # Inline setup_results_directory
    results_dir = "results/generate_dataset_from_abstracts"
    os.makedirs(results_dir, exist_ok=True)

    # Get template
    if key not in TEMPLATES:
        raise ValueError(
            f"Template key {key} not found. Available keys: {list(TEMPLATES.keys())}"
        )

    template = TEMPLATES[key]

    # Load the indexer
    print("Loading PubMed abstracts database...")
    parquet_file = 'data/allMeSH_2022.parquet'
    if not os.path.exists(parquet_file):
        raise FileNotFoundError(f"Parquet file not found: {parquet_file}")

    indexer = return_indexer(parquet_file)
    total_length = len(indexer)
    print(f"Database loaded. Total articles: {total_length}")

    # Inline get_random_sample_indices (simplified)
    random.seed(0)
    np.random.seed(0)
    indices = random.sample(range(total_length), n_samples)
    print(f"Selected {len(indices)} random indices")

    # Process in batches of 50
    batch_size = 50
    all_results = []

    for i in tqdm(range(0, len(indices), batch_size),
                  desc="Processing batches"):
        batch_indices = indices[i:i + batch_size]

        # Get batch data
        batch_data = []
        for idx in batch_indices:
            try:
                item = indexer.iloc(idx)
                batch_data.append(item)
            except Exception as e:
                print(f"Error getting item at index {idx}: {e}")
                continue

        if not batch_data:
            continue

        # Process batch
        batch_results = process_batch(batch_data, template)
        all_results.extend(batch_results)

        # Save intermediate results
        if i % (batch_size * 10) == 0:  # Save every 10 batches
            temp_file = os.path.join(
                results_dir, f"temp_results_batch_{i//batch_size}.csv")
            pd.DataFrame(all_results).to_csv(temp_file, index=False)
            print(f"Saved intermediate results to {temp_file}")

    # Save initial results
    if all_results:
        df = pd.DataFrame(all_results)
        output_file = os.path.join(
            results_dir,
            f"generated_dataset_key_{key}_n_{n_samples}_initial.csv")
        df.to_csv(output_file, index=False)
        print(f"Initial Q&A results saved to {output_file}")
        print(f"Generated {len(all_results)} Q&A pairs")

        # Generate golden answers
        all_results_with_golden = generate_golden_answers(all_results)

        # Set debug breakpoint as requested
        ipdb.set_trace()

        # Save final results with golden answers
        df_final = pd.DataFrame(all_results_with_golden)
        final_output_file = os.path.join(
            results_dir,
            f"generated_dataset_key_{key}_n_{n_samples}_with_golden.csv")
        df_final.to_csv(final_output_file, index=False)
        print(
            f"Final results with golden answers saved to {final_output_file}")

        # Create and push to HuggingFace Hub
        if hub_name is None:
            hub_name = f"PaperSearchRL_n{len(all_results_with_golden)}_test{n_test}"

        dataset = create_and_push_dataset(all_results_with_golden,
                                          n_test=n_test,
                                          hub_name=hub_name)
        print(f"Dataset pushed to HuggingFace Hub: {hub_name}")

    else:
        print("No results generated")

    pass


if __name__ == "__main__":
    # You can modify these parameters or add command line argument parsing
    generate_dataset_from_abstracts(
        key=1,
        n_samples=1700,
        n_test=200,
        hub_name="jmhb/PaperSearchRL_v0_n1500_test200")
