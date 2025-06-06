"""
python -m ipdb data_gen/develop_abstract_to_question_pipeline.py

Classifying questions into categories. This is the result of doing multiple rounds with different prompts. 
"""
import os
import pandas as pd
from datasets import load_dataset
import re
from tqdm import tqdm
from data_gen.api import call_llm_batch
import ipdb

# User will paste the template here
CLASSIFY_TEMPLATE = """
You are a domain-expert biomedical NLP assistant.

GOAL  
For the single Question–Answer (Q–A) pair supplied to you, output  
• the best-matching category *number* (1-10) and name, or **none** if nothing fits  
• a concise reasoning sentence explaining your choice

CATEGORIES  
1. Genetic inheritance & disease-linked mutations  
2. Drug / treatment indications & molecular targets  
3. Protein function, localization & structure  
4. Molecular-biology techniques & acronyms  
5. Computational biology tools & algorithms  
6. Signalling pathways & enzymatic interactions  
7. Disease causation & pathogens  
8. Clinical trials & therapeutic guidelines  j
9. Biomarkers & diagnostic tests  
10. Epidemiology & quantitative-biology figures  

OUTPUT FORMAT (exactly)  
<classification> (number|none) - (category name or "none")</classification>
<reasoning> (1–2 sentences)</reasoning>

🟢 FEW-SHOT EXAMPLES
---------------------------------
Example A  
Question: What is the mode of inheritance of Wilson's disease?  
Answer: autosomal recessive  
Classification: 1 - Genetic inheritance & disease-linked mutations  
Reasoning: Wilson's disease question asks for hereditary pattern, directly matching inheritance/mutation category.

Example B  
Question: Roflumilast Cream is effective for which disease?  
Answer: psoriasis  
Classification: 2 - Drug / treatment indications & molecular targets  
Reasoning: Seeks therapeutic indication of a drug.

Example C  
Question: What is the function of the protein Magt1?  
Answer: Magnesium transporter  
Classification: 3 - Protein function, localization & structure  
Reasoning: Asks for molecular role of a specific protein.

Example D  
Question: What kind of chromatography is HILIC?  
Answer: Hydrophilic-Interaction Chromatography  
Classification: 4 - Molecular-biology techniques & acronyms  
Reasoning: Identifies a lab technique by its acronym.

Example E  
Question: Which algorithm extracts co-expressed gene clusters from expression data?  
Answer: Clust  
Classification: 5 - Computational biology tools & algorithms  
Reasoning: Asks for a bioinformatics algorithm/tool.

Example F  
Question: Which MAP-kinase phosphorylates the transcription factor c-Jun?  
Answer: JNK  
Classification: 6 - Signalling pathways & enzymatic interactions  
Reasoning: Concerns enzymatic interaction within a signalling cascade.

Example G  
Question: What causes Ocular Thelaziasis?  
Answer: *Thelazia callipaeda*  
Classification: 7 - Disease causation & pathogens  
Reasoning: Seeks etiological pathogen of a disease.

Example H  
Question: Treatment of which disease was investigated in the MR CLEAN study?  
Answer: acute ischaemic stroke  
Classification: 8 - Clinical trials & therapeutic guidelines  
Reasoning: References a named clinical trial and its disease focus.

Example I  
Question: When is serum AFP used as a marker?  
Answer: in hepatocellular carcinoma  
Classification: 9 - Biomarkers & diagnostic tests  
Reasoning: Asks about diagnostic use of a biomarker.

Example J  
Question: What is the incidence of cystic fibrosis in the Caucasian population?  
Answer: ≈ 1 in 7 000–10 000  
Classification: 10 - Epidemiology & quantitative-biology figures  
Reasoning: Requests a prevalence figure—an epidemiological metric.
---------------------------------

### NOW CLASSIFY
Provide your result for the **next** Q–A pair only, following the exact output format.
"""

CLASSIFY_TEMPLATE_1 = """
You are a domain-expert biomedical NLP assistant.

GOAL  
For the single Question–Answer (Q–A) pair supplied to you, output  
• the best-matching category *number* (1-10) and name, or **none** if nothing fits  
• a concise reasoning sentence explaining your choice


1. Genetic inheritance & disease-linked mutations
2. Therapeutics, indications & clinical evidence
3. Protein function, localization & signalling/enzymatic interactions
4. Experimental & computational methods, resources & acronyms
5. Disease causation & pathogens
6. Biomarkers & diagnostic tests
7. Epidemiology & quantitative-biology figures
8. Bioinformatics databases & curated resources	
9. Clinical grading & diagnostic scales / classification systems	
10. Anatomical / cellular structures & localisation

OUTPUT FORMAT (exactly)  
<classification> (number|none) - (category name or "none")</classification>
<reasoning> (1–2 sentences)</reasoning>

🟢 FEW-SHOT EXAMPLES
---------------------------------
Example A
Question: What is the mode of inheritance of Wilson's disease?
Answer: autosomal recessive
Classification: 1 - Genetic inheritance & disease-linked mutations
Reasoning: The question asks for the hereditary pattern of a disease, directly matching the inheritance/mutation category.

Example B
Question: Roflumilast Cream is effective for which disease?
Answer: psoriasis
Classification: 2 - Therapeutics, indications & clinical evidence
Reasoning: Seeks the therapeutic indication of a drug.

Example C
Question: What is the function of the protein Magt1?
Answer: Magnesium transporter
Classification: 3 - Protein function, localization & signalling/enzymatic interactions
Reasoning: Asks for the molecular role of a specific protein.

Example D
Question: What kind of chromatography is HILIC?
Answer: Hydrophilic-Interaction Chromatography
Classification: 4 - Experimental & computational methods, resources & acronyms
Reasoning: Identifies a laboratory technique by its acronym.

Example E
Question: What causes Ocular Thelaziasis?
Answer: Thelazia callipaeda
Classification: 5 - Disease causation & pathogens
Reasoning: Requests the etiological pathogen responsible for the disease.

Example F
Question: When is serum AFP used as a marker?
Answer: in hepatocellular carcinoma
Classification: 6 - Biomarkers & diagnostic tests
Reasoning: Asks about the diagnostic use of a biomarker.

Example G
Question: What is the incidence of cystic fibrosis in the Caucasian population?
Answer: ≈ 1 in 7 000–10 000
Classification: 7 - Epidemiology & quantitative-biology figures
Reasoning: Requests an epidemiological prevalence figure.

Example H
Question: Which database contains experimentally confirmed carbonylated proteins?” 
Answer: CarbonylDB
Classification: 8 - Bioinformatics databases & curated resources

Example I
Question: Where would Schlemm’s canal be found?
Answer: Eye
Classification: 9 - Clinical grading & diagnostic scales / classification systems

Example J
Question: 
Answer: '
Classification: 10 - Anatomical / cellular structures & localisation

---------------------------------
### NOW CLASSIFY
Provide your result for the **next** Q–A pair only, following the exact output format.
"""

RESULTS_DIR = "results/develop_abstract_to_question_pipeline/"


def classify_examples(n_examples: int = 100, template_num: int = 1):
    """
    Loads BioASQ examples, classifies them using an LLM, and saves the results.
    
    Args:
        n_examples: Number of examples to process
        template_num: Template to use (0 for CLASSIFY_TEMPLATE, 1 for CLASSIFY_TEMPLATE_1)
    """
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Select template based on template_num
    if template_num == 0:
        template = CLASSIFY_TEMPLATE
    elif template_num == 1:
        template = CLASSIFY_TEMPLATE_1
    else:
        raise ValueError(
            f"Invalid template_num: {template_num}. Must be 0 or 1.")

    # Load dataset from Hugging Face
    print("Loading BioASQ dataset...")
    dataset = load_dataset("jmhb/BioASQ-taskb", split='train')
    df = dataset.to_pandas()

    # Filter for "factoid" type questions
    print(f"Original dataset size: {len(df)}")
    df = df[df['type'] == 'factoid'].copy()
    print(f"Filtered to 'factoid' questions, new size: {len(df)}")

    # Get a sample of the dataset
    if n_examples > len(df):
        n_examples = len(df)
    df_sample = df.sample(n=n_examples, random_state=42).copy()
    print(
        f"Processing {len(df_sample)} examples with template {template_num}..."
    )

    # Create the prompt for each example
    def create_prompt(row):
        # Using "documents" and "summary" based on dataset viewer on HF
        # Assuming the first document is the relevant one for the answer.
        question = row['question']
        answer = row['answer']
        # The user said "Question: {question}\\n\nAnswer: {answer}"
        # but the dataset has a different structure.
        # I'll have to inspect the dataset columns. For now I'll assume 'question' and 'answer' columns exist
        formatted_qa = f"Question: {question}\n\nAnswer: {answer}"
        return f"{template}\n{formatted_qa}"

    df_sample['prompt'] = df_sample.apply(create_prompt, axis=1)

    prompts = df_sample['prompt'].tolist()

    # Call the LLM in batch
    print("Calling LLM for classification...")
    responses, costs = call_llm_batch(
        prompts=prompts,
        model_name="openai/gpt-4o",
        temperature=0.7,
        max_tokens=500,
        include_cost=True,
    )

    if costs:
        total_cost = sum(c for c in costs if c is not None)
        print(f"Total cost for this run: ${total_cost:.6f}")

    df_sample['llm_response'] = responses

    # Extract classification and reasoning
    def extract_tag(text, tag):
        if text is None:
            return ""
        match = re.search(f'<{tag}>(.*?)</{tag}>', text, re.DOTALL)
        return match.group(1).strip() if match else ""

    print("Extracting classification and reasoning...")
    df_sample['classification'] = df_sample['llm_response'].apply(
        lambda x: extract_tag(x, 'classification'))
    df_sample['reasoning'] = df_sample['llm_response'].apply(
        lambda x: extract_tag(x, 'reasoning'))

    # If the classification string begins with '(', remove it and the closing ')'.
    df_sample['classification'] = df_sample['classification'].apply(
        lambda x: x[1:-1]
        if isinstance(x, str) and x.startswith('(') and x.endswith(')') else x)

    def extract_class_number(classification_text: str) -> int:
        """Extracts the leading number from the classification string."""
        if not isinstance(classification_text, str):
            return -1
        # Match one or two digits at the start of the string, possibly after a '('
        match = re.match(r'^\s*\(?(\d{1,2})', classification_text)
        if match:
            return int(match.group(1))
        return -1

    df_sample['class'] = df_sample['classification'].apply(
        extract_class_number)

    # Save results with template number in filename
    output_path = os.path.join(RESULTS_DIR,
                               f"bioasq_classify_template_{template_num}.csv")
    df_sample.to_csv(output_path, index=False)
    print(f"Results saved to {output_path}")

    print("\nClass distribution:")
    print((df_sample.groupby(["class", "classification"
                              ])[["class", "classification"]].count() /
           len(df_sample) * 100).round(0))
    ipdb.set_trace()
    pass


SUBSET_CLASSIFY_TEMPLATE_0 = """
You are a domain-expert biomedical NLP assistant.

I will provide a list of questions and answers from a question-answering dataset. 
These questions all fall under the category of {category_name}.

GOAL
To identify a subset of questions (4) that are diverse and cover the range of attributes we see in these questions. 

OUTPUT FORMAT
example 0 
question: <question>
answer: <answer>

 example 1 
 question: <question>
 answer: <answer>

 example 1 
 question: <question>
 answer: <answer>
 
 Reasoning: <reasoning>
 ---- 
 Here are the sample question and answers you can choose from: 
 {questions_and_answers}
"""


def subset_classify_examples():
    """
    Uses SUBSET_CLASSIFY_TEMPLATE_0 to identify diverse examples for each category.
    
    Note: First run `classify_examples(n_examples=100, template_num=1)` to generate 
    the classification results CSV that this function loads.
    """
    # Load the results from previous classification run
    csv_path = os.path.join(RESULTS_DIR, "bioasq_classify_template_1.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"Results CSV not found at {csv_path}. "
            "Please run classify_examples(n_examples=100, template_num=1) first."
        )

    print(f"Loading classification results from {csv_path}...")
    df = pd.read_csv(csv_path)

    # Define categories
    categories = [
        ('1 - Genetic inheritance & disease-linked mutations', 1),
        ('2 - Therapeutics, indications & clinical evidence', 2),
        ('3 - Protein function, localization & signalling/enzymatic interactions',
         3),
        ('4 - Experimental & computational methods, resources & acronyms', 4),
        ('5 - Disease causation & pathogens', 5),
        ('6 - Biomarkers & diagnostic tests', 6),
        ('7 - Epidemiology & quantitative-biology figures', 7),
        ('8 - Bioinformatics databases & curated resources', 8),
        ('9 - Clinical grading & diagnostic scales / classification systems',
         9), ('10 - Anatomical / cellular structures & localisation', 10),
        ('none - none', -1)
    ]

    all_responses = {}

    for i, (category_name, class_num) in enumerate(categories):
        print(f"\nProcessing category {i+1}/8: {category_name}")

        # Filter data for this category
        if class_num == -1:
            # Handle 'none' category - filter for class values that are -1 or other non 1-7 values
            category_df = df[~df['class'].isin([1, 2, 3, 4, 5, 6, 7])].copy()
        else:
            category_df = df[df['class'] == class_num].copy()

        print(f"Found {len(category_df)} examples for this category")

        if len(category_df) == 0:
            print(
                f"No examples found for category {category_name}, skipping...")
            all_responses[category_name] = "No examples found"
            continue

        # Sample up to 30 Q&A pairs
        n_sample = min(30, len(category_df))
        sample_df = category_df.sample(n=n_sample, random_state=42)

        # Format questions and answers for the template
        qa_pairs = []
        for _, row in sample_df.iterrows():
            qa_pairs.append(
                f"Question: {row['question']}\nAnswer: {row['answer']}")

        questions_and_answers = "\n\n".join(qa_pairs)

        # Create the prompt
        prompt = SUBSET_CLASSIFY_TEMPLATE_0.format(
            category_name=category_name,
            questions_and_answers=questions_and_answers)

        # Call LLM
        print(f"Calling LLM for category: {category_name}")
        responses, costs = call_llm_batch(
            prompts=[prompt],
            model_name="openai/gpt-4.1",
            temperature=0.7,
            max_tokens=1000,
            include_cost=True,
        )

        if costs and costs[0]:
            print(f"Cost for this category: ${costs[0]:.6f}")

        # Store the response
        response = responses[0] if responses else "No response"
        all_responses[category_name] = response

    # Print all results
    print("\n" + "=" * 80)
    print("SUBSET CLASSIFICATION RESULTS")
    print("=" * 80)

    for category_name, response in all_responses.items():
        print(f"\n🔹 Category: {category_name}")
        print("-" * 50)
        print(response)
        print()

    ipdb.set_trace()
    pass

    return all_responses


if __name__ == "__main__":
    # classify_examples(n_examples=500, template_num=1)
    subset_classify_examples()
