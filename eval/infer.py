import transformers
import torch
import random
from datasets import load_dataset
import requests
import os
import ipdb
import pandas as pd
from tqdm import tqdm
from verl.utils.reward_score.qa_em import compute_score_em
import re
import sys

curr_eos = [151645, 151643]  # for Qwen2.5 series models
curr_search_template = '\n\n{output_text}<information>{search_results}</information>\n\n'


# Define the custom stopping criterion
class StopOnSequence(transformers.StoppingCriteria):

    def __init__(self, target_sequences, tokenizer):
        # Encode the string so we have the exact token-IDs pattern
        self.target_ids = [
            tokenizer.encode(target_sequence, add_special_tokens=False)
            for target_sequence in target_sequences
        ]
        self.target_lengths = [len(target_id) for target_id in self.target_ids]
        self._tokenizer = tokenizer

    def __call__(self, input_ids, scores, **kwargs):
        # Make sure the target IDs are on the same device
        targets = [
            torch.as_tensor(target_id, device=input_ids.device)
            for target_id in self.target_ids
        ]

        if input_ids.shape[1] < min(self.target_lengths):
            return False

        # Compare the tail of input_ids with our target_ids
        for i, target in enumerate(targets):
            if torch.equal(input_ids[0, -self.target_lengths[i]:], target):
                return True

        return False


def get_query(text):
    import re
    pattern = re.compile(r"<search>(.*?)</search>", re.DOTALL)
    matches = pattern.findall(text)
    if matches:
        return matches[-1]
    else:
        return None


def search(query: str, topk: int = 3):
    payload = {"queries": [query], "topk": topk, "return_scores": True}

    try:
        response = requests.post("http://127.0.0.1:8000/retrieve",
                                 json=payload,
                                 timeout=10)
        response.raise_for_status()
        results = response.json()['result']
    except (requests.exceptions.ConnectionError,
            requests.exceptions.Timeout) as e:
        print(
            f"\n❌ ERROR: Cannot connect to retrieval server on localhost:8000")
        print(f"Connection error: {e}")
        print(
            "Please start the retrieval server before running SearchR1/PaperSearchR1 methods."
        )
        print("Example: python search_r1/search/retrieval_server.py")
        sys.exit(1)
    except requests.exceptions.RequestException as e:
        print(f"\n❌ ERROR: HTTP request to retrieval server failed")
        print(f"Request error: {e}")
        print("Please check the retrieval server status and configuration.")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ ERROR: Unexpected error during retrieval")
        print(f"Error: {e}")
        sys.exit(1)

    def _passages2string(retrieval_result):
        format_reference = ''
        for idx, doc_item in enumerate(retrieval_result):

            content = doc_item['document']['contents']
            title = content.split("\n")[0]
            text = "\n".join(content.split("\n")[1:])
            format_reference += f"Doc {idx+1}(Title: {title}) {text}\n"
        return format_reference

    return _passages2string(results[0])


def load_model_and_tokenizer(model_id, checkpoint_path=None, verbose=0):
    """Load model and tokenizer, optionally from a checkpoint."""
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_id)

    if checkpoint_path:
        assert os.path.exists(checkpoint_path)
        if verbose:
            print(f"Loading model from checkpoint: {checkpoint_path}")
        model = transformers.AutoModelForCausalLM.from_pretrained(
            checkpoint_path, torch_dtype=torch.bfloat16, device_map="auto")
    else:
        if verbose:
            print(f"Loading model from HuggingFace: {model_id}")
        model = transformers.AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=torch.bfloat16, device_map="auto")

    return tokenizer, model


def inference_with_search(question, tokenizer, model, verbose=0):
    """Run inference with search for a single question and return the full trace."""
    question = question.strip()
    if question[-1] != '?':
        question += '?'

    # Prepare the message
    prompt = f"""Answer the given question. \
You must conduct reasoning inside <think> and </think> first every time you get new information. \
After reasoning, if you find you lack some knowledge, you can call a search engine by <search> query </search> and it will return the top searched results between <information> and </information>. \
You can search as many times as your want. \
If you find no further external knowledge needed, you can directly provide the answer inside <answer> and </answer>, without detailed illustrations. For example, <answer> Beijing </answer>. Question: {question}\n"""

    # Initialize the stopping criteria
    target_sequences = [
        "</search>", " </search>", "</search>\n", " </search>\n",
        "</search>\n\n", " </search>\n\n"
    ]
    stopping_criteria = transformers.StoppingCriteriaList(
        [StopOnSequence(target_sequences, tokenizer)])

    cnt = 0
    full_trace = ""

    if tokenizer.chat_template:
        prompt = tokenizer.apply_chat_template([{
            "role": "user",
            "content": prompt
        }],
                                               add_generation_prompt=True,
                                               tokenize=False)

    if verbose:
        print(
            '\n\n################# [Start Reasoning + Searching] ##################\n\n'
        )
        print(prompt)
    full_trace += prompt

    # Encode the chat-formatted prompt
    # Let device_map="auto" handle device placement automatically
    while True:
        input_ids = tokenizer.encode(prompt, return_tensors='pt')
        # Move to the same device as the model's first parameter
        if hasattr(model, 'device'):
            input_ids = input_ids.to(model.device)
        else:
            # For models with device_map="auto", use the device of the first parameter
            input_ids = input_ids.to(next(model.parameters()).device)

        attention_mask = torch.ones_like(input_ids)

        # Generate text with the stopping criteria
        outputs = model.generate(input_ids,
                                 attention_mask=attention_mask,
                                 max_new_tokens=1024,
                                 stopping_criteria=stopping_criteria,
                                 pad_token_id=tokenizer.eos_token_id,
                                 do_sample=True,
                                 temperature=0.7)

        if outputs[0][-1].item() in curr_eos:
            generated_tokens = outputs[0][input_ids.shape[1]:]
            output_text = tokenizer.decode(generated_tokens,
                                           skip_special_tokens=True)
            if verbose:
                print(output_text)
            full_trace += output_text
            break

        generated_tokens = outputs[0][input_ids.shape[1]:]
        output_text = tokenizer.decode(generated_tokens,
                                       skip_special_tokens=True)

        tmp_query = get_query(
            tokenizer.decode(outputs[0], skip_special_tokens=True))
        if tmp_query:
            # print(f'searching "{tmp_query}"...')
            search_results = search(tmp_query)
        else:
            search_results = ''

        search_text = curr_search_template.format(
            output_text=output_text, search_results=search_results)
        prompt += search_text
        full_trace += search_text
        cnt += 1
        if verbose:
            print(search_text)

    return full_trace


def extract_answer(text):
    """Extract answer from <answer>...</answer> tags."""
    pattern = re.compile(r"<answer>(.*?)</answer>", re.DOTALL | re.IGNORECASE)
    matches = pattern.findall(text)
    if matches:
        return matches[-1].strip()
    else:
        return ""


def eval_inference_with_search(verbose=0):
    """Evaluate inference with search following the structure of eval_direct_inference.py"""
    if verbose:
        print("Loading model and tokenizer...")
    tokenizer, model = load_model_and_tokenizer(model_id,
                                                checkpoint_path,
                                                verbose=verbose)

    if verbose:
        print("Loading dataset...")
    # Load the test dataset
    dataset = load_dataset("jmhb/bioasq_trainv0_n1609_test100", split="test")

    # Convert to DataFrame
    df = dataset.to_pandas()
    if verbose:
        print(f"Loaded {len(df)} test examples")

    # Initialize results
    results = []

    if verbose:
        print("Running inference with search...")
    for idx, row in tqdm(df.iterrows(),
                         total=len(df),
                         desc="Processing questions"):
        question = row['question']
        golden_answers = row['golden_answers']

        # Run inference with search
        output_text = inference_with_search(question,
                                            tokenizer,
                                            model,
                                            verbose=verbose)

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
            score=1.0)

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
    if verbose:
        print(f"\nMean correctness score: {mean_score:.4f}")
        print(
            f"Total correct answers: {(results_df['correctness_score'] > 0).sum()}/{len(results_df)}"
        )

    # Save results to CSV
    output_path = "results/eval_inference_with_search/results.csv"
    results_df.to_csv(output_path, index=False)
    if verbose:
        print(f"Results saved to: {output_path}")

    # Save summary statistics
    summary_stats = {
        'mean_correctness_score': mean_score,
        'total_questions': len(results_df),
        'correct_answers': (results_df['correctness_score'] > 0).sum(),
        'accuracy_percentage':
        (results_df['correctness_score'] > 0).mean() * 100
    }

    summary_df = pd.DataFrame([summary_stats])
    summary_path = "results/eval_inference_with_search/summary.csv"
    summary_df.to_csv(summary_path, index=False)
    if verbose:
        print(f"Summary statistics saved to: {summary_path}")

    ipdb.set_trace()
    return results_df


# Process all questions and collect results
def eval_inference_with_search_questions(verbose=0):
    results = []
    tokenizer, model = load_model_and_tokenizer(model_id,
                                                checkpoint_path,
                                                verbose=verbose)
    for i, question in enumerate(questions):
        if verbose:
            print(f"\n{'='*50}")
            print(f"Processing question {i+1}/{len(questions)}")
            print(f"Question: {question}")
            print(f"{'='*50}")

        trace = inference_with_search(question,
                                      tokenizer,
                                      model,
                                      verbose=verbose)
        results.append(trace)

    if verbose:
        print(f"\nCompleted processing {len(questions)} questions.")
        print(
            f"Results collected in 'results' list with {len(results)} entries."
        )


if __name__ == "__main__":
    questions = [
        "Where, in the body, would the Cobb-Stainsby excision arthroplasty be performed?",
        "What is the origin of  HEp-2 cells?",
        "Which disease is associated with the ectopic expression of the protein encoded by the gene DUX4?",
        "What is disrupted by ALS- and FTD-associated missense mutations in TBK1?",
        "Covid-19 is though to have arisen from what species?",
        "Which hormone abnormalities are common in Williams syndrome ?",
    ]

    # Model ID setup (device removed since device_map="auto" handles it)
    # model_id = "PeterJinGo/SearchR1-nq_hotpotqa_train-qwen2.5-7b-em-ppo"
    model_id = 'Qwen/Qwen2.5-3B-Instruct'
    checkpoint_path = "data/verl_checkpoints/20250604_grpo_bioasqv0_fullcorpus_qwenit_bm25/actor/global_step_100/"
    # Remove the device variable since device_map="auto" handles device placement

    # Load model and tokenizer
    df = eval_inference_with_search(verbose=0)

    ipdb.set_trace()
    pass
