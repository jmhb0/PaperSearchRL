"""
Batch SearchR1 inference evaluation.
This reuses the training code's LLMGenerationManager to handle the complex
asynchronous nature of SearchR1 where different questions can search at 
different times and need different numbers of searches.

Called by run_inference.py functions.
"""

import transformers
import torch
import numpy as np
from datasets import load_dataset
import requests
import pandas as pd
from tqdm import tqdm
from verl.utils.reward_score.qa_em import compute_score_em
import re
import os
import sys
from typing import List, Dict, Tuple, Optional
from tensordict import TensorDict
import ipdb

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from search_r1.llm_agent.generation import LLMGenerationManager, GenerationConfig
from search_r1.llm_agent.tensor_helper import TensorHelper, TensorConfig
from verl import DataProto
from eval.infer import load_model_and_tokenizer, StopOnSequence


class SearchR1ModelWrapper:
    """Simple wrapper to make SearchR1 model compatible with LLMGenerationManager interface"""

    def __init__(self, model, tokenizer, config):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.prompt_lengths = None

    def generate_sequences(self, prompts: DataProto) -> DataProto:
        """Generate sequences using SearchR1 model"""
        input_ids = prompts.batch['input_ids']
        attention_mask = prompts.batch['attention_mask']
        position_ids = prompts.batch['position_ids']

        batch_size = input_ids.shape[0]

        # Use standard generate with stopping criteria for search detection
        target_sequences = [
            "</search>", " </search>", "</search>\n", " </search>\n",
            "</search>\n\n", " </search>\n\n"
        ]
        stopping_criteria = transformers.StoppingCriteriaList(
            [StopOnSequence(target_sequences, self.tokenizer)])

        with torch.no_grad():
            outputs = self.model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=self.config.max_response_length,
                stopping_criteria=stopping_criteria,
                pad_token_id=self.tokenizer.eos_token_id,
                do_sample=True,
                temperature=0.7,
                return_dict_in_generate=True)

        # Extract responses (everything after the prompt)
        full_sequences = outputs.sequences
        responses = []
        for i in range(full_sequences.shape[0]):
            prompt_length = self.prompt_lengths[i]
            response = full_sequences[i, prompt_length:]
            responses.append(response)

        # Pad responses to be able to stack them
        max_response_len = max(r.shape[0] for r in responses)
        padded_responses = []
        for r in responses:
            pad_len = max_response_len - r.shape[0]
            padded_r = torch.cat([
                r,
                torch.full((pad_len, ),
                           self.tokenizer.pad_token_id,
                           device=r.device,
                           dtype=torch.long)
            ])
            padded_responses.append(padded_r)
        responses = torch.stack(padded_responses)

        # Create proper attention mask and position ids for responses
        response_length = responses.shape[1]
        response_attention_mask = torch.ones_like(responses)

        # Update position_ids for responses
        delta_position_id = torch.arange(1,
                                         response_length + 1,
                                         device=position_ids.device)
        delta_position_id = delta_position_id.unsqueeze(0).repeat(
            batch_size, 1)
        response_position_ids = position_ids[:, -1:] + delta_position_id

        # Combine everything
        new_attention_mask = torch.cat(
            [attention_mask, response_attention_mask], dim=1)
        new_position_ids = torch.cat([position_ids, response_position_ids],
                                     dim=1)

        batch_dict = TensorDict(
            {
                'input_ids': full_sequences,
                'responses': responses,
                'attention_mask': new_attention_mask,
                'position_ids': new_position_ids,
            },
            batch_size=batch_size)

        return DataProto(batch=batch_dict)


class BatchSearchR1:
    """Batch SearchR1 inference using the training code's LLMGenerationManager"""

    def __init__(self,
                 model,
                 tokenizer,
                 search_url: str = "http://127.0.0.1:8000/retrieve",
                 topk: int = 3):
        """Initialize BatchSearchR1 with SearchR1 model and tokenizer."""
        self.model = model
        self.tokenizer = tokenizer
        self.search_url = search_url
        self.topk = topk

        # Ensure pad_token_id is set
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        print("BatchSearchR1 initialized successfully!")

    def prepare_searchr1_prompts(self, questions: List[str]) -> List[str]:
        """Prepare SearchR1 prompts in the same format as individual inference."""
        prompts = []

        for question in questions:
            if not question.endswith('?'):
                question = question + '?'

            prompt = f"""first every time you get new information. \\
After reasoning, if you find you lack some knowledge, you can call a search engine by <search> query </search> and it will return the top searched results between <information> and </information>. \\
You can search as many times as your want. \\
If you find no further external knowledge needed, you can directly provide the answer inside <answer> and </answer>, without detailed illustrations. For example, <answer> Beijing </answer>. Question: {question}\n"""

            if self.tokenizer.chat_template:
                prompt = self.tokenizer.apply_chat_template(
                    [{
                        "role": "user",
                        "content": prompt
                    }],
                    add_generation_prompt=True,
                    tokenize=False)

            prompts.append(prompt)

        return prompts

    def prepare_batch_data(
            self, prompts: List[str]) -> Tuple[DataProto, int, List[int]]:
        """Tokenize and prepare batch data for LLMGenerationManager."""
        # Tokenize all prompts
        input_ids = []
        prompt_lengths = []
        for prompt in prompts:
            ids = self.tokenizer.encode(prompt, return_tensors='pt')
            input_ids.append(ids.squeeze(0))
            prompt_lengths.append(ids.shape[1])

        # Pad to same length
        max_len = max(len(ids) for ids in input_ids)
        padded_input_ids = []
        padded_attention_masks = []

        for ids in input_ids:
            pad_len = max_len - len(ids)
            if pad_len > 0:
                padded_ids = torch.cat([
                    torch.full((pad_len, ), self.tokenizer.pad_token_id), ids
                ])
                padded_mask = torch.cat(
                    [torch.zeros(pad_len),
                     torch.ones(len(ids))])
            else:
                padded_ids = ids
                padded_mask = torch.ones(len(ids))
            padded_input_ids.append(padded_ids)
            padded_attention_masks.append(padded_mask)

        batch_input_ids = torch.stack(padded_input_ids)
        batch_attention_masks = torch.stack(padded_attention_masks)

        # Create position_ids
        position_ids = []
        for mask in batch_attention_masks:
            pos_ids = torch.cumsum(mask, dim=0) - 1
            pos_ids[mask == 0] = 0
            position_ids.append(pos_ids)
        batch_position_ids = torch.stack(position_ids)

        # Move to device
        device = next(self.model.parameters()).device
        batch_input_ids = batch_input_ids.to(device)
        batch_attention_masks = batch_attention_masks.to(device)
        batch_position_ids = batch_position_ids.to(device)

        # Create DataProto
        batch_dict = TensorDict(
            {
                'input_ids': batch_input_ids,
                'attention_mask': batch_attention_masks,
                'position_ids': batch_position_ids,
            },
            batch_size=len(prompts))

        return DataProto(batch=batch_dict), max_len, prompt_lengths

    def extract_prediction(self, response: str) -> str:
        """Extract the final prediction from the model response."""
        response = response.strip()

        # Use regex to find all <answer>...</answer> patterns
        answer_matches = re.findall(r'<answer>(.*?)</answer>', response,
                                    re.DOTALL)

        if answer_matches:
            # For searchr1, use the last occurrence if available
            if len(answer_matches) >= 1:
                prediction = answer_matches[-1].strip()
            else:
                prediction = answer_matches[0].strip()

            if prediction:
                return prediction

        # Fallback patterns using regex
        fallback_patterns = [
            r'Final Answer:\s*(.*?)(?:\n|$)',
            r'Answer:\s*(.*?)(?:\n|$)',
        ]

        for pattern in fallback_patterns:
            match = re.search(pattern, response, re.IGNORECASE)
            if match:
                prediction = match.group(1).strip()
                if prediction:
                    break
        else:
            # Take the first meaningful line
            lines = response.split('\n')
            for line in lines:
                line = line.strip()
                if line and not re.match(
                        r'^(Question:|Context:|Instructions:|Let\'s think)',
                        line, re.IGNORECASE):
                    prediction = line
                    break
            else:
                # Fallback to first sentence
                sentences = response.split('.')
                prediction = sentences[0].strip() if sentences else response

        # Clean up prediction - remove common prefixes using regex
        prefix_pattern = r'^(the answer is |the final answer is |therefore,? |thus,? |in conclusion,? |so,? |hence,? )'
        prediction = re.sub(prefix_pattern,
                            '',
                            prediction,
                            flags=re.IGNORECASE).strip()

        # Keep only first line
        prediction = prediction.split('\n')[0].strip()

        return prediction

    def run_batch_searchr1(
            self, questions: List[str]) -> Tuple[List[str], List[str]]:
        """Run complete batch SearchR1 pipeline."""
        print(f"Running batch SearchR1 for {len(questions)} questions...")

        # Step 1: Prepare prompts
        prompts = self.prepare_searchr1_prompts(questions)

        # Step 2: Prepare batch data
        gen_batch, max_len, prompt_lengths = self.prepare_batch_data(prompts)
        # ipdb.set_trace()
        # pass

        # Step 3: Setup LLMGenerationManager
        generation_config = GenerationConfig(max_turns=10,
                                             max_start_length=max_len,
                                             max_prompt_length=4096,
                                             max_response_length=1024,
                                             max_obs_length=2048,
                                             num_gpus=1,
                                             search_url=self.search_url,
                                             topk=self.topk)

        model_wrapper = SearchR1ModelWrapper(self.model, self.tokenizer,
                                             generation_config)
        model_wrapper.prompt_lengths = prompt_lengths  # Pass lengths to wrapper

        llm_manager = LLMGenerationManager(tokenizer=self.tokenizer,
                                           actor_rollout_wg=model_wrapper,
                                           config=generation_config)

        # Step 4: Run the batch generation with the training code!
        initial_input_ids = gen_batch.batch['input_ids']
        final_output = llm_manager.run_llm_loop(gen_batch, initial_input_ids)

        # Step 5: Extract responses and decode
        responses = []
        predictions = []
        for i in range(len(questions)):
            full_sequence = final_output.batch['input_ids'][i]
            response_text = self.tokenizer.decode(full_sequence,
                                                  skip_special_tokens=True)
            prediction = self.extract_prediction(response_text)

            responses.append(response_text)
            predictions.append(prediction)

        return responses, predictions


def eval_batch_searchr1(papersearchr1_model_path,
                        dataset_id="jmhb/PaperSearchRL_v1_n10000_test500",
                        model_id="Qwen/Qwen2.5-3B-Instruct",
                        first_n=50):
    """Evaluate batch SearchR1 on the test dataset (standalone function)."""
    print("Starting BatchSearchR1 evaluation...")

    print(f"Model: {model_id}")
    print(f"PaperSearchR1 path: {papersearchr1_model_path}")
    print(f"Dataset: {dataset_id}")
    print(f"Testing first {first_n} examples")

    # Load model and tokenizer
    print("Loading PaperSearchR1 model...")
    if not os.path.exists(papersearchr1_model_path):
        print(
            f"Warning: Model path {papersearchr1_model_path} does not exist. Using base model."
        )
        papersearchr1_model_path = None

    tokenizer, model = load_model_and_tokenizer(model_id,
                                                papersearchr1_model_path)
    print("Model loaded successfully!")

    # Initialize BatchSearchR1
    batch_searchr1 = BatchSearchR1(model=model,
                                   tokenizer=tokenizer,
                                   search_url="http://127.0.0.1:8000/retrieve",
                                   topk=3)

    # Load dataset
    print("Loading dataset...")
    dataset = load_dataset(dataset_id, split="test")
    df = dataset.to_pandas()

    # Limit to first N examples
    if first_n > 0 and len(df) > first_n:
        df = df.head(first_n)

    print(f"Loaded {len(df)} test examples")

    # Extract questions and answers
    questions = df['question'].tolist()
    golden_answers_list = df['answer'].tolist()

    print("Running batch SearchR1 inference...")
    # Run batch SearchR1
    responses, predictions = batch_searchr1.run_batch_searchr1(questions)

    # Evaluate results
    print("Evaluating results...")
    final_results = []

    for i, (question, response,
            prediction) in enumerate(zip(questions, responses, predictions)):
        golden_answer = golden_answers_list[i]

        # Compute exact match score
        if isinstance(golden_answer, list):
            ground_truth = {'target': golden_answer}
        else:
            ground_truth = {'target': [golden_answer]}

        # The compute_score_em function expects answer tags in the response
        pred_str = f"<answer></answer><answer>{prediction}</answer>"
        em_score = compute_score_em(pred_str, ground_truth)

        result = {
            'question': question,
            'golden_answer': golden_answer,
            'full_response': response,
            'extracted_prediction': prediction,
            'em_score': em_score
        }
        final_results.append(result)

    # Create results DataFrame
    results_df = pd.DataFrame(final_results)

    # Calculate metrics
    mean_em_score = results_df['em_score'].mean()
    correct_count = (results_df['em_score'] > 0).sum()
    total_count = len(results_df)

    print(f"\n=== BATCH SEARCHR1 EVALUATION RESULTS ===")
    print(f"Mean EM score: {mean_em_score:.4f}")
    print(f"Correct answers: {correct_count}/{total_count}")
    print(f"Accuracy: {(correct_count/total_count)*100:.2f}%")

    # Save results
    os.makedirs("results/batch_searchr1", exist_ok=True)
    output_path = "results/batch_searchr1/results.csv"
    results_df.to_csv(output_path, index=False)
    print(f"Results saved to: {output_path}")

    # Print some example results
    print(f"\n=== EXAMPLE RESULTS ===")
    for i in range(min(3, len(results_df))):
        row = results_df.iloc[i]
        print(f"\nExample {i+1}:")
        print(f"Question: {row['question']}")
        print(f"Golden Answer: {row['golden_answer']}")
        print(f"Prediction: {row['extracted_prediction']}")
        print(f"EM Score: {row['em_score']}")
        print(f"Full Response: {row['full_response'][:200]}...")

    return results_df


if __name__ == "__main__":
    # Check if retrieval server is running
    try:
        response = requests.get("http://127.0.0.1:8000/health", timeout=5)
        print("Retrieval server is running")
    except:
        print("ERROR: Retrieval server is not running on localhost:8000")
        print(
            "Please start the retrieval server before running SearchR1 evaluation:"
        )
        print("Example: python search_r1/search/retrieval_server.py")
        sys.exit(1)

    # Run standalone evaluation
    papersearchr1_model_path = "./checkpoints/20250606_papersearchr1v1_qwenit_bm25/global_step_100/"
    first_n = 50
    model_id = "Qwen/Qwen2.5-3B-Instruct"
    dataset_id = "jmhb/PaperSearchRL_v1_n10000_test500"
    results_df = eval_batch_searchr1(papersearchr1_model_path,
                                     model_id=model_id,
                                     dataset_id=dataset_id,
                                     first_n=first_n)

    if results_df is not None:
        print("\nEvaluation completed successfully!")
    else:
        print("\nEvaluation failed!")
