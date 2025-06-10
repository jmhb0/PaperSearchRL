"""
Batch inference utilities for SearchR1 / PaperSearchR1.

This module builds on:
  • `eval/infer.py` – which shows a single-example SearchR1 loop
  • `search_r1.llm_agent.*` – which implements the complex multi-turn
     reasoning / search loop used during PPO training (see verl/trainer/main_ppo.py)

The goal here is to offer *fast* evaluation by:
  1.  Re-using the existing `LLMGenerationManager` logic so we do not have
      to re-implement the whole search-reasoning loop.
  2.  Replacing slow HuggingFace `model.generate` calls with the
      token-efficient `vllm` engine that supports continuous batching on GPU.

The public entry-point is the class `BatchSearchR1` with the method
`run_batch_searchr1(questions)` which returns (full_responses, predictions).

Example
-------
>>> from eval.infer_searchr1 import BatchSearchR1, load_vllm_model_and_tokenizer
>>> tokenizer, llm = load_vllm_model_and_tokenizer("Qwen/Qwen2.5-3B-Instruct")
>>> searchr1 = BatchSearchR1(model=llm, tokenizer=tokenizer)
>>> full, preds = searchr1.run_batch_searchr1(["What is the capital of France?", ...])
"""

from __future__ import annotations

import os
import re
from typing import List, Tuple, Optional

import ipdb
import torch
from tensordict import TensorDict
from vllm import LLM, SamplingParams

from verl import DataProto
from search_r1.llm_agent.generation import LLMGenerationManager, GenerationConfig
from search_r1.llm_agent.tensor_helper import TensorHelper, TensorConfig

# Re-use helper utilities from the single-example script
from eval.infer import load_model_and_tokenizer, get_query, search  # noqa: F401 – re-exported utils

__all__ = [
    "load_vllm_model_and_tokenizer",
    "BatchSearchR1",
]

# -----------------------------------------------------------------------------
# Light-weight loaders
# -----------------------------------------------------------------------------


def load_vllm_model_and_tokenizer(
    model_id: str,
    checkpoint_path: Optional[str] = None,
    gpu_memory_utilization: float = 0.9,
    max_model_len: int = 4096,
    trust_remote_code: bool = True,
):
    """Load tokenizer (HF) + vLLM engine.

    If *checkpoint_path* is supplied it will be given to vLLM's *model*
    argument so we can evaluate locally-trained checkpoints.
    """

    from transformers import AutoTokenizer  # local import to keep deps light

    model_source = checkpoint_path if checkpoint_path else model_id

    tokenizer = AutoTokenizer.from_pretrained(
        model_source, trust_remote_code=trust_remote_code)
    # Make sure we have a pad token
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    llm = LLM(
        model=model_source,
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=max_model_len,
        trust_remote_code=trust_remote_code,
    )

    return tokenizer, llm


# -----------------------------------------------------------------------------
# Internal: vLLM-powered wrapper that mimics the interface expected by
#           `LLMGenerationManager` (i.e. it has a `.generate_sequences` method
#           that takes a DataProto and returns a DataProto containing at least
#           the key ``responses``).
# -----------------------------------------------------------------------------


class _SearchR1VLLMWrapper:
    """Thin adapter so that vLLM works with LLMGenerationManager."""

    def __init__(self, llm: LLM, tokenizer, config: GenerationConfig):
        self.llm = llm
        self.tokenizer = tokenizer
        self.config = config
        # Helper to build masks / positions when returning tensors
        self._tensor_fn = TensorHelper(
            TensorConfig(
                pad_token_id=tokenizer.pad_token_id,
                max_prompt_length=config.max_prompt_length,
                max_obs_length=config.max_obs_length,
                max_start_length=config.max_start_length,
            ))

    # ------------------------------------------------------------------
    # NOTE: The *only* public method used by GenerationManager
    # ------------------------------------------------------------------

    def generate_sequences(self,
                           prompts: DataProto) -> DataProto:  # noqa: D401
        """Generate next responses for *each* prompt in *prompts* (batched).

        Parameters
        ----------
        prompts: DataProto
            Must contain tensors ``input_ids`` and ``attention_mask``.

        Returns
        -------
        DataProto
            With keys: ``responses`` (ids of next action), ``input_ids``
            (concatenated prompt+response), ``attention_mask``,
            ``position_ids``.
        """
        batch_input_ids: torch.Tensor = prompts.batch["input_ids"]  # [B, L]
        batch_attention: torch.Tensor = prompts.batch[
            "attention_mask"]  # [B, L]
        batch_size = batch_input_ids.size(0)

        # ------------------------------------------------------------------
        # 1) Decode *just* the non-pad tokens so we can feed strings to vLLM
        # ------------------------------------------------------------------
        prompt_texts: List[str] = []
        prompt_lengths: List[int] = []
        pad_id = self.tokenizer.pad_token_id
        for ids, mask in zip(batch_input_ids, batch_attention):
            valid_ids = ids[mask.bool()].tolist()
            prompt_lengths.append(len(valid_ids))
            prompt_texts.append(
                self.tokenizer.decode(valid_ids, skip_special_tokens=True))

        # ------------------------------------------------------------------
        # 2) Run vLLM generation in *one* batched call
        # ------------------------------------------------------------------
        sampling_params = SamplingParams(
            temperature=0.7,
            max_tokens=self.config.max_response_length,
            stop=["</search>", "</answer>", "<|im_end|>", "</s>"]
            # Keeping default top_p etc – can be adjusted via config
        )
        vllm_outputs = self.llm.generate(prompt_texts, sampling_params)

        # Extract generated strings (take the *first* candidate)
        gen_strs: List[str] = [out.outputs[0].text for out in vllm_outputs]

        # ------------------------------------------------------------------
        # 3) Tokenize generated responses & build tensor batch (left-pad)
        # ------------------------------------------------------------------
        resp_ids_list: List[torch.Tensor] = [
            torch.tensor(self.tokenizer.encode(s, add_special_tokens=False),
                         dtype=torch.long) for s in gen_strs
        ]
        max_resp_len = max(t.numel()
                           for t in resp_ids_list) if resp_ids_list else 1

        def _left_pad(t: torch.Tensor, pad_to: int,
                      pad_val: int) -> torch.Tensor:
            pad_len = pad_to - t.numel()
            if pad_len <= 0:
                return t
            return torch.cat(
                [torch.full((pad_len, ), pad_val, dtype=torch.long), t])

        resp_batch = torch.stack([
            _left_pad(t, max_resp_len, pad_id) for t in resp_ids_list
        ])  # [B, R]

        # ------------------------------------------------------------------
        # 4) Build full sequences (prompt + response) and masks/positions
        # ------------------------------------------------------------------
        full_seq_list: List[torch.Tensor] = []
        for i in range(batch_size):
            prompt_valid = batch_input_ids[i][batch_attention[i].bool()]
            full_seq = torch.cat([prompt_valid, resp_ids_list[i]])
            full_seq_list.append(full_seq)

        max_full_len = max(seq.numel() for seq in full_seq_list)
        full_ids = torch.full((batch_size, max_full_len),
                              pad_id,
                              dtype=torch.long)
        full_attention = torch.zeros((batch_size, max_full_len),
                                     dtype=torch.long)

        for i, seq in enumerate(full_seq_list):
            full_ids[i, -seq.numel():] = seq  # left-pad so tokens align at end
            full_attention[i, -seq.numel():] = 1

        position_ids = (torch.cumsum(full_attention, dim=1) -
                        1) * full_attention

        batch_dict = TensorDict(
            {
                "input_ids": full_ids,
                "responses": resp_batch,
                "attention_mask": full_attention,
                "position_ids": position_ids,
            },
            batch_size=batch_size,
        )

        return DataProto(batch=batch_dict)


# -----------------------------------------------------------------------------
# Public convenience wrapper
# -----------------------------------------------------------------------------


class BatchSearchR1:
    """High-level SearchR1 batch inference (vLLM-powered)."""

    def __init__(
        self,
        model: LLM,
        tokenizer,
        search_url: str = "http://127.0.0.1:8000/retrieve",
        topk: int = 3,
        max_turns: int = 10,
        max_prompt_len: int = 4096,
        max_response_len: int = 1024,
        max_obs_len: int = 2048,
        num_gpus: Optional[int] = None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.search_url = search_url
        self.topk = topk
        self.max_turns = max_turns

        # Tokenizer safety – ensure pad token exists
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        # Generation config reused by training code
        self.gen_config = GenerationConfig(
            max_turns=max_turns,
            max_start_length=0,  # will be updated per batch
            max_prompt_length=max_prompt_len,
            max_response_length=max_response_len,
            max_obs_length=max_obs_len,
            num_gpus=num_gpus,
            search_url=search_url,
            topk=topk,
        )

        # Determine number of GPUs for GenerationManager padding logic
        if num_gpus is None:
            try:
                import torch
                detected = torch.cuda.device_count()
                num_gpus = max(1, detected)
            except Exception:
                num_gpus = 1
        self.num_gpus = num_gpus

        print("[BatchSearchR1] Initialised (vLLM backend).")

    # ------------------------------------------------------------------
    # Helper: prompt construction (same as eval/infer.py logic)
    # ------------------------------------------------------------------

    def _build_prompts(self, questions: List[str]) -> List[str]:
        prompts = []
        for q in questions:
            q = q.strip()
            if not q.endswith("?"):
                q += "?"
            prompt = (
                "Answer the given question. "
                "You must conduct reasoning inside <think> and </think> first every time you get new information. "
                "After reasoning, if you find you lack some knowledge, you can call a search engine by <search> query </search> "
                "and it will return the top searched results between <information> and </information>. "
                "You can search as many times as your want. "
                "If you find no further external knowledge needed, you can directly provide the answer inside <answer> and </answer>, "
                "without detailed illustrations. For example, <answer> Beijing </answer>. "
                f"Question: {q}\n")
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

    # ------------------------------------------------------------------
    # Pipeline entry point
    # ------------------------------------------------------------------

    def run_batch_searchr1(
            self, questions: List[str]) -> Tuple[List[str], List[str]]:
        """Run the full SearchR1 loop for *questions*.

        Returns
        -------
        full_responses : List[str]
            The entire reasoning trace produced by the model.
        predictions : List[str]
            The extracted final answers (content inside <answer> tags).
        """
        if not questions:
            return [], []

        prompts = self._build_prompts(questions)

        # ------------------------------------------------------------------
        # Build *start* DataProto (only initial prompt)
        # ------------------------------------------------------------------
        enc = self.tokenizer(prompts,
                             add_special_tokens=False,
                             padding="longest",
                             return_tensors="pt")
        input_ids = enc["input_ids"]
        attention_mask = enc["attention_mask"]
        position_ids = torch.cumsum(attention_mask, dim=1) - 1

        start_batch = DataProto(batch=TensorDict(
            {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "position_ids": position_ids,
            },
            batch_size=len(prompts),
        ))

        # Update start length in config for proper clipping inside manager
        self.gen_config.max_start_length = input_ids.shape[1]

        # ------------------------------------------------------------------
        # Plug everything into GenerationManager
        # ------------------------------------------------------------------
        model_wrapper = _SearchR1VLLMWrapper(self.model, self.tokenizer,
                                             self.gen_config)
        manager = LLMGenerationManager(
            tokenizer=self.tokenizer,
            actor_rollout_wg=model_wrapper,
            config=self.gen_config,
        )

        final_output = manager.run_llm_loop(start_batch, input_ids)

        # ``final_output['input_ids']`` contains whole transcript; decode
        full_sequences = final_output.batch["input_ids"]
        full_responses: List[str] = self.tokenizer.batch_decode(
            full_sequences, skip_special_tokens=True)

        # Extract predictions (last <answer>...</answer>)
        predictions: List[str] = [
            self._extract_answer(resp) for resp in full_responses
        ]
        return full_responses, predictions

    # ------------------------------------------------------------------
    # Utility – extract answer from full trace
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_answer(text: str) -> str:
        pattern = re.compile(r"<answer>(.*?)</answer>",
                             re.DOTALL | re.IGNORECASE)
        matches = pattern.findall(text)
        if matches:
            return matches[-1].strip()
        return ""


# -----------------------------------------------------------------------------
# Quick test harness
# -----------------------------------------------------------------------------


def test():
    """Minimal sanity-check when this file is executed directly.

    Example usage (retrieval server must already be running):

    python -m eval.infer_searchr1 \
        --checkpoint_path checkpoints/20250606_papersearchr1v1_qwenit_bm25/global_step_100/ \
        --dataset_id jmhb/PaperSearchRL_v4_gv2_n3000_test500 \
        --model_id Qwen/Qwen2.5-3B-Instruct --first_n 20
    """

    import argparse
    import sys
    import requests
    from datasets import load_dataset
    from verl.utils.reward_score.qa_em import compute_score_em
    import time

    parser = argparse.ArgumentParser(
        description="Quick batch SearchR1 evaluation")
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default=
        "checkpoints/20250606_papersearchr1v1_qwenit_bm25/global_step_100/",
        help="Path to fine-tuned checkpoint directory")
    parser.add_argument("--model_id",
                        type=str,
                        default="Qwen/Qwen2.5-3B-Instruct",
                        help="Base model identifier (HF hub or local)")
    parser.add_argument(
        "--dataset_id",
        type=str,
        default="jmhb/PaperSearchRL_v4_gv2_n3000_test500",
        help="HF dataset to evaluate on (test split will be used)")
    parser.add_argument(
        "--first_n",
        type=int,
        default=20,
        help="Evaluate only the first N examples for a quick smoke test")
    parser.add_argument(
        "--num_gpus",
        type=int,
        default=None,
        help=
        "Number of GPUs to tell GenerationManager to expect (padding divisor). Default = auto-detect"
    )
    args = parser.parse_args()

    start_time = time.time()

    # ------------------------------------------------------------------
    # Check retrieval server is up
    # ------------------------------------------------------------------
    try:
        requests.get("http://127.0.0.1:8000/health", timeout=5)
    except Exception as e:
        print(
            "[ERROR] Retrieval server not reachable at http://127.0.0.1:8000 – please start it first."
        )
        print(f"Details: {e}")
        sys.exit(1)

    # ------------------------------------------------------------------
    # Load model + tokenizer
    # ------------------------------------------------------------------
    print("[LOAD] Model + tokenizer…")
    tokenizer, llm = load_vllm_model_and_tokenizer(
        args.model_id, checkpoint_path=args.checkpoint_path)

    # ------------------------------------------------------------------
    # Prepare dataset
    # ------------------------------------------------------------------
    print("[LOAD] Dataset…")
    dataset = load_dataset(args.dataset_id, split="test")
    if args.first_n > 0:
        dataset = dataset.select(range(min(args.first_n, len(dataset))))
    questions = dataset["question"]

    # PaperSearchRL variants use either "answer" or "golden_answers"
    if "answer" in dataset.column_names:
        gt_answers = dataset["answer"]
    else:
        gt_answers = dataset["golden_answers"]

    # ------------------------------------------------------------------
    # Run inference
    # ------------------------------------------------------------------
    print(f"[RUN] Running BatchSearchR1 on {len(questions)} examples…")
    bs_runner = BatchSearchR1(model=llm,
                              tokenizer=tokenizer,
                              num_gpus=args.num_gpus)
    full_traces, predictions = bs_runner.run_batch_searchr1(questions)

    # ------------------------------------------------------------------
    # Simple EM evaluation
    # ------------------------------------------------------------------
    correct = 0
    for pred, gold in zip(predictions, gt_answers):
        # Ensure list format for compute_score_em
        gold_list = gold if isinstance(gold, list) else [gold]
        em = compute_score_em(f"<answer>{pred}</answer>",
                              {"target": gold_list})
        if em > 0:
            correct += 1
    acc = correct / len(predictions)
    print(
        f"[RESULT] Exact-match accuracy: {correct}/{len(predictions)} = {acc:.2%}"
    )

    # Show a few samples
    for i in range(min(3, len(questions))):
        print("\n--- Example", i + 1)
        print("Q:", questions[i])
        print("Pred:", predictions[i])
        print("GT:", gt_answers[i])
        print("Trace snippet:", full_traces[i][:250].replace("\n", " ") + "…")

    elapsed = time.time() - start_time
    print(
        f"[TIME] Total elapsed: {elapsed:.2f} seconds ({elapsed/len(questions):.2f} s/question)"
    )
    ipdb.set_trace()
    pass


if __name__ == "__main__":
    test()
