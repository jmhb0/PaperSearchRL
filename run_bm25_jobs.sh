# all the bm25 scripts
python eval/infer.py \
	--checkpoint_path checkpoints/202507014_psr1_qwen3b_base_grpo_bm25/actor/global_step_180  \
	--model_id Qwen/Qwen2.5-3B \
	--dataset_name jmhb/papersearchr1 \
	--retriever_type bm25 \
	--first_n 250

# all the bm25 scripts
python eval/infer.py \
	--checkpoint_path checkpoints/202507013_psr1_qwen7b_it_grpo_bm25/actor/global_step_75  \
	--model_id Qwen/Qwen2.5-3B \
	--dataset_name jmhb/papersearchr1 \
	--retriever_type bm25 \
	--first_n 250

python eval/infer.py \
	--checkpoint_path checkpoints/202507013_psr1_qwen3b_it_grpo_bm25/actor/global_step_75  \
	--model_id Qwen/Qwen2.5-3B \
	--dataset_name jmhb/papersearchr1 \
	--retriever_type bm25 \
	--first_n 250


# two of the same things as above, but with earlier checkpoints 
python eval/infer.py \
	--checkpoint_path checkpoints/202507014_psr1_qwen3b_base_grpo_bm25/actor/global_step_60  \
	--model_id Qwen/Qwen2.5-3B \
	--dataset_name jmhb/papersearchr1 \
	--retriever_type bm25 \
	--first_n 250

python eval/infer.py \
	--checkpoint_path checkpoints/202507013_psr1_qwen7b_it_grpo_bm25/actor/global_step_30  \
	--model_id Qwen/Qwen2.5-3B \
	--dataset_name jmhb/papersearchr1 \
	--retriever_type bm25 \
	--first_n 250


