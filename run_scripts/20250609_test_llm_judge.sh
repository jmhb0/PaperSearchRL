#! /bin/bash
set -e 
export WORLD_SIZE=1
export RANK=0
export LOCAL_RANK=0
export MASTER_ADDR=localhost
export MASTER_PORT=29500

## configs
export TOTAL_GPUS=${TOTAL_GPUS:-1}  # Set total GPUs available 
GPU_COUNT=4

export BASE_MODEL='Qwen/Qwen2.5-3B-Instruct'
export EXPERIMENT_NAME=$(basename "$0" .sh)
echo $EXPERIMENT_NAME

export HF_HOME=data/hf_cache
export TRANSFORMERS_CACHE=data/hf_cache
export HF_DATASETS_CACHE=data/hf_cache
mkdir -p data/hf_cache


DATA_SOURCE=jmhb/PaperSearchRL_v4_gv2_n3000_test500
CUDA_VISIBLE_DEVICES=0,1,2,3

## training 
export DATA_DIR="data/${DATA_SOURCE}_search/"
export TRAIN_DATA_DIR=$DATA_DIR
export TEST_DATA_DIR=$DATA_DIR
# export RETRIEVER_URL='http://pasteur6.stanford.edu:8000/retrieve'
export RETRIEVER_URL='http://0.0.0.0:8000/retrieve'
WAND_PROJECT='Search-R1'

# copy the dataset over 
python scripts/data_process/nq_search.py --data_source $DATA_SOURCE


export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# export BASE_MODEL='meta-llama/Llama-3.2-3B'
# export EXPERIMENT_NAME=nq-search-r1-grpo-llama3.2-3b-em
# export BASE_MODEL='meta-llama/Llama-3.2-3B-Instruct'
# export EXPERIMENT_NAME=nq-search-r1-grpo-llama3.2-3b-it-em
# export BASE_MODEL='meta-llama/Llama-3.1-8B'
# export EXPERIMENT_NAME=nq-search-r1-grpo-llama3.1-8b-em
# export BASE_MODEL='meta-llama/Llama-3.1-8B-Instruct'
# export EXPERIMENT_NAME=nq-search-r1-grpo-llama3.1-8b-it-em

# export BASE_MODEL='Qwen/Qwen2.5-3B'
# export EXPERIMENT_NAME=nq-search-r1-grpo-qwen2.5-3b-em

# export BASE_MODEL='Qwen/Qwen2.5-3B-Instruct'
# export EXPERIMENT_NAME=20250530_grpo_nq_qwenit_e5
# export EXPERIMENT_NAME=20250601_grpo_nqhotpot_qwenit_e5
# export EXPERIMENT_NAME=nq-search-r1-grpo-qwen2.5-3b-it-em
# export EXPERIMENT_NAME=nq-1000-search-r1-grpo-qwen2.5-3b-it-em
# export EXPERIMENT_NAME=nq-512-search-r1-grpo-qwen2.5-3b-it-em
# export EXPERIMENT_NAME=nq-5000-search-r1-grpo-qwen2.5-3b-it-em
# export EXPERIMENT_NAME=nq-5000-search-r1-grpo-qwen2.5-3b-it-em-bm25

# export BASE_MODEL='Qwen/Qwen2.5-7B'
# export EXPERIMENT_NAME=nq-search-r1-grpo-qwen2.5-7b-em
# export BASE_MODEL='Qwen/Qwen2.5-7B-Instruct'
# export EXPERIMENT_NAME=nq-search-r1-grpo-qwen2.5-7b-it-em

# set -x
export VLLM_ATTENTION_BACKEND=XFORMERS # vllm + qwen2-7b with flash_attn has some issues
# max_prompt_length = (config['training']['max_start_length'] + config['training']['max_response_length'] * (config['training']['max_turns'] - 1) + config['training']['max_obs_length'] * config['training']['max_turns'])

# Adjust batch sizes based on available training GPUs
if [ "$GPU_COUNT" -eq 1 ] || [ "$GPU_COUNT" -eq 2 ]; then
  BATCH_SIZE=256          
  VAL_BATCH_SIZE=128      
  PPO_MINI_BATCH_SIZE=128 
  PPO_MICRO_BATCH_SIZE=32
elif [ "$GPU_COUNT" -eq 4 ]; then
  BATCH_SIZE=512
  VAL_BATCH_SIZE=256
  PPO_MINI_BATCH_SIZE=256
  PPO_MICRO_BATCH_SIZE=64
elif [ "$GPU_COUNT" -eq 6 ]; then  # Separate case for 6 GPUs
  BATCH_SIZE=480  # 480 * 5 = 2400, which is divisible by 6 (2400/6 = 400)
  VAL_BATCH_SIZE=240
  PPO_MINI_BATCH_SIZE=240
  PPO_MICRO_BATCH_SIZE=60
elif [ "$GPU_COUNT" -eq 8 ]; then
  BATCH_SIZE=512
  VAL_BATCH_SIZE=256
  PPO_MINI_BATCH_SIZE=256
  PPO_MICRO_BATCH_SIZE=64
else
  echo "Warning: Unexpected GPU_COUNT=$GPU_COUNT, using default settings"
  BATCH_SIZE=512
  VAL_BATCH_SIZE=256
  PPO_MINI_BATCH_SIZE=256
  PPO_MICRO_BATCH_SIZE=64
fi

# BATCH_SIZE=60
# PPO_MINI_BATCH_SIZE=20
# PPO_MICRO_BATCH_SIZE=10
# VAL_BATCH_SIZE=20




echo "Training configuration for $GPU_COUNT GPUs:"
echo "  BATCH_SIZE=$BATCH_SIZE"
echo "  VAL_BATCH_SIZE=$VAL_BATCH_SIZE"
echo "  PPO_MINI_BATCH_SIZE=$PPO_MINI_BATCH_SIZE"
echo "  PPO_MICRO_BATCH_SIZE=$PPO_MICRO_BATCH_SIZE"

# Clean up any existing Ray processes first
echo "Cleaning up existing Ray processes..."
ray stop --force 2>/dev/null || true
pkill -f ray 2>/dev/null || true
pkill -f raylet 2>/dev/null || true
sleep 3

# Start Ray cluster with only the training GPUs
export RAY_DEBUG=legacy

echo "Starting Ray cluster with $GPU_COUNT GPUs..."
ray start --head --num-cpus=64 --num-gpus=$GPU_COUNT --object-store-memory=10000000000 --ray-debugger-external

# Wait a moment for Ray to fully initialize
sleep 5

# Set Ray environment variables
export RAY_DISABLE_IMPORT_WARNING=1
export RAY_OBJECT_STORE_ALLOW_SLOW_STORAGE=1
export RAY_TMPDIR=/tmp/ray_tmp_$(whoami)
mkdir -p $RAY_TMPDIR
export RAY_memory_monitor_refresh_ms=1000
export RAY_object_spilling_threshold=0.8

# Set RAY_ADDRESS to connect to the cluster we just started
export RAY_ADDRESS="auto"



PYTHONUNBUFFERED=1 python3 -m verl.trainer.main_ppo \
    data.train_files=$TRAIN_DATA_DIR/train.parquet \
    data.val_files=$TEST_DATA_DIR/test.parquet \
    data.train_data_num=null \
    data.val_data_num=null \
    data.train_batch_size=$BATCH_SIZE \
    data.val_batch_size=$VAL_BATCH_SIZE \
    data.max_prompt_length=4096 \
    data.max_response_length=500 \
    data.max_start_length=2048 \
    data.max_obs_length=500 \
    data.shuffle_train_dataloader=True \
    algorithm.adv_estimator=grpo \
    actor_rollout_ref.model.path=$BASE_MODEL \
    actor_rollout_ref.model.enable_gradient_checkpointing=true \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.285 \
    actor_rollout_ref.actor.use_kl_loss=true \
    actor_rollout_ref.actor.ppo_mini_batch_size=$PPO_MINI_BATCH_SIZE \
    actor_rollout_ref.actor.ppo_micro_batch_size=$PPO_MICRO_BATCH_SIZE \
    actor_rollout_ref.actor.fsdp_config.param_offload=true \
    actor_rollout_ref.actor.fsdp_config.grad_offload=true \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=true \
    actor_rollout_ref.rollout.log_prob_micro_batch_size=128 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.17 \
    actor_rollout_ref.ref.log_prob_micro_batch_size=128 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    algorithm.no_think_rl=false \
    actor_rollout_ref.rollout.n_agent=5 \
    actor_rollout_ref.rollout.temperature=1 \
    actor_rollout_ref.actor.state_masking=true \
    trainer.logger=[] \
    +trainer.val_only=false \
    +trainer.val_before_train=true \
    trainer.default_hdfs_dir=null \
    trainer.n_gpus_per_node=$GPU_COUNT \
    trainer.nnodes=1 \
    trainer.save_freq=100 \
    trainer.test_freq=50 \
    trainer.project_name=$WAND_PROJECT \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.total_epochs=10000 \
    trainer.total_training_steps=1005 \
    trainer.default_hdfs_dir=null \
    trainer.default_local_dir=data/verl_checkpoints/$EXPERIMENT_NAME \
    reward_model.use_llm_judge=True \
    reward_model.llm_judge_model="openai/gpt-4.1-nano" \
    reward_model.llm_judge_max_concurrent=50 \
    max_turns=2 \
    retriever.url=$RETRIEVER_URL \
    retriever.topk=3 \
    2>&1 | tee data/${EXPERIMENT_NAME}.log
