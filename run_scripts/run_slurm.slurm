#!/bin/bash
#SBATCH --job-name=searchr1-grpo
#SBATCH --partition=pasteur
#SBATCH --account=pasteur
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=300G
#SBATCH --gres=gpu:2
#SBATCH --time=48:00:00
#SBATCH --output=logs/grpo_8gpus_%j.out
#SBATCH --error=logs/grpo_8gpus_%j.err
#SBATCH --mail-type=END,FAIL,REQUEUE
#SBATCH --mail-user=jmhb@stanford.edu
#SBATCH --exclude=pasteur[1-4]

# Print some job info
echo "SLURM_JOB_ID: $SLURM_JOB_ID"
echo "SLURM_JOB_NODELIST: $SLURM_JOB_NODELIST"
echo "SLURM_GPUS_ON_NODE: $SLURM_GPUS_ON_NODE"
echo "Starting time: $(date)"

# Change to the directory where the script is located
cd /pasteur/u/jmhb/search_r1

# Set environment variables that might be needed
export CUDA_DEVICE_ORDER=PCI_BUS_ID

# Check GPU availability
nvidia-smi

# Run the training script
echo "Starting training..."
bash train_grpo.sh

echo "Training completed at: $(date)" 
