#!/bin/bash
#SBATCH --partition=gpuq 
#SBATCH --qos=gpu 
#SBATCH --job-name=clip336_qwen2_5_dynamics_a100_batch_16
#SBATCH --output=gpu_basics.%j.out 
#SBATCH --error=gpu_basics.%j.out 
#SBATCH --nodes=1 
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:A100.80gb:1
#SBATCH --mem=96G
#SBATCH --export=ALL 
#SBATCH --time=0-72:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=bwang25@gmu.edu

# 激活 conda（Hopper 官方推荐）
source $HOME/miniforge/bin/activate llava

echo "Python Path:"
which python
python --version

# 配置 Cache（推荐）
export HF_HOME=/scratch/$USER/hf_cache
export TRANSFORMERS_CACHE=/scratch/$USER/hf_cache
export WANDB_DIR=/scratch/$USER/wandb

mkdir -p $HF_HOME $WANDB_DIR

# 启动训练
bash /home/bwang25/projects/LLaVA-NeXT/scripts/train/direct_finetune_clip_a100.sh
