#!/bin/bash
# Fine-tune Qwen2.5 VL on a single A100 80GB with ~60GB host RAM
# Adjust partition/qos/mail if your cluster policy differs.

#SBATCH --partition=gpuq
#SBATCH --qos=gpu
#SBATCH --job-name=qwen2_5_vl_7b_lora_a100_0_200k_ddp
#SBATCH --output=report/qwen2_5_vl_7b_lora_a100_0_200k_ddp.%j.out
#SBATCH --error=report/qwen2_5_vl_7b_lora_a100_0_200k_ddp.%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:A100.80gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --export=ALL
#SBATCH --time=0-72:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ylu22@gmu.edu

source $HOME/miniforge/bin/activate lmms-finetune-qwen

echo "Python Path:"
which python
python --version

# Cache locations (avoid filling home)
export HF_HOME=/scratch/$USER/hf_cache
export TRANSFORMERS_CACHE=/scratch/$USER/hf_cache
export WANDB_DIR=/scratch/ylu22/wandb
mkdir -p $HF_HOME $WANDB_DIR

# Launch finetune from project root to keep relative paths valid
cd /scratch/bwang25/appvlm_ws/src/vlm_pipeline/lmms-finetune-qwen
bash example_scripts/regression_example.sh
