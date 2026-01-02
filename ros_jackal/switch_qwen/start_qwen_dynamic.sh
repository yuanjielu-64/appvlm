#!/bin/bash
# 启动支持动态切换 checkpoint 的 Qwen 服务

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# 配置
BASE_MODEL="Qwen/Qwen2.5-VL-7B-Instruct"
INITIAL_CHECKPOINT="/home/yuanjielu/robot_navigation/noetic/appvlm_ws/src/ros_jackal/model/ddp/checkpoint-12500"
ALGORITHM="DDP"
HEAD_TYPE="dpt"
NUM_PARAMS=6
PORT=5000

# Conda 环境
#CONDA_PYTHON="/home/ylu22/miniforge/envs/lmms-finetune-qwen/bin/python"
CONDA_PYTHON="/home/yuanjielu/miniforge3/envs/lmms-finetune-qwen/bin/python"

# 环境变量
export TRANSFORMERS_VERBOSITY=error
export TOKENIZERS_PARALLELISM=false
export PYTHONWARNINGS="ignore::FutureWarning,ignore::UserWarning,ignore::DeprecationWarning"

echo "=================================================="
echo "  Qwen Dynamic Checkpoint Service"
echo "=================================================="
echo "Base Model:    ${BASE_MODEL}"
echo "Initial Checkpoint: ${INITIAL_CHECKPOINT}"
echo "Algorithm:     ${ALGORITHM}"
echo "Port:          ${PORT}"
echo "=================================================="

${CONDA_PYTHON} ${SCRIPT_DIR}/qwen_server_dynamic.py \
    --base_model "${BASE_MODEL}" \
    --checkpoint_path "${INITIAL_CHECKPOINT}" \
    --algorithm "${ALGORITHM}" \
    --head_type "${HEAD_TYPE}" \
    --num_params ${NUM_PARAMS} \
    --port ${PORT} \
    --load_in_4bit
