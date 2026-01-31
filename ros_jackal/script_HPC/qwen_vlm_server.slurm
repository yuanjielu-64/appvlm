#!/bin/bash
# ============================================================
# Hopper 集群 SLURM 脚本 - Qwen VLM 推理服务
# ============================================================
# 使用场景：
#   - 通过 SLURM 在 Hopper GPU 节点上运行
#   - 正式训练和大规模评估
#
# 提交方式：
#   bash script_HPC/submit_single_qwen.sh DWA
#   bash script_HPC/submit_all_qwen.sh
#
# 如果要在本地调试，请使用：
#   bash script_HPC/start_qwen_dynamic.sh
# ============================================================

#SBATCH --partition=gpuq
#SBATCH --qos=gpu
#SBATCH --job-name=qwen_dynamic
#SBATCH --output=cpu_report/qwen_dynamic-%j.out
#SBATCH --error=cpu_report/qwen_dynamic-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:3g.40gb:1
#SBATCH --mem=40G
#SBATCH --export=ALL
#SBATCH --cpus-per-task=8
#SBATCH --time=4-23:59:00

echo "=================================================="
echo "Qwen Dynamic Checkpoint Service"
echo "=================================================="
echo "SLURM Job ID: $SLURM_JOB_ID"
echo "Running on node: $(hostname)"
echo "Node IP: $(hostname -I | awk '{print $1}')"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo ""
echo "Configuration:"
echo "  Initial Checkpoint: ${INITIAL_CHECKPOINT:-NOT_SET}"
echo "  Algorithm: ${ALGORITHM:-NOT_SET}"
echo "  Num Params: ${NUM_PARAMS:-NOT_SET}"
echo "  Port: ${PORT:-5000}"
echo "=================================================="
echo ""
echo "🔔 IMPORTANT: Copy this for robot test:"
echo "   QWEN_HOST=\"$(hostname)\""
echo ""
echo "📡 API Endpoints:"
echo "   Health:  http://$(hostname):${PORT:-5000}/health"
echo "   Infer:   http://$(hostname):${PORT:-5000}/infer"
echo "   Switch:  http://$(hostname):${PORT:-5000}/switch_checkpoint"
echo "   List:    http://$(hostname):${PORT:-5000}/list_checkpoints"
echo "=================================================="

# 创建日志目录
mkdir -p cpu_report

# 激活 Python 3.10 Conda 环境
source /home/ylu22/miniforge/etc/profile.d/conda.sh
conda activate lmms-finetune-qwen

# 打印环境信息
echo ""
echo "Python version: $(python --version)"
echo "PyTorch version: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo ""

# 切换到脚本目录
SCRIPT_DIR="/scratch/ylu22/appvlm_ws/src/ros_jackal/script_HPC"
cd ${SCRIPT_DIR}

# 设置 Python 完整路径（确保使用正确的 conda 环境）
CONDA_PYTHON="/home/ylu22/miniforge/envs/lmms-finetune-qwen/bin/python"

# 设置默认值
BASE_MODEL="${BASE_MODEL:-Qwen/Qwen2.5-VL-7B-Instruct}"
HEAD_TYPE="${HEAD_TYPE:-dpt}"
PORT="${PORT:-5000}"

# 验证必需参数（这些参数必须从 sbatch 传入）
echo "=================================================="
echo "  Received Parameters"
echo "=================================================="
echo "  ALGORITHM: ${ALGORITHM:-NOT_SET}"
echo "  NUM_PARAMS: ${NUM_PARAMS:-NOT_SET}"
echo "  INITIAL_CHECKPOINT: ${INITIAL_CHECKPOINT:-NOT_SET}"
echo "=================================================="
echo ""

if [ -z "$INITIAL_CHECKPOINT" ] || [ -z "$ALGORITHM" ] || [ -z "$NUM_PARAMS" ]; then
    echo "❌ Error: Missing required parameters!"
    echo "  INITIAL_CHECKPOINT, ALGORITHM, and NUM_PARAMS must be set"
    exit 1
fi

# 设置 Python 环境变量（减少日志噪音）
export TRANSFORMERS_VERBOSITY=error
export TOKENIZERS_PARALLELISM=false
export PYTHONWARNINGS="ignore::FutureWarning,ignore::UserWarning,ignore::DeprecationWarning"

echo "🚀 Starting Qwen VLM Service..."
echo ""
echo "Service Configuration:"
echo "  Base Model:    ${BASE_MODEL}"
echo "  Checkpoint:    ${INITIAL_CHECKPOINT}"
echo "  Algorithm:     ${ALGORITHM}"
echo "  Head Type:     ${HEAD_TYPE}"
echo "  Num Params:    ${NUM_PARAMS}"
echo "  Port:          ${PORT}"
echo "=================================================="
echo ""

# 使用 qwen_server.py（支持动态切换 checkpoint）
${CONDA_PYTHON} ${SCRIPT_DIR}/qwen_server.py \
    --base_model "${BASE_MODEL}" \
    --lora_path "${INITIAL_CHECKPOINT}" \
    --algorithm "${ALGORITHM}" \
    --head_type "${HEAD_TYPE}" \
    --num_params ${NUM_PARAMS} \
    --port ${PORT} \
    --load_in_4bit

echo ""
echo "Qwen VLM service stopped."

