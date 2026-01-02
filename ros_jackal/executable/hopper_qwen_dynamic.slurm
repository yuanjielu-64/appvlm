#!/bin/bash
# ============================================================
# 启动支持动态切换 checkpoint 的 Qwen 服务
# ============================================================

#SBATCH --partition=gpuq
#SBATCH --qos=gpu
#SBATCH --job-name=qwen_dynamic
#SBATCH --output=cpu_report/qwen_dynamic-%j.out
#SBATCH --error=cpu_report/qwen_dynamic-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:A100.80gb:1
#SBATCH --mem=40G
#SBATCH --export=ALL
#SBATCH --cpus-per-task=8
#SBATCH --time=4-23:59:00

echo "=================================================="
echo "Qwen Dynamic Checkpoint Service"
echo "SLURM Job ID: $SLURM_JOB_ID"
echo "Running on node: $(hostname)"
echo "Node IP: $(hostname -I | awk '{print $1}')"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "=================================================="
echo ""
echo "🔔 IMPORTANT: Copy this for robot test:"
echo "   QWEN_HOST=\"$(hostname)\""
echo ""
echo "📡 API Endpoints:"
echo "   Health:  http://$(hostname):5000/health"
echo "   Infer:   http://$(hostname):5000/infer"
echo "   Switch:  http://$(hostname):5000/switch_checkpoint"
echo "   List:    http://$(hostname):5000/list_checkpoints"
echo "=================================================="

# 创建日志目录
mkdir -p cpu_report

# 激活 Python 3.10 Conda 环境
source /home/ylu22/miniforge/etc/profile.d/conda.sh
conda activate lmms-finetune-qwen

# 打印环境信息
echo "Python version: $(python --version)"
echo "PyTorch version: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"

# 切换到脚本目录
cd /scratch/ylu22/appvlm_ws/src/ros_jackal/switch_qwen

echo ""
echo "🚀 Starting Qwen Dynamic Service..."
echo ""

# 运行服务
bash start_qwen_dynamic.sh

echo "Qwen dynamic service stopped."
