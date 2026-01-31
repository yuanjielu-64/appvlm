#!/bin/bash

# 启动 Transformer IL 推理服务
# 为每个 planner 启动一个服务，使用不同端口

# ============================================================
# 配置
# ============================================================
GPU_ID=0
#CHECKPOINT_BASE="/home/yuanjielu/robot_navigation/noetic/appvlm_ws/src/Imitation_Learning/output"
CHECKPOINT_BASE="/data/local/yl2832/appvlm_ws/src/Imitation_Learning/output"
# 端口配置: DWA=6000, TEB=6001, MPPI=6002, DDP=6003
declare -A PORTS
PORTS["dwa"]=6000
PORTS["teb"]=6001
PORTS["mppi"]=6002
PORTS["ddp"]=6003

# 模型配置 (必须与训练时一致)
# ViT-Base: vit_base_patch16_224 (768-d)
# ViT-Large: vit_large_patch16_224 (1024-d)
VISION_MODEL="vit_base_patch16_224"
NUM_TRANSFORMER_LAYERS=2
NUM_HISTORY_FRAMES=2

# Python 环境 (使用与训练相同的环境)
#PYTHON_ENV="/home/yuanjielu/miniforge3/envs/lmms-finetune-qwen/bin/python"
PYTHON_ENV="/common/home/yl2832/miniconda3/envs/lmms-finetune-qwen/bin/python"

# ============================================================
# 函数：启动单个服务
# ============================================================
start_service() {
    local planner=$1
    local port=${PORTS[$planner]}
    local checkpoint_dir="${CHECKPOINT_BASE}/${planner}_transformer_il"
    local checkpoint_file="${checkpoint_dir}/best_model.pth"
    local session_name="IL_server_${planner}"
    local algorithm="${planner^^}"

    # 检查 checkpoint 文件是否存在
    if [ ! -f "$checkpoint_file" ]; then
        echo "[ERROR] Checkpoint not found: $checkpoint_file"
        return 1
    fi

    # 检查 session 是否已存在
    if tmux has-session -t "$session_name" 2>/dev/null; then
        echo "[WARN] Session $session_name already exists, skipping..."
        return 0
    fi

    echo "Starting IL service for $planner on port $port..."

    tmux new-session -d -s "$session_name" "\
        export CUDA_VISIBLE_DEVICES=${GPU_ID} && \
        echo '==========================================' && \
        echo 'IL Service: ${planner}' && \
        echo 'Port: ${port}' && \
        echo 'Checkpoint: ${checkpoint_file}' && \
        echo 'GPU: ${GPU_ID}' && \
        echo '==========================================' && \
        cd ${SCRIPT_DIR} && \
        ${PYTHON_ENV} il_server.py \
            --checkpoint_path ${checkpoint_file} \
            --algorithm ${algorithm} \
            --vision_model ${VISION_MODEL} \
            --num_transformer_layers ${NUM_TRANSFORMER_LAYERS} \
            --num_history_frames ${NUM_HISTORY_FRAMES} \
            --device cuda \
            --port ${port}; \
        echo ''; \
        echo 'Service stopped. Press Enter to close...'; \
        read"

    echo "  -> Session $session_name started on port $port"
}

# ============================================================
# 主程序
# ============================================================

# 获取脚本目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=========================================="
echo "Starting Transformer IL Services"
echo "GPU: $GPU_ID"
echo "=========================================="

# 解析命令行参数
if [ $# -eq 0 ]; then
    # 启动所有服务
    PLANNERS=("dwa" "teb" "mppi" "ddp")
else
    # 启动指定的服务
    PLANNERS=("$@")
fi

for planner in "${PLANNERS[@]}"; do
    planner_lower=$(echo "$planner" | tr '[:upper:]' '[:lower:]')
    start_service "$planner_lower"
done

echo ""
echo "=========================================="
echo "Services started!"
echo "=========================================="
echo ""
echo "端口映射:"
echo "  DWA:  http://localhost:6000"
echo "  TEB:  http://localhost:6001"
echo "  MPPI: http://localhost:6002"
echo "  DDP:  http://localhost:6003"
echo ""
echo "常用命令:"
echo "  查看所有 session:  tmux ls"
echo "  进入某个 session:  tmux attach -t IL_server_dwa"
echo "  测试服务:          curl http://localhost:6000/health"
echo "  关闭所有服务:      ./stop_il_service.sh"
