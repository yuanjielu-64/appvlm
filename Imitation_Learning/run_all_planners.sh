#!/bin/bash

# 使用 tmux 并行训练所有 planner
# 创建4个 tmux session: IL_dwa, IL_teb, IL_mppi, IL_ddp
# 全部使用 GPU 6

GPU_ID=6
PLANNERS=("dwa" "teb" "mppi" "ddp")

# 获取脚本所在目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=========================================="
echo "Starting training for all planners on GPU $GPU_ID"
echo "=========================================="

for planner in "${PLANNERS[@]}"; do
    SESSION_NAME="IL_${planner}"

    # 检查 session 是否已存在
    if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
        echo "Session $SESSION_NAME already exists, skipping..."
        continue
    fi

    echo "Creating tmux session: $SESSION_NAME"

    # 创建 tmux session 并运行训练
    tmux new-session -d -s "$SESSION_NAME" "
        cd $SCRIPT_DIR
        export CUDA_VISIBLE_DEVICES=$GPU_ID

        echo '=========================================='
        echo 'Training: $planner on GPU $GPU_ID'
        echo '=========================================='

        python train.py \
            --planner $planner \
            --data_root /data/local/yl2832/appvlm/ \
            --output_dir ./output/${planner}_transformer_il \
            --num_history_frames 2 \
            --batch_size 32 \
            --num_epochs 20 \
            --learning_rate 1e-4 \
            --eval_samples 2000 \
            --vision_model vit_large_patch16_224 \
            --num_transformer_layers 2 \
            --normalize_params \
            --use_velocity \
            --vision_pretrained \
            --mixed_precision \
            --num_workers 4 \
            --log_interval 50 \
            --eval_interval 1 \
            --save_interval 5 \
            --save_steps 5000 \
            --save_total_limit 20 \
            --save_best \
            --device cuda

        echo ''
        echo 'Training completed for $planner'
        echo 'Press Enter to close this session...'
        read
    "

    echo "  -> Session $SESSION_NAME started"
done

echo ""
echo "=========================================="
echo "All sessions started!"
echo "=========================================="
echo ""
echo "查看所有 session:  tmux ls"
echo "进入某个 session:  tmux attach -t IL_dwa"
echo "切换 session:      Ctrl+b, then s"
echo "退出但保持运行:    Ctrl+b, then d"
echo "关闭某个 session:  tmux kill-session -t IL_dwa"
echo "关闭所有 IL session: tmux kill-session -t IL_dwa; tmux kill-session -t IL_teb; tmux kill-session -t IL_mppi; tmux kill-session -t IL_ddp"
