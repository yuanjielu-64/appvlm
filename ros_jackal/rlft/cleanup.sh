#!/bin/bash
# RLFT DDP 训练清理脚本
# 用途：清理残留的训练进程、VLM 服务器、数据收集容器
#
# 保护规则：不清理 qwen 评估任务（s_* 开头的 tmux sessions）

echo "=========================================="
echo "  RLFT 清理脚本 - 清理训练相关进程"
echo "=========================================="

# 1. 杀死 Python 训练进程（train.py）
echo "[1/3] 正在终止 Python 训练进程..."
pkill -f "train.py" 2>/dev/null
pkill -f "torchrun.*train.py" 2>/dev/null
sleep 1
echo "      ✓ Python 训练进程已终止"

# 2. 杀死 RLFT VLM 服务器（tmux: ftrl_*）
echo "[2/3] 正在终止 VLM 服务器 (tmux sessions)..."
count=0
for session in $(tmux ls 2>/dev/null | grep "ftrl_" | cut -d: -f1); do
    # 保护规则：不清理 s_ 开头的 sessions（qwen 评估）
    if [[ ! "$session" =~ ^s_ ]]; then
        tmux kill-session -t "$session" 2>/dev/null && {
            echo "      ✓ 已终止 tmux session: $session"
            ((count++))
        }
    else
        echo "      ⊗ 跳过保护的 session: $session"
    fi
done

if [ $count -eq 0 ]; then
    echo "      ✓ 没有 VLM 服务器需要清理"
fi

# 3. 杀死数据收集容器（tmux: collect_*）
echo "[3/3] 正在终止数据收集容器 (tmux sessions)..."
count=0
for session in $(tmux ls 2>/dev/null | grep "collect_" | cut -d: -f1); do
    tmux kill-session -t "$session" 2>/dev/null && {
        echo "      ✓ 已终止 tmux session: $session"
        ((count++))
    }
done

if [ $count -eq 0 ]; then
    echo "      ✓ 没有数据收集容器需要清理"
fi

echo ""
echo "=========================================="
echo "  ✓ 清理完成！"
echo "=========================================="
echo ""
echo "提示：运行 'nvidia-smi' 检查 GPU 是否已释放"
echo "      运行 'tmux ls' 检查是否还有残留的 tmux sessions"
