#!/bin/bash
# DDP 多 GPU 训练启动脚本（带自动清理）
#
# 使用方法：
#   ./launch_ddp.sh [NUM_GPUS] [其他参数]
#
# 示例：
#   ./launch_ddp.sh 3  # 使用 3 张 GPU
#   ./launch_ddp.sh 3 --config_file ddp --policy_name ddp_rlft
#   ./launch_ddp.sh 3 --skip_test  # 跳过 test 评估
#
# 日志：
#   - 自动保存到 ./ddp_train.log（覆盖旧日志）
#   - 同时输出到终端
#
# 注意：
#   - 4-bit 量化模型不兼容 DataParallel，必须使用 DDP
#   - 每个进程加载自己的模型副本到对应的 GPU
#   - 使用 accelerate 进行梯度同步
#   - 按 Ctrl+C 会自动清理所有子进程和 GPU 资源

set -e

# 默认参数
NUM_GPUS=${1:-3}
shift 2>/dev/null || true  # 移除第一个参数（NUM_GPUS）

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CLEANUP_SCRIPT="$SCRIPT_DIR/cleanup.sh"
LOG_FILE="$SCRIPT_DIR/ddp_train.log"

# 检查 GPU 数量
AVAILABLE_GPUS=$(nvidia-smi -L | wc -l)
if [ "$NUM_GPUS" -gt "$AVAILABLE_GPUS" ]; then
    echo "错误：请求 $NUM_GPUS 张 GPU，但只有 $AVAILABLE_GPUS 张可用"
    exit 1
fi

# 日志函数：同时输出到终端和日志文件
log() {
    echo "$@" | tee -a "$LOG_FILE"
}

# 清空旧日志并写入启动信息
echo "========================================" > "$LOG_FILE"
echo "  DDP 训练日志" >> "$LOG_FILE"
echo "  启动时间: $(date '+%Y-%m-%d %H:%M:%S')" >> "$LOG_FILE"
echo "========================================" >> "$LOG_FILE"

echo "========================================"
echo "  DDP 多 GPU 训练 (带自动清理)"
echo "========================================"
echo "  GPU 数量: $NUM_GPUS"
echo "  可用 GPU: $AVAILABLE_GPUS"
echo "  额外参数: $@"
echo "  日志文件: $LOG_FILE"
echo "========================================"
echo "  提示: 按 Ctrl+C 会自动清理所有进程"
echo "========================================"
echo ""

# ========== 信号处理函数 ==========
cleanup_and_exit() {
    echo ""
    echo "=========================================="
    echo "  🛑 收到退出信号 (Ctrl+C)，正在清理..."
    echo "=========================================="

    # 1. 优雅地终止 torchrun 进程组
    if [ -n "$TORCHRUN_PID" ]; then
        echo "[1/3] 优雅终止 torchrun 进程组 (PID: $TORCHRUN_PID)..."

        # Step 1: 向所有子进程发送 SIGTERM（优雅退出）
        pkill -TERM -P $TORCHRUN_PID 2>/dev/null || true
        kill -TERM $TORCHRUN_PID 2>/dev/null || true

        # 也终止 tee 进程
        if [ -n "$TEE_PID" ]; then
            kill -TERM $TEE_PID 2>/dev/null || true
        fi

        # Step 2: 等待进程退出（最多5秒）
        echo "      等待进程优雅退出..."
        for i in {1..5}; do
            if ! kill -0 $TORCHRUN_PID 2>/dev/null; then
                echo "      ✓ 进程已正常退出"
                break
            fi
            sleep 1
        done

        # Step 3: 如果还没退出，强制杀死
        if kill -0 $TORCHRUN_PID 2>/dev/null; then
            echo "      强制终止残留进程..."
            pkill -9 -P $TORCHRUN_PID 2>/dev/null || true
            kill -9 $TORCHRUN_PID 2>/dev/null || true
            sleep 1
        fi

        echo "      ✓ torchrun 已终止"
    fi

    # 2. 清理残留的 Python 进程（防止有漏网之鱼）
    echo "[2/3] 清理残留的训练进程..."
    pkill -9 -f "train.py" 2>/dev/null || true
    pkill -9 -f "torchrun.*train.py" 2>/dev/null || true
    echo "      ✓ 残留进程已清理"

    # 3. 运行清理脚本（清理 tmux sessions）
    echo "[3/3] 运行清理脚本..."
    if [ -f "$CLEANUP_SCRIPT" ]; then
        bash "$CLEANUP_SCRIPT"
    else
        echo "      ⚠ 清理脚本不存在: $CLEANUP_SCRIPT"
        # 手动清理 tmux
        for session in $(tmux ls 2>/dev/null | grep -E "ftrl_|collect_" | cut -d: -f1); do
            # 跳过 s_ 开头的 sessions（qwen 评估）
            if [[ ! "$session" =~ ^s_ ]]; then
                tmux kill-session -t "$session" 2>/dev/null || true
            fi
        done
    fi

    echo "=========================================="
    echo "  ✓ 清理完成，已安全退出"
    echo "=========================================="
    exit 0
}

# 注册信号处理器
trap cleanup_and_exit SIGINT SIGTERM

# ========== 启动训练 ==========
# 使用 torchrun 启动 DDP 训练
# --nproc_per_node: 每个节点的进程数（即 GPU 数）
# --master_port: 主节点端口（避免与其他训练冲突）
MASTER_PORT=${MASTER_PORT:-29501}

cd "$SCRIPT_DIR"

# 抑制 DeepSpeed 详细日志
export DEEPSPEED_LOG_LEVEL=ERROR
export DS_LOG_LEVEL=ERROR
export TRANSFORMERS_VERBOSITY=error

# NCCL 超时设置（默认 600 秒太短，test 收集可能超过 10 分钟）
export NCCL_TIMEOUT=1800  # 30 分钟
export TORCH_NCCL_BLOCKING_WAIT=1

echo "开始训练（按 Ctrl+C 安全退出）..."
echo "日志保存到: $LOG_FILE"
echo ""

# 启动 torchrun，输出同时到终端和日志文件
# 使用 stdbuf 禁用缓冲，确保实时输出
torchrun \
    --nproc_per_node=$NUM_GPUS \
    --master_port=$MASTER_PORT \
    train.py "$@" 2>&1 | tee -a "$LOG_FILE" &

TEE_PID=$!

# 获取 torchrun 的实际 PID（tee 的父进程的子进程）
sleep 1
TORCHRUN_PID=$(pgrep -P $$ -f "torchrun" | head -1)
if [ -z "$TORCHRUN_PID" ]; then
    TORCHRUN_PID=$TEE_PID
fi
echo "Torchrun PID: $TORCHRUN_PID, Tee PID: $TEE_PID"

# 等待完成
wait $TEE_PID
EXIT_CODE=$?

# 正常退出
echo ""
echo "=========================================="
if [ $EXIT_CODE -eq 0 ]; then
    echo "  ✓ 训练正常完成"
else
    echo "  ✗ 训练退出，退出码: $EXIT_CODE"
fi
echo "  日志已保存到: $LOG_FILE"
echo "=========================================="
