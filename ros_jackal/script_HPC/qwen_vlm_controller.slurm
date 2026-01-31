#!/bin/bash
# ============================================================
# 在 SLURM 中运行 Checkpoint 控制器
# 适用于长时间监控和自动评估
# ============================================================

#SBATCH --partition=normal
#SBATCH --job-name=qwen_controller
#SBATCH --output=cpu_report1/controller-%j.out
#SBATCH --error=cpu_report1/controller-%j.err
#SBATCH --time=5-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
# --mem-per-cpu removed (passed via command line instead)

# ============================================================
# 配置参数（通过 --export 传入）
# ============================================================
# 必需参数
CHECKPOINT_DIR=${CHECKPOINT_DIR:-/scratch/ylu22/appvlm_ws/src/ros_jackal/model/ddp/qwen2.5-vl-regression_lora-True_ddp_regression}
QWEN_HOST=${QWEN_HOST:-gpu011}

# 可选参数
QWEN_PORT=${QWEN_PORT:-5000}
START_WORLD=${START_WORLD:-0}
END_WORLD=${END_WORLD:-299}
RUNS_PER_WORLD=${RUNS_PER_WORLD:-3}
CHECK_INTERVAL=${CHECK_INTERVAL:-30}
WATCH_MODE=${WATCH_MODE:-true}  # true=监控模式, false=批量模式
ALGORITHM=${ALGORITHM:-}
NUM_PARAMS=${NUM_PARAMS:-}
CHECKPOINT_ORDER=${CHECKPOINT_ORDER:-}  # 自定义顺序，如 17500,15000,12500

echo "========================================="
echo "Checkpoint Controller (SLURM)"
echo "========================================="
echo "SLURM Job ID: $SLURM_JOB_ID"
echo "Running on: $(hostname)"
echo ""
echo "Configuration:"
echo "  Checkpoint dir: $CHECKPOINT_DIR"
echo "  Qwen service: $QWEN_HOST:$QWEN_PORT"
echo "  World range: $START_WORLD - $END_WORLD"
echo "  Runs per world: $RUNS_PER_WORLD"
echo "  Check interval: ${CHECK_INTERVAL}s"
echo "  Watch mode: $WATCH_MODE"
if [ -n "$ALGORITHM" ]; then echo "  Fixed algorithm: $ALGORITHM"; fi
if [ -n "$NUM_PARAMS" ]; then echo "  Fixed num_params: $NUM_PARAMS"; fi
if [ -n "$CHECKPOINT_ORDER" ]; then echo "  Checkpoint order: $CHECKPOINT_ORDER"; fi
echo "========================================="
echo ""

# 创建日志目录
mkdir -p cpu_report1

# 切换到项目根目录
cd /scratch/ylu22/appvlm_ws/src/ros_jackal

source /home/ylu22/miniforge/etc/profile.d/conda.sh
conda activate lmms-finetune-qwen

# 构建命令
CMD="python3 script_HPC/qwen_controller.py \
    --checkpoint_dir $CHECKPOINT_DIR \
    --qwen_host $QWEN_HOST \
    --qwen_port $QWEN_PORT \
    --start_world $START_WORLD \
    --end_world $END_WORLD \
    --runs_per_world $RUNS_PER_WORLD \
    --check_interval $CHECK_INTERVAL"

if [ "$WATCH_MODE" = "true" ]; then
    CMD="$CMD --watch"
fi

if [ -n "$ALGORITHM" ]; then
    CMD="$CMD --algorithm $ALGORITHM"
fi
if [ -n "$NUM_PARAMS" ]; then
    CMD="$CMD --num_params $NUM_PARAMS"
fi
if [ -n "$CHECKPOINT_ORDER" ]; then
    CMD="$CMD --checkpoint_order $CHECKPOINT_ORDER"
fi

echo "Executing command:"
echo "$CMD"
echo ""
echo "========================================="
echo ""

# 运行控制器
$CMD

echo ""
echo "========================================="
echo "Controller finished at: $(date '+%Y-%m-%d %H:%M:%S')"
echo "========================================="
