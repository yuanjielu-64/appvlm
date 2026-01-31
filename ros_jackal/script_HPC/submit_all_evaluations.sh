#!/bin/bash
# 直接提交 4 个 planner 的评估任务（运行 300 个环境）

set -e  # 遇到错误立即退出

echo "=================================================="
echo "提交 4 个 Planner 的评估任务"
echo "=================================================="

# ============================================================
# 配置区域
# ============================================================

# GPU 服务配置（必须与 submit_all_planners.sh 一致）
DWA_HOST="gpu011"
TEB_HOST="gpu011"
MPPI_HOST="gpu014"
DDP_HOST="gpu016"

DWA_PORT=5001
TEB_PORT=5002
MPPI_PORT=5003
DDP_PORT=5004

# 参数数量
DWA_PARAMS=9
TEB_PARAMS=9
MPPI_PARAMS=10
DDP_PARAMS=8

# 评估配置
START_WORLD=0
END_WORLD=299
RUNS_PER_WORLD=1

# Policy 名称（用于结果文件命名）
DWA_POLICY="dwa_qwen"
TEB_POLICY="teb_qwen"
MPPI_POLICY="mppi_qwen"
DDP_POLICY="ddp_qwen"

# Checkpoint 路径（可选，如果不指定会从 health API 获取）
# 如果你想评估特定 checkpoint，在这里指定：
# DWA_CHECKPOINT="/scratch/bwang25/.../checkpoint-2500"
# TEB_CHECKPOINT="/scratch/bwang25/.../checkpoint-2500"
# MPPI_CHECKPOINT="/scratch/bwang25/.../checkpoint-2500"
# DDP_CHECKPOINT="/scratch/bwang25/.../checkpoint-2500"

echo ""
echo "配置信息："
echo "  DWA:  $DWA_HOST:$DWA_PORT (参数: $DWA_PARAMS, policy: $DWA_POLICY)"
echo "  TEB:  $TEB_HOST:$TEB_PORT (参数: $TEB_PARAMS, policy: $TEB_POLICY)"
echo "  MPPI: $MPPI_HOST:$MPPI_PORT (参数: $MPPI_PARAMS, policy: $MPPI_POLICY)"
echo "  DDP:  $DDP_HOST:$DDP_PORT (参数: $DDP_PARAMS, policy: $DDP_POLICY)"
echo ""
echo "评估范围："
echo "  环境: $START_WORLD - $END_WORLD (共 $((END_WORLD - START_WORLD + 1)) 个)"
echo "  每个环境运行: $RUNS_PER_WORLD 次"
echo ""

# ============================================================
# 验证 GPU 服务是否运行
# ============================================================

echo "验证 GPU 服务状态..."
echo ""

check_service() {
    local host=$1
    local port=$2
    local alg=$3

    if curl -s --max-time 5 http://${host}:${port}/health > /dev/null 2>&1; then
        echo "✓ $alg 服务正常 ($host:$port)"
        return 0
    else
        echo "❌ $alg 服务无法访问 ($host:$port)"
        return 1
    fi
}

get_checkpoint_number() {
    local host=$1
    local port=$2

    local ckpt_path=$(curl -s --max-time 5 http://${host}:${port}/health | python3 -c "
import sys, json
try:
    data = json.load(sys.stdin)
    print(data['current_checkpoint'])
except:
    pass
" 2>/dev/null)

    # 提取 checkpoint-XXXX 中的数字
    if [[ $ckpt_path =~ checkpoint-([0-9]+) ]]; then
        echo "${BASH_REMATCH[1]}"
    else
        echo "unknown"
    fi
}

check_service "$DWA_HOST" "$DWA_PORT" "DWA" || exit 1
check_service "$TEB_HOST" "$TEB_PORT" "TEB" || exit 1
check_service "$MPPI_HOST" "$MPPI_PORT" "MPPI" || exit 1
check_service "$DDP_HOST" "$DDP_PORT" "DDP" || exit 1

echo ""
echo "✓ 所有 GPU 服务验证通过"
echo ""

# 获取 checkpoint 编号
echo "获取 checkpoint 编号..."
DWA_CKPT=$(get_checkpoint_number "$DWA_HOST" "$DWA_PORT")
TEB_CKPT=$(get_checkpoint_number "$TEB_HOST" "$TEB_PORT")
MPPI_CKPT=$(get_checkpoint_number "$MPPI_HOST" "$MPPI_PORT")
DDP_CKPT=$(get_checkpoint_number "$DDP_HOST" "$DDP_PORT")

echo "  DWA:  checkpoint-$DWA_CKPT"
echo "  TEB:  checkpoint-$TEB_CKPT"
echo "  MPPI: checkpoint-$MPPI_CKPT"
echo "  DDP:  checkpoint-$DDP_CKPT"
echo ""

# ============================================================
# 提交评估任务
# ============================================================

echo "提交评估任务..."
echo ""

# DWA 评估
echo "提交 DWA 评估任务 (checkpoint-$DWA_CKPT)..."
sbatch --job-name=eval_DWA_${DWA_CKPT} \
    --export=QWEN_HOST="$DWA_HOST",QWEN_PORT="$DWA_PORT",ALGORITHM="DWA",NUM_PARAMS="$DWA_PARAMS",POLICY_NAME="$DWA_POLICY",START_WORLD="$START_WORLD",END_WORLD="$END_WORLD",RUNS_PER_WORLD="$RUNS_PER_WORLD" \
    executable/run_hopper_qwen.slurm

# TEB 评估
echo "提交 TEB 评估任务 (checkpoint-$TEB_CKPT)..."
sbatch --job-name=eval_TEB_${TEB_CKPT} \
    --export=QWEN_HOST="$TEB_HOST",QWEN_PORT="$TEB_PORT",ALGORITHM="TEB",NUM_PARAMS="$TEB_PARAMS",POLICY_NAME="$TEB_POLICY",START_WORLD="$START_WORLD",END_WORLD="$END_WORLD",RUNS_PER_WORLD="$RUNS_PER_WORLD" \
    executable/run_hopper_qwen.slurm

# MPPI 评估
echo "提交 MPPI 评估任务 (checkpoint-$MPPI_CKPT)..."
sbatch --job-name=eval_MPPI_${MPPI_CKPT} \
    --export=QWEN_HOST="$MPPI_HOST",QWEN_PORT="$MPPI_PORT",ALGORITHM="MPPI",NUM_PARAMS="$MPPI_PARAMS",POLICY_NAME="$MPPI_POLICY",START_WORLD="$START_WORLD",END_WORLD="$END_WORLD",RUNS_PER_WORLD="$RUNS_PER_WORLD" \
    executable/run_hopper_qwen.slurm

# DDP 评估
echo "提交 DDP 评估任务 (checkpoint-$DDP_CKPT)..."
sbatch --job-name=eval_DDP_${DDP_CKPT} \
    --export=QWEN_HOST="$DDP_HOST",QWEN_PORT="$DDP_PORT",ALGORITHM="DDP",NUM_PARAMS="$DDP_PARAMS",POLICY_NAME="$DDP_POLICY",START_WORLD="$START_WORLD",END_WORLD="$END_WORLD",RUNS_PER_WORLD="$RUNS_PER_WORLD" \
    executable/run_hopper_qwen.slurm

echo ""
echo "=================================================="
echo "✓ 所有评估任务已提交"
echo "=================================================="
echo ""
echo "查看任务状态："
echo "  squeue -u \$USER | grep eval_"
echo ""
echo "查看评估日志："
echo "  tail -f executable/cpu_report/eval_*.out"
echo ""
echo "预计运行时间："
echo "  每个评估任务约 2-4 小时（300 个环境）"
echo ""
