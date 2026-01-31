#!/bin/bash
# 批量提交 4 个 planner 的控制器（无硬编码路径）
#
# 用法：
#   bash submit_all_controllers.sh
#
# 自定义输出目录和资源（可选）：
#   OUTPUT_DIR="my_logs" bash submit_all_controllers.sh
#   OUTPUT_DIR="ctrl_logs" CTRL_CPUS=16 CTRL_MEM=8GB bash submit_all_controllers.sh
#
# 支持的环境变量：
#   - OUTPUT_DIR: 输出日志目录（默认：cpu_report1）
#   - CTRL_CPUS: 每个 controller 的 CPU 核数（默认：8）
#   - CTRL_MEM: 每个 controller 的总内存（默认：32GB，即 4GB*8核）
#   - CTRL_TIME: 最大运行时间（默认：5-00:00:00，即5天）
#   - CHECKPOINT_ORDER: 自定义 checkpoint 顺序（如 17500,15000,12500）
#
# 示例：只评估特定 checkpoint（按指定顺序）：
#   CHECKPOINT_ORDER="17500,15000,12500" bash submit_all_controllers.sh

set -euo pipefail  # 遇到错误立即退出，未定义变量报错

echo "=================================================="
echo "提交 4 个 Planner 的控制器"
echo "=================================================="

# 解析项目根目录（src/ros_jackal）为脚本所在目录的上一级
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
PROJECT_ROOT=$(cd "${SCRIPT_DIR}/.." && pwd)

# ============================================================
# 资源配置（可通过环境变量覆盖）
# ============================================================
OUTPUT_DIR="${OUTPUT_DIR:-cpu_report1}"  # 默认：cpu_report1
CTRL_CPUS="${CTRL_CPUS:-8}"              # 默认：8核
CTRL_MEM="${CTRL_MEM:-32GB}"             # 默认：32GB（4GB*8核）
CTRL_TIME="${CTRL_TIME:-5-00:00:00}"     # 默认：5天
CHECKPOINT_ORDER="${CHECKPOINT_ORDER:-}" # 自定义顺序，如 17500,15000,12500

LOG_DIR="${PROJECT_ROOT}/script_HPC/${OUTPUT_DIR}"
mkdir -p "$LOG_DIR"

echo ""
echo "资源配置:"
echo "  CPUs per task: $CTRL_CPUS"
echo "  Memory:        $CTRL_MEM"
echo "  Max time:      $CTRL_TIME"
echo "  Output dir:    $OUTPUT_DIR"
if [ -n "$CHECKPOINT_ORDER" ]; then
echo "  Checkpoint order: $CHECKPOINT_ORDER"
fi
echo ""
echo "Project root: $PROJECT_ROOT"
echo "Log dir:      $LOG_DIR"

# ============================================================
# 配置区域
# ============================================================

# Checkpoint 目录（去掉 /checkpoint-2500 后缀）
DWA_DIR="/scratch/bwang25/appvlm_ws/src/ros_jackal/model/dwa/qwen2.5-vl-regression_lora-True_dwa_regression_1"
TEB_DIR="/scratch/bwang25/appvlm_ws/src/ros_jackal/model/teb/qwen2.5-vl-regression_lora-True_teb_regression_1"
MPPI_DIR="/scratch/bwang25/appvlm_ws/src/ros_jackal/model/mppi/qwen2.5-vl-regression_lora-True_mppi_regression_1"
DDP_DIR="/scratch/bwang25/appvlm_ws/src/ros_jackal/model/ddp/qwen2.5-vl-regression_lora-True_ddp_regression_1"

# 端口配置（必须与 submit_all_planners.sh 一致）
# 与 submit_all_planners.sh 的端口分配保持一致：DWA=5001, TEB=5002, MPPI=5003, DDP=5004
DWA_PORT=5001
TEB_PORT=5002
MPPI_PORT=5003
DDP_PORT=5004

# GPU 节点配置（每个 planner 可以在不同节点）
# 👇 根据实际情况修改这些节点名
DWA_HOST="gpu019"   # 👈 修改为 DWA 服务的实际节点
TEB_HOST="gpu022"   # 👈 修改为 TEB 服务的实际节点
MPPI_HOST="gpu016"  # 👈 修改为 MPPI 服务的实际节点
DDP_HOST="gpu022"   # 👈 修改为 DDP 服务的实际节点

# 提示：可以从日志中查看实际节点
# grep "QWEN_HOST" cpu_report/qwen_*.out

echo ""
echo "配置信息："
echo "  DWA:  $DWA_HOST:$DWA_PORT"
echo "  TEB:  $TEB_HOST:$TEB_PORT"
echo "  MPPI: $MPPI_HOST:$MPPI_PORT"
echo "  DDP:  $DDP_HOST:$DDP_PORT"
echo ""

# ============================================================
# 提交控制器 (分配到不同节点)
# ============================================================

echo "提交控制器 (强制分配到不同CPU节点)..."
echo ""

# 用于记录已占用的节点
EXCLUDE_NODES=""

# DWA 控制器
echo "提交 DWA 控制器 ($DWA_HOST:$DWA_PORT)..."
JOB1=$(sbatch --job-name=ctrl_DWA \
    --chdir="$PROJECT_ROOT" \
    --cpus-per-task="$CTRL_CPUS" \
    --mem="$CTRL_MEM" \
    --time="$CTRL_TIME" \
    --output="${LOG_DIR}/controller_DWA-%j.out" \
    --error="${LOG_DIR}/controller_DWA-%j.err" \
    --export=CHECKPOINT_DIR="$DWA_DIR",QWEN_HOST="$DWA_HOST",QWEN_PORT="$DWA_PORT",WATCH_MODE="true",ALGORITHM="DWA",NUM_PARAMS="9",PROJECT_ROOT="$PROJECT_ROOT",CONDA_BASE="${CONDA_BASE:-}",CONDA_ENV="${CONDA_ENV:-}",CHECKPOINT_ORDER="${CHECKPOINT_ORDER:-}" \
    script_HPC/qwen_vlm_controller.slurm | awk '{print $4}')
echo "  Job ID: $JOB1"

# 等待节点分配
sleep 3
NODE1=$(squeue -j $JOB1 -h -o %N 2>/dev/null | grep -v "None" || echo "")
if [ -n "$NODE1" ]; then
    EXCLUDE_NODES="$NODE1"
    echo "  已分配节点: $NODE1 (后续作业将排除此节点)"
fi

# TEB 控制器 (排除 DWA 的节点)
echo "提交 TEB 控制器 ($TEB_HOST:$TEB_PORT)..."
if [ -n "$EXCLUDE_NODES" ]; then
    JOB2=$(sbatch --job-name=ctrl_TEB \
        --exclude="$EXCLUDE_NODES" \
        --chdir="$PROJECT_ROOT" \
        --cpus-per-task="$CTRL_CPUS" \
        --mem="$CTRL_MEM" \
        --time="$CTRL_TIME" \
        --output="${LOG_DIR}/controller_TEB-%j.out" \
        --error="${LOG_DIR}/controller_TEB-%j.err" \
        --export=CHECKPOINT_DIR="$TEB_DIR",QWEN_HOST="$TEB_HOST",QWEN_PORT="$TEB_PORT",WATCH_MODE="true",ALGORITHM="TEB",NUM_PARAMS="9",PROJECT_ROOT="$PROJECT_ROOT",CONDA_BASE="${CONDA_BASE:-}",CONDA_ENV="${CONDA_ENV:-}",CHECKPOINT_ORDER="${CHECKPOINT_ORDER:-}" \
        script_HPC/qwen_vlm_controller.slurm | awk '{print $4}')
else
    JOB2=$(sbatch --job-name=ctrl_TEB \
        --chdir="$PROJECT_ROOT" \
        --cpus-per-task="$CTRL_CPUS" \
        --mem="$CTRL_MEM" \
        --time="$CTRL_TIME" \
        --output="${LOG_DIR}/controller_TEB-%j.out" \
        --error="${LOG_DIR}/controller_TEB-%j.err" \
        --export=CHECKPOINT_DIR="$TEB_DIR",QWEN_HOST="$TEB_HOST",QWEN_PORT="$TEB_PORT",WATCH_MODE="true",ALGORITHM="TEB",NUM_PARAMS="9",PROJECT_ROOT="$PROJECT_ROOT",CONDA_BASE="${CONDA_BASE:-}",CONDA_ENV="${CONDA_ENV:-}",CHECKPOINT_ORDER="${CHECKPOINT_ORDER:-}" \
        script_HPC/qwen_vlm_controller.slurm | awk '{print $4}')
fi
echo "  Job ID: $JOB2"

sleep 3
NODE2=$(squeue -j $JOB2 -h -o %N 2>/dev/null | grep -v "None" || echo "")
if [ -n "$NODE2" ]; then
    EXCLUDE_NODES="${EXCLUDE_NODES:+$EXCLUDE_NODES,}$NODE2"
    echo "  已分配节点: $NODE2"
fi

# MPPI 控制器 (排除 DWA 和 TEB 的节点)
echo "提交 MPPI 控制器 ($MPPI_HOST:$MPPI_PORT)..."
if [ -n "$EXCLUDE_NODES" ]; then
    JOB3=$(sbatch --job-name=ctrl_MPPI \
        --exclude="$EXCLUDE_NODES" \
        --chdir="$PROJECT_ROOT" \
        --cpus-per-task="$CTRL_CPUS" \
        --mem="$CTRL_MEM" \
        --time="$CTRL_TIME" \
        --output="${LOG_DIR}/controller_MPPI-%j.out" \
        --error="${LOG_DIR}/controller_MPPI-%j.err" \
        --export=CHECKPOINT_DIR="$MPPI_DIR",QWEN_HOST="$MPPI_HOST",QWEN_PORT="$MPPI_PORT",WATCH_MODE="true",ALGORITHM="MPPI",NUM_PARAMS="10",PROJECT_ROOT="$PROJECT_ROOT",CONDA_BASE="${CONDA_BASE:-}",CONDA_ENV="${CONDA_ENV:-}",CHECKPOINT_ORDER="${CHECKPOINT_ORDER:-}" \
        script_HPC/qwen_vlm_controller.slurm | awk '{print $4}')
else
    JOB3=$(sbatch --job-name=ctrl_MPPI \
        --chdir="$PROJECT_ROOT" \
        --cpus-per-task="$CTRL_CPUS" \
        --mem="$CTRL_MEM" \
        --time="$CTRL_TIME" \
        --output="${LOG_DIR}/controller_MPPI-%j.out" \
        --error="${LOG_DIR}/controller_MPPI-%j.err" \
        --export=CHECKPOINT_DIR="$MPPI_DIR",QWEN_HOST="$MPPI_HOST",QWEN_PORT="$MPPI_PORT",WATCH_MODE="true",ALGORITHM="MPPI",NUM_PARAMS="10",PROJECT_ROOT="$PROJECT_ROOT",CONDA_BASE="${CONDA_BASE:-}",CONDA_ENV="${CONDA_ENV:-}",CHECKPOINT_ORDER="${CHECKPOINT_ORDER:-}" \
        script_HPC/qwen_vlm_controller.slurm | awk '{print $4}')
fi
echo "  Job ID: $JOB3"

sleep 3
NODE3=$(squeue -j $JOB3 -h -o %N 2>/dev/null | grep -v "None" || echo "")
if [ -n "$NODE3" ]; then
    EXCLUDE_NODES="${EXCLUDE_NODES:+$EXCLUDE_NODES,}$NODE3"
    echo "  已分配节点: $NODE3"
fi

# DDP 控制器 (排除所有已分配的节点)
echo "提交 DDP 控制器 ($DDP_HOST:$DDP_PORT)..."
if [ -n "$EXCLUDE_NODES" ]; then
    JOB4=$(sbatch --job-name=ctrl_DDP \
        --exclude="$EXCLUDE_NODES" \
        --chdir="$PROJECT_ROOT" \
        --cpus-per-task="$CTRL_CPUS" \
        --mem="$CTRL_MEM" \
        --time="$CTRL_TIME" \
        --output="${LOG_DIR}/controller_DDP-%j.out" \
        --error="${LOG_DIR}/controller_DDP-%j.err" \
        --export=CHECKPOINT_DIR="$DDP_DIR",QWEN_HOST="$DDP_HOST",QWEN_PORT="$DDP_PORT",WATCH_MODE="true",ALGORITHM="DDP",NUM_PARAMS="8",PROJECT_ROOT="$PROJECT_ROOT",CONDA_BASE="${CONDA_BASE:-}",CONDA_ENV="${CONDA_ENV:-}",CHECKPOINT_ORDER="${CHECKPOINT_ORDER:-}" \
        script_HPC/qwen_vlm_controller.slurm | awk '{print $4}')
else
    JOB4=$(sbatch --job-name=ctrl_DDP \
        --chdir="$PROJECT_ROOT" \
        --cpus-per-task="$CTRL_CPUS" \
        --mem="$CTRL_MEM" \
        --time="$CTRL_TIME" \
        --output="${LOG_DIR}/controller_DDP-%j.out" \
        --error="${LOG_DIR}/controller_DDP-%j.err" \
        --export=CHECKPOINT_DIR="$DDP_DIR",QWEN_HOST="$DDP_HOST",QWEN_PORT="$DDP_PORT",WATCH_MODE="true",ALGORITHM="DDP",NUM_PARAMS="8",PROJECT_ROOT="$PROJECT_ROOT",CONDA_BASE="${CONDA_BASE:-}",CONDA_ENV="${CONDA_ENV:-}",CHECKPOINT_ORDER="${CHECKPOINT_ORDER:-}" \
        script_HPC/qwen_vlm_controller.slurm | awk '{print $4}')
fi
echo "  Job ID: $JOB4"

sleep 3
NODE4=$(squeue -j $JOB4 -h -o %N 2>/dev/null | grep -v "None" || echo "")
if [ -n "$NODE4" ]; then
    echo "  已分配节点: $NODE4"
fi

echo ""
echo "=================================================="
echo "✓ 所有控制器已提交"
echo "=================================================="
echo ""
echo "节点分配情况："
if [ -n "$NODE1" ]; then echo "  DWA:  $NODE1"; else echo "  DWA:  (pending)"; fi
if [ -n "$NODE2" ]; then echo "  TEB:  $NODE2"; else echo "  TEB:  (pending)"; fi
if [ -n "$NODE3" ]; then echo "  MPPI: $NODE3"; else echo "  MPPI: (pending)"; fi
if [ -n "$NODE4" ]; then echo "  DDP:  $NODE4"; else echo "  DDP:  (pending)"; fi
echo ""
echo "查看任务状态："
echo "  squeue -u \$USER"
echo ""
echo "查看节点分配 (等待1-2分钟后)："
echo "  squeue -u \$USER -o '%.18i %.9P %.30j %.8u %.8T %.10M %.6D %R'"
echo ""
echo "查看控制器日志："
echo "  tail -f ${LOG_DIR}/controller_DWA-${JOB1}.out"
echo "  tail -f ${LOG_DIR}/controller_TEB-${JOB2}.out"
echo "  tail -f ${LOG_DIR}/controller_MPPI-${JOB3}.out"
echo "  tail -f ${LOG_DIR}/controller_DDP-${JOB4}.out"
echo ""
echo "查看所有控制器日志（通配符）："
echo "  tail -f ${LOG_DIR}/controller_*.out"
echo ""
echo "查看评估任务："
echo "  squeue -u \$USER | grep qwen_robot_test"
echo ""
echo "取消所有控制器："
echo "  scancel $JOB1 $JOB2 $JOB3 $JOB4"
