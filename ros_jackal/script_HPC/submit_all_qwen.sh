#!/bin/bash
# 批量提交 4 个 planner 的 GPU 服务
#
# 用法：
#   bash submit_all_qwen.sh
#
# 自定义 GPU 资源和输出目录（可选）：
#   GPU_GRES="gpu:A100:1" OUTPUT_DIR="my_logs" bash submit_all_qwen.sh
#   GPU_GRES="gpu:3g.40gb:1" bash submit_all_qwen.sh
#
# 支持的 GPU GRES 示例：
#   - gpu:3g.40gb:1  (默认，MIG 3g 分片)
#   - gpu:A100:1     (整卡 A100)
#   - gpu:1          (任意1块GPU)

set -e  # 遇到错误立即退出

echo "=================================================="
echo "提交 4 个 Planner 的 GPU 服务"
echo "=================================================="

# ============================================================
# GPU 资源配置（可通过环境变量覆盖）
# ============================================================
GPU_GRES="${GPU_GRES:-gpu:3g.40gb:1}"  # 默认：3g.40gb 分片
OUTPUT_DIR="${OUTPUT_DIR:-cpu_report}"  # 默认：cpu_report

echo ""
echo "GPU 资源配置:"
echo "  GPU GRES: $GPU_GRES"
echo "  Output Dir: $OUTPUT_DIR"
echo ""

# 创建日志目录
mkdir -p "$OUTPUT_DIR"

# ============================================================
# 配置区域
# ============================================================

# 定义路径
DWA_PATH="/scratch/bwang25/appvlm_ws/src/ros_jackal/model/dwa/qwen2.5-vl-regression_lora-True_dwa_regression_1/checkpoint-17500"
TEB_PATH="/scratch/bwang25/appvlm_ws/src/ros_jackal/model/teb/qwen2.5-vl-regression_lora-True_teb_regression_1/checkpoint-17500"
MPPI_PATH="/scratch/bwang25/appvlm_ws/src/ros_jackal/model/mppi/qwen2.5-vl-regression_lora-True_mppi_regression_1/checkpoint-17500"
DDP_PATH="/scratch/bwang25/appvlm_ws/src/ros_jackal/model/ddp/qwen2.5-vl-regression_lora-True_ddp_regression_1/checkpoint-17500"

# 定义端口（避免多个服务在同一节点时冲突）
DWA_PORT=5001
TEB_PORT=5002
MPPI_PORT=5003
DDP_PORT=5004

# 定义参数数量
DWA_PARAMS=9
TEB_PARAMS=9
MPPI_PARAMS=10
DDP_PARAMS=8

# ============================================================
# 验证路径
# ============================================================

echo ""
echo "验证路径..."
for path in "$DWA_PATH" "$TEB_PATH" "$MPPI_PATH" "$DDP_PATH"; do
    if [ ! -d "$path" ]; then
        echo "❌ 路径不存在: $path"
        exit 1
    fi
done
echo "✓ 所有路径验证通过"

# ============================================================
# 提交 GPU 服务
# ============================================================

echo ""
echo "提交 GPU 服务..."
echo ""

# 提交 DWA
echo "提交 DWA (端口 $DWA_PORT)..."
DWA_OUTPUT=$(sbatch --job-name=qwen_DWA \
    --gres="$GPU_GRES" \
    --output="${OUTPUT_DIR}/qwen_DWA-%j.out" \
    --error="${OUTPUT_DIR}/qwen_DWA-%j.err" \
    --export=INITIAL_CHECKPOINT="$DWA_PATH",ALGORITHM="DWA",NUM_PARAMS="$DWA_PARAMS",PORT="$DWA_PORT" \
    script_HPC/qwen_vlm_server.slurm)
DWA_JOB=$(echo $DWA_OUTPUT | grep -oP '\d+')
echo "  ✓ Job ID: $DWA_JOB"

# 提交 TEB
echo "提交 TEB (端口 $TEB_PORT)..."
TEB_OUTPUT=$(sbatch --job-name=qwen_TEB \
    --gres="$GPU_GRES" \
    --output="${OUTPUT_DIR}/qwen_TEB-%j.out" \
    --error="${OUTPUT_DIR}/qwen_TEB-%j.err" \
    --export=INITIAL_CHECKPOINT="$TEB_PATH",ALGORITHM="TEB",NUM_PARAMS="$TEB_PARAMS",PORT="$TEB_PORT" \
    script_HPC/qwen_vlm_server.slurm)
TEB_JOB=$(echo $TEB_OUTPUT | grep -oP '\d+')
echo "  ✓ Job ID: $TEB_JOB"

# 提交 MPPI
echo "提交 MPPI (端口 $MPPI_PORT)..."
MPPI_OUTPUT=$(sbatch --job-name=qwen_MPPI \
    --gres="$GPU_GRES" \
    --output="${OUTPUT_DIR}/qwen_MPPI-%j.out" \
    --error="${OUTPUT_DIR}/qwen_MPPI-%j.err" \
    --export=INITIAL_CHECKPOINT="$MPPI_PATH",ALGORITHM="MPPI",NUM_PARAMS="$MPPI_PARAMS",PORT="$MPPI_PORT" \
    script_HPC/qwen_vlm_server.slurm)
MPPI_JOB=$(echo $MPPI_OUTPUT | grep -oP '\d+')
echo "  ✓ Job ID: $MPPI_JOB"

# 提交 DDP
echo "提交 DDP (端口 $DDP_PORT)..."
DDP_OUTPUT=$(sbatch --job-name=qwen_DDP \
    --gres="$GPU_GRES" \
    --output="${OUTPUT_DIR}/qwen_DDP-%j.out" \
    --error="${OUTPUT_DIR}/qwen_DDP-%j.err" \
    --export=INITIAL_CHECKPOINT="$DDP_PATH",ALGORITHM="DDP",NUM_PARAMS="$DDP_PARAMS",PORT="$DDP_PORT" \
    script_HPC/qwen_vlm_server.slurm)
DDP_JOB=$(echo $DDP_OUTPUT | grep -oP '\d+')
echo "  ✓ Job ID: $DDP_JOB"

# ============================================================
# 查询节点分配
# ============================================================

echo ""
echo "正在查询节点分配..."
sleep 2  # 等待任务进入队列

# 查询每个任务的节点和状态
DWA_NODE=$(squeue -j $DWA_JOB -h -o "%N" 2>/dev/null || echo "")
DWA_STATE=$(squeue -j $DWA_JOB -h -o "%T" 2>/dev/null || echo "")
TEB_NODE=$(squeue -j $TEB_JOB -h -o "%N" 2>/dev/null || echo "")
TEB_STATE=$(squeue -j $TEB_JOB -h -o "%T" 2>/dev/null || echo "")
MPPI_NODE=$(squeue -j $MPPI_JOB -h -o "%N" 2>/dev/null || echo "")
MPPI_STATE=$(squeue -j $MPPI_JOB -h -o "%T" 2>/dev/null || echo "")
DDP_NODE=$(squeue -j $DDP_JOB -h -o "%N" 2>/dev/null || echo "")
DDP_STATE=$(squeue -j $DDP_JOB -h -o "%T" 2>/dev/null || echo "")

# 处理节点显示（None 表示待分配）
[ -z "$DWA_NODE" ] || [ "$DWA_NODE" == "(None)" ] && DWA_NODE="⏳待分配" || DWA_NODE="✓$DWA_NODE"
[ -z "$TEB_NODE" ] || [ "$TEB_NODE" == "(None)" ] && TEB_NODE="⏳待分配" || TEB_NODE="✓$TEB_NODE"
[ -z "$MPPI_NODE" ] || [ "$MPPI_NODE" == "(None)" ] && MPPI_NODE="⏳待分配" || MPPI_NODE="✓$MPPI_NODE"
[ -z "$DDP_NODE" ] || [ "$DDP_NODE" == "(None)" ] && DDP_NODE="⏳待分配" || DDP_NODE="✓$DDP_NODE"

echo ""
echo "=================================================="
echo "所有 GPU 服务已提交"
echo "=================================================="
echo ""
printf "%-10s %-12s %-8s %-20s %-12s\n" "Planner" "Job ID" "端口" "节点" "状态"
echo "------------------------------------------------------------------------"
printf "%-10s %-12s %-8s %-20s %-12s\n" "DWA" "$DWA_JOB" "$DWA_PORT" "$DWA_NODE" "$DWA_STATE"
printf "%-10s %-12s %-8s %-20s %-12s\n" "TEB" "$TEB_JOB" "$TEB_PORT" "$TEB_NODE" "$TEB_STATE"
printf "%-10s %-12s %-8s %-20s %-12s\n" "MPPI" "$MPPI_JOB" "$MPPI_PORT" "$MPPI_NODE" "$MPPI_STATE"
printf "%-10s %-12s %-8s %-20s %-12s\n" "DDP" "$DDP_JOB" "$DDP_PORT" "$DDP_NODE" "$DDP_STATE"

# 统计分配状态
ALLOCATED=0
PENDING=0
[ "$DWA_NODE" != "⏳待分配" ] && ALLOCATED=$((ALLOCATED+1)) || PENDING=$((PENDING+1))
[ "$TEB_NODE" != "⏳待分配" ] && ALLOCATED=$((ALLOCATED+1)) || PENDING=$((PENDING+1))
[ "$MPPI_NODE" != "⏳待分配" ] && ALLOCATED=$((ALLOCATED+1)) || PENDING=$((PENDING+1))
[ "$DDP_NODE" != "⏳待分配" ] && ALLOCATED=$((ALLOCATED+1)) || PENDING=$((PENDING+1))

echo ""
echo "分配状态总结："
if [ $ALLOCATED -eq 4 ]; then
    echo "  🎉 所有 4 个服务已成功分配到节点！"
elif [ $ALLOCATED -gt 0 ]; then
    echo "  ✓ $ALLOCATED/4 个服务已分配，$PENDING/4 个等待中"
    echo "  ⏳ 部分任务仍在队列中，通常需要等待 1-5 分钟"
else
    echo "  ⚠️  所有任务都在等待资源分配"
    echo ""
    echo "可能原因："
    echo "  - GPU 资源不足（所有 $GPU_GRES 都在使用中）"
    echo "  - 等待其他任务释放 GPU"
    echo ""
    echo "建议操作："
    echo "  squeue -p gpuq  # 查看 GPU 队列"
    echo "  sinfo -p gpuq   # 查看 GPU 可用性"
fi
echo ""
echo "查看任务状态："
echo "  squeue -u \$USER"
echo "  squeue -j $DWA_JOB,$TEB_JOB,$MPPI_JOB,$DDP_JOB"
echo ""
echo "查看节点分配（等待1-2分钟后）："
echo "  grep 'QWEN_HOST\\|Port:' ${OUTPUT_DIR}/qwen_*.out"
echo ""
echo "查看日志："
echo "  tail -f ${OUTPUT_DIR}/qwen_DWA-${DWA_JOB}.out"
echo "  tail -f ${OUTPUT_DIR}/qwen_TEB-${TEB_JOB}.out"
echo "  tail -f ${OUTPUT_DIR}/qwen_MPPI-${MPPI_JOB}.out"
echo "  tail -f ${OUTPUT_DIR}/qwen_DDP-${DDP_JOB}.out"
echo ""
echo "取消所有任务："
echo "  scancel $DWA_JOB $TEB_JOB $MPPI_JOB $DDP_JOB"
echo ""
echo "⏳ 等待所有服务启动后，运行控制器脚本："
echo "  bash script_HPC/submit_all_controllers.sh"

