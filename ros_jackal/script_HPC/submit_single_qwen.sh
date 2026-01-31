#!/bin/bash
# 测试单个 planner 的 GPU 服务

set -e  # 遇到错误立即退出

# ============================================================
# 参数解析
# ============================================================

if [ $# -eq 0 ]; then
    echo "用法: $0 <planner> [GPU_GRES] [OUTPUT_DIR]"
    echo "  planner: DWA | TEB | MPPI | DDP"
    echo "  GPU_GRES: GPU 资源规格（可选，默认：gpu:3g.40gb:1）"
    echo "  OUTPUT_DIR: 输出日志目录（可选，默认：cpu_report）"
    echo ""
    echo "示例:"
    echo "  $0 DWA"
    echo "  $0 DWA gpu:3g.40gb:1"
    echo "  $0 DWA gpu:A100:1 my_logs"
    exit 1
fi

PLANNER=$1
GPU_GRES="${2:-${GPU_GRES:-gpu:3g.40gb:1}}"  # 第二个参数或环境变量，默认 3g.40gb
OUTPUT_DIR="${3:-${OUTPUT_DIR:-cpu_report}}"  # 第三个参数或环境变量，默认 cpu_report

# 转换为大写
PLANNER=$(echo "$PLANNER" | tr '[:lower:]' '[:upper:]')

echo "=================================================="
echo "提交 $PLANNER Planner 的 GPU 服务"
echo "=================================================="

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

# 基础路径
BASE_PATH="/scratch/bwang25/appvlm_ws/src/ros_jackal/model"

# 根据 planner 设置路径、端口和参数数量
case $PLANNER in
    DWA)
        MODEL_PATH="$BASE_PATH/dwa/qwen2.5-vl-regression_lora-True_dwa_regression_1/checkpoint-17500"
        PORT=5001
        NUM_PARAMS=7
        ;;
    TEB)
        MODEL_PATH="$BASE_PATH/teb/qwen2.5-vl-regression_lora-True_teb_regression_1/checkpoint-17500"
        PORT=5002
        NUM_PARAMS=7
        ;;
    MPPI)
        MODEL_PATH="$BASE_PATH/mppi/qwen2.5-vl-regression_lora-True_mppi_regression_1/checkpoint-17500"
        PORT=5003
        NUM_PARAMS=8
        ;;
    DDP)
        MODEL_PATH="$BASE_PATH/ddp/qwen2.5-vl-regression_lora-True_ddp_regression_1/checkpoint-17500"
        PORT=5004
        NUM_PARAMS=6
        ;;
    *)
        echo "❌ 不支持的 planner: $PLANNER"
        echo "   支持的 planner: DWA, TEB, MPPI, DDP"
        exit 1
        ;;
esac

# ============================================================
# 验证路径
# ============================================================

echo ""
echo "验证路径..."
if [ ! -d "$MODEL_PATH" ]; then
    echo "❌ 路径不存在: $MODEL_PATH"
    exit 1
fi
echo "✓ 路径验证通过: $MODEL_PATH"

# ============================================================
# 提交 GPU 服务
# ============================================================

echo ""
echo "提交 GPU 服务..."
echo ""

echo "提交 $PLANNER (端口 $PORT)..."
JOB_OUTPUT=$(sbatch --job-name=qwen_$PLANNER \
    --gres="$GPU_GRES" \
    --output="${OUTPUT_DIR}/qwen_${PLANNER}-%j.out" \
    --error="${OUTPUT_DIR}/qwen_${PLANNER}-%j.err" \
    --export=INITIAL_CHECKPOINT="$MODEL_PATH",ALGORITHM="$PLANNER",NUM_PARAMS="$NUM_PARAMS",PORT="$PORT" \
    script_HPC/qwen_vlm_server.slurm)

# 提取 job ID
JOB_ID=$(echo $JOB_OUTPUT | grep -oP '\d+')

echo "✓ 任务已提交，Job ID: $JOB_ID"
echo ""

# 等待任务被调度到节点（最多等待30秒）
echo "正在查询节点分配..."
for i in {1..30}; do
    NODE=$(squeue -j $JOB_ID -h -o "%N" 2>/dev/null)
    JOB_STATE=$(squeue -j $JOB_ID -h -o "%T" 2>/dev/null)
    if [ ! -z "$NODE" ] && [ "$NODE" != "(None)" ]; then
        break
    fi
    sleep 1
done

echo ""
echo "=================================================="
echo "GPU 服务信息"
echo "=================================================="
echo ""
echo "Planner:  $PLANNER"
echo "Job ID:   $JOB_ID"
echo "端口:     $PORT"
echo "参数数:   $NUM_PARAMS"
echo "GPU GRES: $GPU_GRES"

# 检查节点分配状态并给出明确提示
if [ ! -z "$NODE" ] && [ "$NODE" != "(None)" ]; then
    echo "节点:     $NODE"
    echo "状态:     ✓ 已分配到节点 ($JOB_STATE)"
    echo ""
    echo "🎉 GPU 服务已成功分配！"
else
    echo "节点:     ⏳ 待分配"
    if [ "$JOB_STATE" == "PENDING" ]; then
        echo "状态:     等待资源中 (PENDING)"
        echo ""
        echo "⚠️  任务在队列中等待，可能原因："
        echo "   - GPU 资源不足（所有 $GPU_GRES 都在使用中）"
        echo "   - 优先级较低"
        echo "   - 等待其他任务释放资源"
        echo ""
        echo "建议操作："
        echo "  1. 查看队列情况: squeue -p gpuq"
        echo "  2. 查看任务详情: scontrol show job $JOB_ID"
        echo "  3. 等待几分钟后再次检查"
    elif [ -z "$JOB_STATE" ]; then
        echo "状态:     ❌ 任务不在队列中（可能失败）"
        echo ""
        echo "请检查错误日志："
        echo "  cat ${OUTPUT_DIR}/qwen_${PLANNER}-${JOB_ID}.err"
    else
        echo "状态:     $JOB_STATE"
    fi
fi
echo ""
echo "模型路径: $MODEL_PATH"
echo ""
echo "查看任务状态："
echo "  squeue -j $JOB_ID"
echo "  scontrol show job $JOB_ID"
echo ""
echo "查看日志："
echo "  tail -f ${OUTPUT_DIR}/qwen_${PLANNER}-${JOB_ID}.out"
echo "  tail -f ${OUTPUT_DIR}/qwen_${PLANNER}-${JOB_ID}.err"
echo ""
echo "查看节点分配（等待1-2分钟后）："
echo "  grep 'QWEN_HOST\\|Port:' ${OUTPUT_DIR}/qwen_${PLANNER}-${JOB_ID}.out"
echo ""
echo "取消任务："
echo "  scancel $JOB_ID"
echo ""
echo "⏳ 等待服务启动后，可以运行对应的控制器脚本"

