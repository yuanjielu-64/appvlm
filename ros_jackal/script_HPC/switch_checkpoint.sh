#!/bin/bash
# 快速切换 checkpoint 的脚本

QWEN_HOST=${QWEN_HOST:-localhost}
QWEN_PORT=${QWEN_PORT:-5000}

if [ $# -lt 2 ]; then
    echo "Usage: $0 <planner> <checkpoint_number> [num_params]"
    echo ""
    echo "Examples:"
    echo "  $0 ddp 10000         # 切换到 DDP checkpoint-10000"
    echo "  $0 dwa 12500 7       # 切换到 DWA checkpoint-12500 (7个参数)"
    echo "  $0 teb 5000          # 切换到 TEB checkpoint-5000"
    echo ""
    echo "Environment variables:"
    echo "  QWEN_HOST=gpu017     # Qwen 服务所在节点"
    echo "  QWEN_PORT=5000       # 服务端口"
    exit 1
fi

PLANNER=$1
CHECKPOINT_NUM=$2
NUM_PARAMS=${3:-6}  # 默认 6 个参数

# 构建完整路径
CHECKPOINT_PATH="${PLANNER}/qwen2.5-vl-regression_lora-True_${PLANNER}_regression/checkpoint-${CHECKPOINT_NUM}"

echo "Switching to: $CHECKPOINT_PATH"
echo "Service: http://${QWEN_HOST}:${QWEN_PORT}"

curl -X POST http://${QWEN_HOST}:${QWEN_PORT}/switch_checkpoint \
  -H "Content-Type: application/json" \
  -d "{
    \"checkpoint_path\": \"${CHECKPOINT_PATH}\",
    \"algorithm\": \"${PLANNER^^}\",
    \"head_type\": \"dpt\",
    \"num_params\": ${NUM_PARAMS}
  }" | python3 -m json.tool

echo ""
echo "Current status:"
curl -s http://${QWEN_HOST}:${QWEN_PORT}/health | python3 -m json.tool
