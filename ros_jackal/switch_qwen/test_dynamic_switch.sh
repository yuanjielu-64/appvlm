#!/bin/bash
# 测试动态切换 checkpoint 功能

QWEN_URL="http://localhost:5000"

echo "=================================================="
echo "Testing Dynamic Checkpoint Switching"
echo "=================================================="

# 1. 检查服务健康状态
echo ""
echo "Step 1: Check service health..."
curl -s ${QWEN_URL}/health | python3 -m json.tool

# 2. 列出可用的 checkpoints
echo ""
echo "Step 2: List available checkpoints..."
curl -s ${QWEN_URL}/list_checkpoints | python3 -m json.tool

# 3. 切换到 checkpoint-10000
echo ""
echo "Step 3: Switch to checkpoint-10000..."
curl -s -X POST ${QWEN_URL}/switch_checkpoint \
    -H "Content-Type: application/json" \
    -d '{
        "checkpoint_path": "ddp/checkpoint-12500",
        "algorithm": "DDP",
        "head_type": "dpt",
        "num_params": 6
    }' | python3 -m json.tool

# 4. 检查切换后的状态
echo ""
echo "Step 4: Check health after switch..."
curl -s ${QWEN_URL}/health | python3 -m json.tool


echo ""
echo "=================================================="
echo "Test completed!"
echo "=================================================="
