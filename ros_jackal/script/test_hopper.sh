#!/bin/bash
# Hopper 版本的测试脚本 - 连接到远程 GPU 节点上的 Qwen 服务

# ============================================================
# 配置区
# ============================================================

# Qwen 服务节点 (从环境变量读取，或手动设置)
export QWEN_HOST=${QWEN_HOST:-gpu017}  # 👈 修改为你的实际 GPU 节点
export QWEN_PORT=${QWEN_PORT:-5000}

echo "========================================="
echo "Hopper Qwen Evaluation Test"
echo "========================================="
echo "Qwen Service: http://${QWEN_HOST}:${QWEN_PORT}"
echo "========================================="
echo ""

# 检查 Qwen 服务是否可达
echo "Checking Qwen service..."
if ! curl -s --connect-timeout 5 http://${QWEN_HOST}:${QWEN_PORT}/health > /dev/null 2>&1; then
    echo "❌ Error: Cannot reach Qwen service at http://${QWEN_HOST}:${QWEN_PORT}"
    echo ""
    echo "Please check:"
    echo "  1. Is the service running?"
    echo "     squeue -u \$USER"
    echo "  2. Is QWEN_HOST correct?"
    echo "     export QWEN_HOST=gpu017  # your actual node"
    echo "  3. Can you reach it from this node?"
    echo "     curl http://\${QWEN_HOST}:5000/health"
    exit 1
fi

echo "✓ Qwen service is reachable"
echo ""

# ============================================================
# 清理进程
# ============================================================

killall -9 rosmaster 2>/dev/null
killall gzclient 2>/dev/null
killall gzserver 2>/dev/null

sleep 2

# ============================================================
# 运行测试
# ============================================================

# 测试世界范围 (可以根据需要修改)
START_WORLD=0
END_WORLD=10
RUNS_PER_WORLD=1

echo "Testing worlds ${START_WORLD} to ${END_WORLD} (${RUNS_PER_WORLD} runs each)"
echo ""

for i in $(seq $START_WORLD $END_WORLD) ; do
    for j in $(seq 1 $RUNS_PER_WORLD) ; do
        echo "========================================="
        echo "World: $i, Run: $j/$RUNS_PER_WORLD"
        echo "========================================="

        # 🔧 使用 Hopper 版本的评估脚本
        python evaluate_qwen_hopper.py \
            --world_idx $i \
            --qwen_host $QWEN_HOST \
            --qwen_port $QWEN_PORT \
            --policy_name ddp_qwen

        sleep 4
    done
done

echo ""
echo "========================================="
echo "All tests completed!"
echo "========================================="
