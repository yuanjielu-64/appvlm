#!/bin/bash
# 检查 Qwen 服务状态

QWEN_HOST=${QWEN_HOST:-localhost}
QWEN_PORT=${QWEN_PORT:-5000}
QWEN_URL="http://${QWEN_HOST}:${QWEN_PORT}"

echo "========================================"
echo "Qwen Service Status Check"
echo "========================================"
echo "Service URL: $QWEN_URL"
echo ""

# 检查服务是否可达
echo "1. Checking service connectivity..."
if curl -s --connect-timeout 5 ${QWEN_URL}/health > /dev/null 2>&1; then
    echo "✓ Service is reachable"
else
    echo "❌ Cannot reach service at $QWEN_URL"
    echo ""
    echo "Troubleshooting:"
    echo "  1. Check if service is running:"
    echo "     squeue -u \$USER"
    echo "  2. Check service logs:"
    echo "     tail -f qwen_dynamic_*.out"
    echo "  3. Verify QWEN_HOST is correct:"
    echo "     export QWEN_HOST=gpu017  # or your actual GPU node"
    exit 1
fi

# 获取健康状态
echo ""
echo "2. Service health status:"
curl -s ${QWEN_URL}/health | python3 -c "
import sys, json
try:
    data = json.load(sys.stdin)
    print(f\"  Status: {data['status']}\")
    print(f\"  Current checkpoint: {data['current_checkpoint']}\")
    print(f\"  Algorithm: {data['algorithm']}\")
    print(f\"  Num params: {data['num_params']}\")
    print(f\"  Device: {data['device']}\")
except Exception as e:
    print(f\"  Error parsing response: {e}\")
"

# 列出可用的 checkpoints
echo ""
echo "3. Available checkpoints:"
curl -s ${QWEN_URL}/list_checkpoints | python3 -c "
import sys, json
try:
    data = json.load(sys.stdin)
    checkpoints = data['available_checkpoints']

    # 按规划器分组
    planners = {}
    for ckpt in checkpoints:
        planner = ckpt['planner']
        if planner not in planners:
            planners[planner] = []
        planners[planner].append(ckpt)

    for planner, ckpts in sorted(planners.items()):
        print(f\"  {planner.upper()}: {len(ckpts)} checkpoints\")
        for ckpt in sorted(ckpts, key=lambda x: int(x['checkpoint_num']))[:3]:
            print(f\"    - checkpoint-{ckpt['checkpoint_num']}\")
        if len(ckpts) > 3:
            print(f\"    ... and {len(ckpts)-3} more\")

    print(f\"\\n  Total: {len(checkpoints)} checkpoints\")
except Exception as e:
    print(f\"  Error: {e}\")
"

echo ""
echo "========================================"
echo "Service is ready for use!"
echo "========================================"
echo ""
echo "Quick commands:"
echo "  # Switch checkpoint:"
echo "    bash switch_checkpoint.sh ddp 12500"
echo ""
echo "  # Run full test:"
echo "    bash run_test_on_hopper.sh"
echo ""
