#!/bin/bash
# 测试所有 4 个 Qwen 服务是否正常工作

set -e

echo "=========================================="
echo "Testing All Qwen Services"
echo "=========================================="
echo ""

# 切换到项目目录
cd /scratch/ylu22/appvlm_ws/src/ros_jackal

# 查找测试图片
TEST_IMAGE=$(find buffer/*/actor_*/VLM_*.png 2>/dev/null | head -1)

if [ -z "$TEST_IMAGE" ]; then
    echo "❌ No test image found in buffer/"
    echo "Please provide a test image path as argument:"
    echo "  $0 /path/to/test_image.png"
    exit 1
fi

# 如果提供了参数，使用参数作为测试图片
if [ $# -gt 0 ]; then
    TEST_IMAGE=$1
fi

echo "Using test image: $TEST_IMAGE"
echo ""

# 服务配置（与 submit_all_planners.sh 一致）
declare -A SERVICES=(
    ["DWA"]="gpu019:5001"
    ["TEB"]="gpu019:5002"
    ["MPPI"]="gpu020:5003"
    ["DDP"]="gpu021:5004"
)

# 测试结果统计
TOTAL=0
SUCCESS=0
FAILED=0

# 测试每个服务
for PLANNER in DWA TEB MPPI DDP; do
    TOTAL=$((TOTAL + 1))
    IFS=':' read -r HOST PORT <<< "${SERVICES[$PLANNER]}"

    echo "=========================================="
    echo "Testing $PLANNER ($HOST:$PORT)"
    echo "=========================================="

    # 先检查服务是否可达
    if ! curl -s --connect-timeout 5 http://${HOST}:${PORT}/health > /dev/null 2>&1; then
        echo "❌ Service unreachable: http://${HOST}:${PORT}"
        FAILED=$((FAILED + 1))
        echo ""
        continue
    fi

    echo "✓ Service is reachable"

    # 运行推理测试
    QWEN_HOST=$HOST QWEN_PORT=$PORT ./script_HPC/simple_infer.sh "$TEST_IMAGE" "$PLANNER"

    if [ $? -eq 0 ]; then
        SUCCESS=$((SUCCESS + 1))
        echo "✓ $PLANNER inference successful"
    else
        FAILED=$((FAILED + 1))
        echo "❌ $PLANNER inference failed"
    fi

    echo ""
done

# 总结
echo "=========================================="
echo "Test Summary"
echo "=========================================="
echo "Total:   $TOTAL"
echo "Success: $SUCCESS"
echo "Failed:  $FAILED"
echo "=========================================="

if [ $FAILED -eq 0 ]; then
    echo "✓ All services are working!"
    exit 0
else
    echo "⚠ Some services failed. Check the logs above."
    exit 1
fi
