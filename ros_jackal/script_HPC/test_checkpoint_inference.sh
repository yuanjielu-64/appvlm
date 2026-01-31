#!/bin/bash
# 测试动态切换 checkpoint 并进行推理

QWEN_HOST=${QWEN_HOST:-localhost}
QWEN_PORT=${QWEN_PORT:-5000}
QWEN_URL="http://${QWEN_HOST}:${QWEN_PORT}"

# 测试图片路径（你需要提供一个真实的 costmap 图片）
TEST_IMAGE=${TEST_IMAGE:-"actor_01/VLM_000000.png"}

echo "=================================================="
echo "Testing Checkpoint Switching & Inference"
echo "=================================================="
echo "Qwen Service: $QWEN_URL"
echo "Test Image: $TEST_IMAGE"
echo "=================================================="

# 检查图片是否存在
if [ ! -f "$TEST_IMAGE" ]; then
    echo "❌ Error: Test image not found: $TEST_IMAGE"
    echo "Please set TEST_IMAGE environment variable or provide a valid path"
    echo ""
    echo "Example:"
    echo "  export TEST_IMAGE=/path/to/your/costmap.png"
    echo "  bash test_checkpoint_inference.sh"
    exit 1
fi

# 检查服务是否运行
echo ""
echo "Step 1: Checking service health..."
if ! curl -s --connect-timeout 5 ${QWEN_URL}/health > /dev/null 2>&1; then
    echo "❌ Error: Cannot reach Qwen service at $QWEN_URL"
    echo "Please make sure the service is running:"
    echo "  bash script_HPC/submit_single_qwen.sh DWA"
    exit 1
fi
echo "✓ Service is running"

# 列出可用的 checkpoints
echo ""
echo "Step 2: Available checkpoints..."
curl -s ${QWEN_URL}/list_checkpoints | python3 -c "
import sys, json
data = json.load(sys.stdin)
print(f\"Found {len(data['available_checkpoints'])} checkpoints:\")
for ckpt in data['available_checkpoints'][:5]:
    print(f\"  - {ckpt['path']}\")
if len(data['available_checkpoints']) > 5:
    print(f\"  ... and {len(data['available_checkpoints'])-5} more\")
"

# 测试函数
test_checkpoint() {
    local planner=$1
    local checkpoint_num=$2
    local num_params=${3:-6}

    echo ""
    echo "=================================================="
    echo "Testing: ${planner^^} checkpoint-${checkpoint_num}"
    echo "=================================================="

    # 切换 checkpoint
    local checkpoint_path="${planner}/qwen2.5-vl-regression_lora-True_${planner}_regression/checkpoint-${checkpoint_num}"

    echo "Switching to: $checkpoint_path"
    switch_result=$(curl -s -X POST ${QWEN_URL}/switch_checkpoint \
        -H "Content-Type: application/json" \
        -d "{
            \"checkpoint_path\": \"${checkpoint_path}\",
            \"algorithm\": \"${planner^^}\",
            \"head_type\": \"dpt\",
            \"num_params\": ${num_params}
        }")

    success=$(echo $switch_result | python3 -c "import sys, json; print(json.load(sys.stdin).get('success', False))")

    if [ "$success" != "True" ]; then
        echo "❌ Failed to switch checkpoint"
        echo $switch_result | python3 -m json.tool
        return 1
    fi

    switch_time=$(echo $switch_result | python3 -c "import sys, json; print(json.load(sys.stdin).get('switch_time', 0))")
    echo "✓ Switched in ${switch_time}s"

    # 进行推理
    echo ""
    echo "Running inference..."

    # 将图片转换为 base64
    image_base64=$(base64 -w 0 $TEST_IMAGE)

    infer_result=$(curl -s -X POST ${QWEN_URL}/infer \
        -H "Content-Type: application/json" \
        -d "{
            \"image_base64\": \"${image_base64}\",
            \"linear_vel\": 0.5,
            \"angular_vel\": 0.0,
            \"algorithm\": \"${planner^^}\"
        }")

    # 解析结果
    echo $infer_result | python3 -c "
import sys, json
try:
    data = json.load(sys.stdin)
    if data.get('success'):
        print('✓ Inference successful')
        print(f\"  Time: {data['inference_time']:.3f}s\")
        print(f\"  Checkpoint: {data['checkpoint'].split('/')[-1]}\")
        print('  Parameters:')
        for key, val in data['parameters'].items():
            print(f'    {key:<25} = {val:>10.4f}')
    else:
        print('❌ Inference failed')
        print(json.dumps(data, indent=2))
except Exception as e:
    print(f'❌ Error parsing result: {e}')
    print(sys.stdin.read())
"
}

# 测试不同的 checkpoints
echo ""
echo "=================================================="
echo "Starting Tests"
echo "=================================================="

# 测试 1: DDP checkpoint-12500
test_checkpoint ddp 12500 6

# 测试 2: DDP checkpoint-10000
test_checkpoint ddp 10000 6

# 可选：测试其他规划器（如果你想的话）
# test_checkpoint dwa 12500 7
# test_checkpoint teb 12500 7
# test_checkpoint mppi 12500 6

echo ""
echo "=================================================="
echo "All tests completed!"
echo "=================================================="
