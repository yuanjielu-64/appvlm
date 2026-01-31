#!/bin/bash
# 简单的推理脚本 - 只传一张图，获取预测参数

if [ $# -lt 1 ]; then
    echo "Usage: $0 <image_path> [algorithm]"
    echo ""
    echo "Example:"
    echo "  $0 /path/to/costmap.png"
    echo "  $0 /path/to/costmap.png DDP"
    exit 1
fi

IMAGE_PATH=$1
ALGORITHM=${2:-DDP}  # 默认使用 DDP

QWEN_HOST=${QWEN_HOST:-localhost}
QWEN_PORT=${QWEN_PORT:-5000}
QWEN_URL="http://${QWEN_HOST}:${QWEN_PORT}"

echo "Sending image to Qwen for inference..."
echo "Image: $IMAGE_PATH"
echo "Algorithm: $ALGORITHM"
echo "Service: $QWEN_URL"
echo ""

# 检查图片是否存在
if [ ! -f "$IMAGE_PATH" ]; then
    echo "❌ Error: Image not found: $IMAGE_PATH"
    exit 1
fi

# 转换图片为 base64
image_base64=$(base64 -w 0 "$IMAGE_PATH")

# 发送推理请求
result=$(curl -s -X POST ${QWEN_URL}/infer \
    -H "Content-Type: application/json" \
    -d "{
        \"image_base64\": \"${image_base64}\",
        \"linear_vel\": 0.0,
        \"angular_vel\": 0.0,
        \"algorithm\": \"${ALGORITHM}\"
    }")

# 解析并显示结果
echo "$result" | python3 -c "
import sys, json
try:
    data = json.load(sys.stdin)
    if data.get('success'):
        print('✓ Inference successful')
        print(f\"  Time: {data['inference_time']:.3f}s\")
        print(f\"  Checkpoint: {data['checkpoint']}\")
        print('  Parameters:')
        for key, val in data['parameters'].items():
            print(f'    {key:<25} = {val:>10.4f}')
    else:
        print('❌ Inference failed')
        print(json.dumps(data, indent=2))
except Exception as e:
    print(f'❌ Error: {e}')
    print(sys.stdin.read())
"
