#!/bin/bash
# 在 Hopper 上运行 checkpoint switching 和 inference 测试
#
# 使用方法:
#   1. 先启动 Qwen 服务 (已经通过 hopper_qwen_dynamic.slurm 启动)
#   2. 确认服务在哪个节点运行 (例如: gpu017)
#   3. 运行此脚本: bash run_test_on_hopper.sh
#
# 注意:
#   - 需要修改 QWEN_HOST 为实际的 GPU 节点名
#   - 需要提供一个真实的 costmap 图片路径

# ===================== 配置 =====================

# Qwen 服务节点 (根据 squeue 查看你的 job 实际运行在哪个节点)
export QWEN_HOST=${QWEN_HOST:-gpu017}  # 👈 修改为你的节点
export QWEN_PORT=${QWEN_PORT:-5000}

# 测试图片 (使用 buffer 中的一个示例图片)
export TEST_IMAGE=${TEST_IMAGE:-"/home/yuanjielu/robot_navigation/noetic/appvlm_ws/src/ros_jackal/buffer/ddp_gpt/actor_0/VLM_000250.png"}

# ===================== 运行测试 =====================

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

echo "================================================================"
echo "Testing Qwen Dynamic Checkpoint Switching on Hopper"
echo "================================================================"
echo "Qwen Service: ${QWEN_HOST}:${QWEN_PORT}"
echo "Test Image: ${TEST_IMAGE}"
echo "================================================================"
echo ""

# 检查图片是否存在
if [ ! -f "$TEST_IMAGE" ]; then
    echo "❌ Error: Test image not found: $TEST_IMAGE"
    echo ""
    echo "Available example images:"
    ls -lh /home/yuanjielu/robot_navigation/noetic/appvlm_ws/src/ros_jackal/buffer/ddp_gpt/actor_0/VLM_*.png | head -5
    echo "..."
    echo ""
    echo "Please set TEST_IMAGE to a valid path:"
    echo "  export TEST_IMAGE=/path/to/your/costmap.png"
    exit 1
fi

# 运行测试脚本
cd "$SCRIPT_DIR"
bash test_checkpoint_inference.sh

echo ""
echo "================================================================"
echo "Test completed!"
echo "================================================================"
