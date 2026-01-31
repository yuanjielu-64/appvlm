#!/bin/bash
# 单世界快速评估 - 在 Singularity 容器中运行单个世界的单次评估
# 用途: 快速测试/调试某个特定世界的表现
#
# 用法:
#   bash eval_single_world_singularity.sh [WORLD_IDX] [POLICY_NAME]
#
# 示例:
#   bash eval_single_world_singularity.sh                    # 评估世界0，使用默认policy (ddp_qwen)
#   bash eval_single_world_singularity.sh 100                # 评估世界100
#   bash eval_single_world_singularity.sh 50 dwa_qwen        # 评估世界50，使用dwa_qwen

set -e

# ============================================================
# 参数配置
# ============================================================

# 获取脚本目录，固定容器路径
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONTAINER_IMAGE="$(cd "$SCRIPT_DIR/../.." && pwd)/jackal.sif"  # script/qwen/ -> ros_jackal/

WORLD_IDX="${1:-0}"
POLICY_NAME="${2:-ddp_qwen}"
QWEN_URL="${QWEN_URL:-http://localhost:5000}"

# ============================================================
# 🔍 智能检测路径
# ============================================================

# 方法1: 向上查找 appvlm_ws
APPVLM_WS="$(cd "${SCRIPT_DIR}" && while [[ "$PWD" != "/" ]]; do
  if [[ "$(basename "$PWD")" == "appvlm_ws" ]]; then
    echo "$PWD";
    break;
  fi;
  cd ..;
done)"

# 方法2: 备用路径列表
if [[ -z "$APPVLM_WS" ]]; then
  for candidate in \
    "/home/yuanjielu/robot_navigation/noetic/appvlm_ws" \
    "/data/local/yl2832/appvlm_ws" \
    "$HOME/robot_navigation/noetic/appvlm_ws" \
    "$HOME/appvlm_ws"; do
    if [[ -d "$candidate" ]]; then
      APPVLM_WS="$candidate"
      break
    fi
  done
fi

# 验证
if [[ -z "$APPVLM_WS" ]]; then
  echo "❌ Error: Cannot find appvlm_ws directory!"
  exit 1
fi

ROS_JACKAL_DIR="${APPVLM_WS}/src/ros_jackal"

echo "=================================================="
echo "  Singularity 单次评估"
echo "=================================================="
echo "容器镜像:  $CONTAINER_IMAGE"
echo "环境 ID:   $WORLD_IDX"
echo "策略名称:  $POLICY_NAME"
echo "Qwen URL:  $QWEN_URL"
echo "工作区:    $APPVLM_WS"
echo "=================================================="
echo ""

# ============================================================
# 检查容器
# ============================================================

if [ ! -f "$CONTAINER_IMAGE" ]; then
    echo "❌ 容器镜像不存在: $CONTAINER_IMAGE"
    echo ""
    echo "请确保容器位于: src/ros_jackal/jackal.sif"
    exit 1
fi

# ============================================================
# 检查 Qwen 服务
# ============================================================

echo "检查 Qwen 服务..."
if ! curl -s --max-time 5 "${QWEN_URL}/health" > /dev/null 2>&1; then
    echo "❌ 无法连接到 Qwen 服务: ${QWEN_URL}"
    echo ""
    echo "请先启动 Qwen 服务:"
    echo "  cd ${ROS_JACKAL_DIR}/script/qwen"
    echo "  ./start_qwen_service.sh"
    echo ""
    exit 1
fi

echo "✓ Qwen 服务正常"
echo ""

# ============================================================
# 运行评估
# ============================================================

echo "启动容器评估..."
echo ""

singularity exec \
  --network host \
  --bind ${APPVLM_WS}:/workspace/appvlm_ws \
  --pwd /workspace/appvlm_ws \
  --nv \
  ${CONTAINER_IMAGE} \
  bash -c "
    set -e
    echo '容器内环境初始化...'
    source /opt/ros/noetic/setup.bash
    source /workspace/appvlm_ws/devel/setup.bash
	
	
	
    # 清理之前的进程
    killall -9 rosmaster 2>/dev/null || true
    killall gzclient 2>/dev/null || true
    killall gzserver 2>/dev/null || true

    echo '运行评估脚本...'
    python3 /workspace/appvlm_ws/src/ros_jackal/script/qwen/evaluate_qwen_single.py \
      --world_idx ${WORLD_IDX} --buffer /workspace/appvlm_ws/src/ros_jackal/buffer/ --policy_name ddp_qwen
  "

echo ""
echo "=================================================="
echo "✓ 评估完成"
echo "=================================================="
