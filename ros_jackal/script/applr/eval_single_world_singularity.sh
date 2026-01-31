#!/bin/bash
# 在 Singularity 容器中运行单世界评估（单个实例）
# 用法:
#   bash eval_single_world_singularity.sh --id WORLD_IDX --policy_name POLICY
#
# 示例:
#   bash eval_single_world_singularity.sh --id 0 --policy_name dwa_heurstic
#   bash eval_single_world_singularity.sh --id 100 --policy_name ddp_heurstic

set -e

# ============================================================
# 参数配置
# ============================================================

# 获取脚本目录和项目路径
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROS_JACKAL_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"  # script/applr/ -> ros_jackal/
CONTAINER_IMAGE="${ROS_JACKAL_DIR}/jackal.sif"

# 默认值
POLICY_NAME="dwa_heurstic"
WORLD_IDX=""

# 解析命名参数
while [[ $# -gt 0 ]]; do
  case $1 in
    --id)
      WORLD_IDX="$2"
      shift 2
      ;;
    --policy_name)
      POLICY_NAME="$2"
      shift 2
      ;;
    *)
      echo "未知参数: $1"
      echo "用法: $0 --id WORLD_IDX --policy_name POLICY"
      exit 1
      ;;
  esac
done

# 检查必需参数
if [ -z "$WORLD_IDX" ]; then
    echo "❌ 错误: 必须指定 --id 参数"
    echo "用法: $0 --id WORLD_IDX --policy_name POLICY"
    exit 1
fi

# ============================================================
# 端口和环境配置
# ============================================================

# 使用 WORLD_IDX 创建唯一的端口，避免冲突
ROS_PORT=$((11311 + WORLD_IDX))
GAZEBO_PORT=$((11345 + WORLD_IDX))

export ROS_HOSTNAME=localhost
export ROS_MASTER_URI=http://localhost:${ROS_PORT}
export GAZEBO_MASTER_URI=http://localhost:${GAZEBO_PORT}

# 创建唯一的 ROS 日志目录
export ROS_LOG_DIR=/tmp/ros_eval_${WORLD_IDX}
mkdir -p $ROS_LOG_DIR
chmod 777 $ROS_LOG_DIR

# 禁用 GPU (Gazebo 仿真不需要 GPU，避免 GLIBC 版本冲突)
export USE_GPU=0

# 检查容器是否存在
if [ ! -f "$CONTAINER_IMAGE" ]; then
    echo "❌ 容器镜像不存在: $CONTAINER_IMAGE"
    echo ""
    echo "请确保容器位于: ${ROS_JACKAL_DIR}/jackal.sif"
    exit 1
fi

echo "=================================================="
echo "  Singularity 单世界评估 (世界 #${WORLD_IDX})"
echo "=================================================="
echo "容器镜像:    $CONTAINER_IMAGE"
echo "策略名称:    $POLICY_NAME"
echo "世界索引:    $WORLD_IDX"
echo "--------------------------------------------------"
echo "ROS 配置:"
echo "  ROS_MASTER_URI: $ROS_MASTER_URI"
echo "  GAZEBO_MASTER_URI: $GAZEBO_MASTER_URI"
echo "  ROS_LOG_DIR:    $ROS_LOG_DIR"
echo "=================================================="
echo ""

# ============================================================
# 切换到 ros_jackal 目录 (singularity_run.sh 使用 pwd)
# ============================================================

cd "${ROS_JACKAL_DIR}"

# ============================================================
# 清理当前实例使用端口的残留进程
# ============================================================

echo "清理残留进程..."
fuser -k ${ROS_PORT}/tcp 2>/dev/null || true
fuser -k ${GAZEBO_PORT}/tcp 2>/dev/null || true
sleep 1
echo ""

# ============================================================
# 运行评估
# ============================================================

echo "========================================="
echo "开始评估"
echo "Time: $(date '+%Y-%m-%d %H:%M:%S')"
echo "========================================="

# 设置实例 ID（用于容器隔离）
export INSTANCE_ID=${WORLD_IDX}

# 使用 singularity_run.sh 运行评估
./singularity_run.sh ${CONTAINER_IMAGE} \
    python3 script/applr/evaluate_applr_single.py \
        --world_id ${WORLD_IDX} \
        --policy_name ${POLICY_NAME} \
        --buffer_path buffer/ \
        --ros_port ${ROS_PORT}

echo ""
echo "========================================="
echo "评估完成"
echo "Time: $(date '+%Y-%m-%d %H:%M:%S')"
echo "========================================="
echo ""

# ============================================================
# 清理临时目录
# ============================================================

echo "清理临时文件..."
INSTANCE_TMP="/tmp/singularity_instance_${WORLD_IDX}"
if [ -d "$INSTANCE_TMP" ]; then
    rm -rf "$INSTANCE_TMP"
    echo "✓ 已删除临时目录: $INSTANCE_TMP"
fi

# 清理 ROS 日志目录（可选）
if [ -d "$ROS_LOG_DIR" ]; then
    echo "ℹ️  ROS 日志保留在: $ROS_LOG_DIR"
    # 如果需要删除，取消注释下面这行
    # rm -rf "$ROS_LOG_DIR"
fi

echo ""
echo "=================================================="
echo "✓ 世界 #${WORLD_IDX} 评估完成！"
echo "=================================================="
