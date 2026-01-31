#!/bin/bash
# 批量世界评估 - 在 Singularity 容器中运行多个世界的完整评估
# 用途: 完整性能评估，支持多实例并行运行，默认评估300个世界x3次
#
# 用法:
#   bash eval_batch_worlds_singularity.sh --id ID [--start N] [--end N] [--runs N] [--policy NAME]
#
# 示例:
#   bash eval_batch_worlds_singularity.sh --id 1                                   # 实例1: 评估世界0-299 x3次
#   bash eval_batch_worlds_singularity.sh --id 2 --policy mppi_heurstic           # 实例2: 使用mppi_heurstic
#   bash eval_batch_worlds_singularity.sh --id 3 --start 0 --end 99               # 实例3: 只评估世界0-99

set -e

# ============================================================
# 参数配置
# ============================================================

# 获取脚本目录和项目路径
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROS_JACKAL_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"  # script/HB/ -> ros_jackal/
CONTAINER_IMAGE="${ROS_JACKAL_DIR}/jackal.sif"

# 默认值
START_WORLD=0
END_WORLD=299
RUNS_PER_WORLD=3
POLICY_NAME="dwa_heurstic"
CONTAINER_ID=""

# 解析命名参数
while [[ $# -gt 0 ]]; do
  case $1 in
    --id)
      CONTAINER_ID="$2"
      shift 2
      ;;
    --start)
      START_WORLD="$2"
      shift 2
      ;;
    --end)
      END_WORLD="$2"
      shift 2
      ;;
    --runs)
      RUNS_PER_WORLD="$2"
      shift 2
      ;;
    --policy)
      POLICY_NAME="$2"
      shift 2
      ;;
    *)
      echo "未知参数: $1"
      echo "用法: $0 --id CONTAINER_ID [--start N] [--end N] [--runs N] [--policy NAME]"
      exit 1
      ;;
  esac
done

# 检查必需参数
if [ -z "$CONTAINER_ID" ]; then
    echo "❌ 错误: 必须指定 --id 参数"
    echo "用法: $0 --id CONTAINER_ID [--start N] [--end N] [--runs N] [--policy NAME]"
    exit 1
fi

# ============================================================
# 端口和环境配置
# ============================================================

# 使用 CONTAINER_ID 创建唯一的端口，避免冲突
ROS_PORT=$((11311 + CONTAINER_ID))
GAZEBO_PORT=$((11345 + CONTAINER_ID))

export ROS_HOSTNAME=localhost
export ROS_MASTER_URI=http://localhost:${ROS_PORT}
export GAZEBO_MASTER_URI=http://localhost:${GAZEBO_PORT}

# 创建唯一的 ROS 日志目录
export ROS_LOG_DIR=/tmp/ros_instance_${CONTAINER_ID}
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
echo "  Singularity 批量评估 (实例 #${CONTAINER_ID})"
echo "=================================================="
echo "容器镜像:    $CONTAINER_IMAGE"
echo "环境范围:    $START_WORLD - $END_WORLD"
echo "每环境运行:  ${RUNS_PER_WORLD} 次"
echo "策略名称:    $POLICY_NAME"
echo "--------------------------------------------------"
echo "ROS 配置:"
echo "  ROS_MASTER_URI: $ROS_MASTER_URI"
echo "  GAZEBO_MASTER_URI: $GAZEBO_MASTER_URI"
echo "  ROS_LOG_DIR:    $ROS_LOG_DIR"
echo "=================================================="
echo ""

# ============================================================
# 计算总任务数
# ============================================================

TOTAL_ENVS=$((END_WORLD - START_WORLD + 1))
TOTAL_RUNS=$((TOTAL_ENVS * RUNS_PER_WORLD))

echo "总任务数: $TOTAL_RUNS (${TOTAL_ENVS} 个环境 x ${RUNS_PER_WORLD} 次)"
echo ""

# ============================================================
# 切换到 ros_jackal 目录 (singularity_run.sh 使用 pwd)
# ============================================================

cd "${ROS_JACKAL_DIR}"

# ============================================================
# 初始清理：清理当前实例使用端口的残留进程
# ============================================================

echo "清理端口 ${ROS_PORT} 和 ${GAZEBO_PORT} 的残留进程..."
fuser -k ${ROS_PORT}/tcp 2>/dev/null || true
fuser -k ${GAZEBO_PORT}/tcp 2>/dev/null || true
sleep 2
echo ""

# ============================================================
# 运行评估循环 (使用 singularity_run.sh)
# ============================================================

for i in $(seq $END_WORLD -1 $START_WORLD); do  # 倒序
    for j in $(seq 1 $RUNS_PER_WORLD); do
        echo "========================================="
        echo "World: $i, Run: $j/$RUNS_PER_WORLD"
        echo "Time: $(date '+%Y-%m-%d %H:%M:%S')"
        echo "========================================="

        # 清理当前实例使用端口的残留进程
        fuser -k ${ROS_PORT}/tcp 2>/dev/null || true
        fuser -k ${GAZEBO_PORT}/tcp 2>/dev/null || true
        sleep 1

        # 设置实例 ID（用于容器隔离）
        export INSTANCE_ID=${CONTAINER_ID}

        # 使用 singularity_run.sh 运行评估
        ./singularity_run.sh ${CONTAINER_IMAGE} \
            python3 script/HB/evaluate_HB_single.py \
                --world_id $i \
                --policy_name ${POLICY_NAME} \
                --buffer_path buffer/ \
                --ros_port ${ROS_PORT}

        echo "World $i Run $j completed"
        echo ""

        sleep 4
    done
done

echo ""
echo "=================================================="
echo "✓ 实例 #${CONTAINER_ID} 所有评估完成！"
echo "=================================================="
echo "Time: $(date '+%Y-%m-%d %H:%M:%S')"
echo "总运行次数: $TOTAL_RUNS"
echo "环境范围: $START_WORLD - $END_WORLD"
echo "ROS_MASTER_URI: $ROS_MASTER_URI"
echo "=================================================="
echo ""

# 清理临时目录
echo "清理临时文件..."
INSTANCE_TMP="/tmp/singularity_instance_${CONTAINER_ID}"
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
