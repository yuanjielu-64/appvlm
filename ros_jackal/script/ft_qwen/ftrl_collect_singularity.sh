#!/bin/bash
# FTRL 数据收集 - 在 Singularity 容器中运行单个 actor
#
# 架构说明:
#   - 一次脚本调用 = 一个 FTRL 服务器 = 一个容器
#   - world_lists 是该容器要收集的环境列表（逗号分隔）
#   - 在 rlft/train.py 中会多次调用本脚本，每次使用不同的 server_url 和 world_lists
#
# 用法:
#   bash ftrl_collect_singularity.sh --server_url URL --world_lists "0,1,2,3,4"
#
# 示例:
#   # 收集环境 [0,1,2,3,4]，连接到服务器 6000
#   bash ftrl_collect_singularity.sh \
#       --server_url http://localhost:6000 \
#       --world_lists "0,1,2,3,4"
#
#   # 收集环境 [5,6,7,8,9]，连接到服务器 6001
#   bash ftrl_collect_singularity.sh \
#       --server_url http://localhost:6001 \
#       --world_lists "5,6,7,8,9"

set -e

# ============================================================
# 参数配置
# ============================================================

# 获取脚本目录和项目路径
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROS_JACKAL_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"  # script/ft_qwen/ -> ros_jackal/
CONTAINER_IMAGE="${ROS_JACKAL_DIR}/jackal.sif"

# 默认值
WORLD_LISTS=""                     # 环境列表（逗号分隔）
SERVER_URL=""                      # FTRL 服务 URL
POLICY_NAME="ddp_rlft"             # 策略名称
BUFFER_PATH="buffer/"              # Buffer 路径
TRAIN_LIMIT="3"                    # 每个环境训练轮数
TEST_LIMIT="1"                     # 每个环境测试轮数

# 解析命名参数
while [[ $# -gt 0 ]]; do
  case $1 in
    --world_lists)
      WORLD_LISTS="$2"
      shift 2
      ;;
    --server_url)
      SERVER_URL="$2"
      shift 2
      ;;
    --policy_name)
      POLICY_NAME="$2"
      shift 2
      ;;
    --buffer_path)
      BUFFER_PATH="$2"
      shift 2
      ;;
    --train_limit)
      TRAIN_LIMIT="$2"
      shift 2
      ;;
    --test_limit)
      TEST_LIMIT="$2"
      shift 2
      ;;
    --actor_id)
      # 兼容旧参数，但不使用
      shift 2
      ;;
    *)
      echo "未知参数: $1"
      echo ""
      echo "用法: $0 --server_url URL --world_lists \"0,1,2\" [--train_limit 3] [--test_limit 1]"
      echo ""
      exit 1
      ;;
  esac
done

# ============================================================
# 检查必需参数和文件
# ============================================================

if [ -z "$SERVER_URL" ]; then
    echo "❌ 错误: 必须指定 --server_url 参数"
    echo "用法: $0 --server_url URL --world_lists \"0,1,2\""
    exit 1
fi

if [ -z "$WORLD_LISTS" ]; then
    echo "❌ 错误: 必须指定 --world_lists 参数"
    echo "用法: $0 --server_url URL --world_lists \"0,1,2\""
    exit 1
fi

# 解析环境列表（去除空格）
# 将 "0, 1, 2" 转换为 "0,1,2"
WORLD_LISTS=$(echo "$WORLD_LISTS" | sed 's/[[:space:]]//g')
IFS=',' read -ra WORLDS <<< "$WORLD_LISTS"
NUM_WORLDS=${#WORLDS[@]}

if [ $NUM_WORLDS -eq 0 ]; then
    echo "❌ 错误: world_lists 为空"
    exit 1
fi

if [ ! -f "$CONTAINER_IMAGE" ]; then
    echo "❌ 容器镜像不存在: $CONTAINER_IMAGE"
    echo ""
    echo "请确保容器位于: ${ROS_JACKAL_DIR}/jackal.sif"
    exit 1
fi

# 创建 buffer 目录
FULL_BUFFER_PATH="${ROS_JACKAL_DIR}/${BUFFER_PATH}"
mkdir -p "${FULL_BUFFER_PATH}"

echo "=================================================="
echo "  FTRL 数据收集"
echo "=================================================="
echo "容器镜像:     $CONTAINER_IMAGE"
echo "环境列表:     [${WORLD_LISTS}]"
echo "环境数量:     $NUM_WORLDS"
echo "策略名称:     $POLICY_NAME"
echo "Buffer 路径:  $BUFFER_PATH"
echo "--------------------------------------------------"
echo "服务配置:"
echo "  FTRL 服务:    $SERVER_URL"
echo "=================================================="
echo ""

# ============================================================
# 检查 FTRL 服务
# ============================================================

echo "检查 FTRL 服务..."
if ! curl -s --max-time 5 "${SERVER_URL}/health" > /dev/null 2>&1; then
    echo "❌ 无法连接到 FTRL 服务: ${SERVER_URL}"
    echo ""
    echo "请确保 FTRL 服务已启动"
    exit 1
fi

echo "✓ FTRL 服务正常"
echo ""

# ============================================================
# 端口和环境配置说明
# ============================================================
# 注意: ROS_PORT 将在循环中根据每个 world_idx 动态设置
# 每个 world 使用独立的端口避免冲突

export ROS_HOSTNAME=localhost
export USE_GPU=0  # 禁用 GPU (Gazebo 仿真不需要 GPU)

# ============================================================
# 切换到 ros_jackal 目录 (singularity_run.sh 使用 pwd)
# ============================================================

cd "${ROS_JACKAL_DIR}"

# ============================================================
# 运行数据收集循环
# ============================================================

# 为这批环境创建日志目录
LOG_DIR="${FULL_BUFFER_PATH}/logs"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/server_$(echo ${SERVER_URL} | sed 's|http://||' | sed 's|[:/]|_|g').log"

echo "开始收集数据..."
echo "环境列表: [${WORLD_LISTS}]"
echo "日志文件: ${LOG_FILE}"
echo ""

{
    echo "========================================="
    echo "服务器: ${SERVER_URL}"
    echo "环境列表: [${WORLD_LISTS}]"
    echo "开始时间: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "========================================="
    echo ""

    # 循环处理每个环境
    for world_idx in "${WORLDS[@]}"; do
        echo ""
        echo "========================================"
        echo "处理环境: world_${world_idx} (actor_${world_idx})"
        echo "时间: $(date '+%Y-%m-%d %H:%M:%S')"
        echo "========================================"

        # 关键修改: 每个 world 使用自己的 actor_id 和端口
        CURRENT_ACTOR_ID=${world_idx}
        ROS_PORT=$((11311 + CURRENT_ACTOR_ID))
        GAZEBO_PORT=$((11345 + CURRENT_ACTOR_ID))

        echo "  - Actor ID:       ${CURRENT_ACTOR_ID}"
        echo "  - ROS Port:       ${ROS_PORT}"
        echo "  - Gazebo Port:    ${GAZEBO_PORT}"

        # actor 目录会在容器内自动创建 (buffer/{policy_name}/actor_X/)
        # 不需要在主机上预创建

        # 设置环境变量
        export ROS_MASTER_URI=http://localhost:${ROS_PORT}
        export GAZEBO_MASTER_URI=http://localhost:${GAZEBO_PORT}
        export ROS_LOG_DIR=/tmp/ros_ftrl_actor_${CURRENT_ACTOR_ID}
        export INSTANCE_ID=${CURRENT_ACTOR_ID}

        mkdir -p ${ROS_LOG_DIR}
        chmod 777 ${ROS_LOG_DIR}

        # 清理当前端口的残留进程
        fuser -k ${ROS_PORT}/tcp 2>/dev/null || true
        fuser -k ${GAZEBO_PORT}/tcp 2>/dev/null || true
        sleep 1

        # 运行 evaluate_ftrl_single.py
        ./singularity_run.sh ${CONTAINER_IMAGE} \
            python3 script/ft_qwen/evaluate_ftrl_single.py \
                --id ${CURRENT_ACTOR_ID} \
                --server_url ${SERVER_URL} \
                --policy_name ${POLICY_NAME} \
                --buffer_path ${BUFFER_PATH} \
                --world_path jackal_helper/worlds/BARN/ \
                --train_limit ${TRAIN_LIMIT} \
                --test_limit ${TEST_LIMIT}

        echo "  ✓ 环境 ${world_idx} 完成"

        sleep 4
    done

    echo ""
    echo "========================================="
    echo "✓ 完成所有环境的数据收集"
    echo "完成时间: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "处理的环境: [${WORLD_LISTS}]"
    echo "========================================="

} 2>&1 | tee -a "${LOG_FILE}"

echo ""
echo "=================================================="
echo "✓ 数据收集完成！"
echo "=================================================="
echo "Time: $(date '+%Y-%m-%d %H:%M:%S')"
echo "服务器: ${SERVER_URL}"
echo "环境数量: $NUM_WORLDS"
echo "环境列表: [${WORLD_LISTS}]"
echo "日志文件: $LOG_FILE"
echo "=================================================="
echo ""

# ============================================================
# 清理临时文件
# ============================================================

echo "清理临时文件..."

# 清理所有处理过的 world 的临时目录
for world_idx in "${WORLDS[@]}"; do
    INSTANCE_TMP="/tmp/singularity_instance_${world_idx}"
    if [ -d "$INSTANCE_TMP" ]; then
        rm -rf "$INSTANCE_TMP"
        echo "✓ 已删除临时目录: $INSTANCE_TMP"
    fi

    ROS_LOG_DIR="/tmp/ros_ftrl_actor_${world_idx}"
    if [ -d "$ROS_LOG_DIR" ]; then
        echo "ℹ️  ROS 日志保留在: $ROS_LOG_DIR"
        # 如果需要删除，取消注释下面这行
        # rm -rf "$ROS_LOG_DIR"
    fi
done

echo ""
echo "数据已保存到: ${FULL_BUFFER_PATH}"
echo "可以在 rlft/train.py 中加载这些数据进行训练"
echo ""
