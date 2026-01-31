#!/bin/bash

# 支持多实例（不同端口）
# 优先使用外部设置的 ROS_MASTER_URI，否则基于 INSTANCE_ID 生成
if [ -z "$ROS_MASTER_URI" ]; then
    INSTANCE_ID=${INSTANCE_ID:-0}
    ROS_MASTER_PORT=$((11311 + INSTANCE_ID))
    export ROS_MASTER_URI=http://localhost:${ROS_MASTER_PORT}
    echo "[INFO] Using INSTANCE_ID $INSTANCE_ID: ROS_MASTER_URI=$ROS_MASTER_URI"

    # Gazebo 使用独立端口池 (12000-12299)
    GAZEBO_MASTER_PORT=$((12000 + INSTANCE_ID))
    export GAZEBO_MASTER_URI=http://localhost:${GAZEBO_MASTER_PORT}
    echo "[INFO] GAZEBO_MASTER_URI=$GAZEBO_MASTER_URI"
else
    echo "[INFO] Using external ROS_MASTER_URI=$ROS_MASTER_URI"
fi

export ROS_HOSTNAME=${ROS_HOSTNAME:-localhost}

# 为每个实例创建独立的临时目录（避免 /tmp 冲突）
INSTANCE_TMP="/tmp/singularity_instance_${INSTANCE_ID:-$$}"
mkdir -p "$INSTANCE_TMP"
echo "[INFO] Instance temp directory: $INSTANCE_TMP"

# 使用主机网络命名空间以访问 Qwen 服务（gpu011:5000）
echo "[INFO] Using host network namespace for Qwen service access"

# 构建环境变量参数
ENV_VARS="--env ROS_MASTER_URI=$ROS_MASTER_URI --env ROS_HOSTNAME=$ROS_HOSTNAME"
if [ -n "$ROS_LOG_DIR" ]; then
    ENV_VARS="$ENV_VARS --env ROS_LOG_DIR=$ROS_LOG_DIR"
    echo "[INFO] ROS_LOG_DIR=$ROS_LOG_DIR"
fi
if [ -n "$QWEN_HOST" ]; then
    ENV_VARS="$ENV_VARS --env QWEN_HOST=$QWEN_HOST"
    echo "[INFO] QWEN_HOST=$QWEN_HOST"
fi
if [ -n "$QWEN_PORT" ]; then
    ENV_VARS="$ENV_VARS --env QWEN_PORT=$QWEN_PORT"
    echo "[INFO] QWEN_PORT=$QWEN_PORT"
fi
if [ -n "$GAZEBO_MASTER_URI" ]; then
    ENV_VARS="$ENV_VARS --env GAZEBO_MASTER_URI=$GAZEBO_MASTER_URI"
    echo "[INFO] GAZEBO_MASTER_URI=$GAZEBO_MASTER_URI"
fi
if [ -n "$INSTANCE_ID" ]; then
    ENV_VARS="$ENV_VARS --env INSTANCE_ID=$INSTANCE_ID"
    echo "[INFO] INSTANCE_ID=$INSTANCE_ID"
fi

# 检测是否需要 GPU 支持 (--nv)
# 设置 USE_GPU=1 强制启用，USE_GPU=0 强制禁用
# 默认：自动检测 nvidia-smi 是否可用
NV_FLAG=""
if [ "${USE_GPU}" = "1" ]; then
    NV_FLAG="--nv"
    echo "[INFO] GPU support enabled (USE_GPU=1)"
elif [ "${USE_GPU}" = "0" ]; then
    echo "[INFO] GPU support disabled (USE_GPU=0)"
elif command -v nvidia-smi &> /dev/null && nvidia-smi &> /dev/null; then
    NV_FLAG="--nv"
    echo "[INFO] GPU detected, enabling --nv"
else
    echo "[INFO] No GPU detected, running in CPU mode"
fi

# 运行容器
# 使用独立的 /tmp 目录避免多实例冲突
# 使用 --writable-tmpfs 提供可写的临时文件系统
singularity exec --writable-tmpfs ${NV_FLAG} \
    $ENV_VARS \
    -B `pwd`:/jackal_ws/src/ros_jackal \
    -B ${INSTANCE_TMP}:/tmp \
    ${1} /bin/bash /jackal_ws/src/ros_jackal/entrypoint.sh ${@:2}
