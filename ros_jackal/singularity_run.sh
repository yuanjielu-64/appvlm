#!/bin/bash

# 支持多实例（不同端口）
INSTANCE_ID=${INSTANCE_ID:-0}
ROS_MASTER_PORT=$((11311 + INSTANCE_ID))

export ROS_HOSTNAME=localhost
export ROS_MASTER_URI=http://localhost:${ROS_MASTER_PORT}

echo "[INFO] Instance $INSTANCE_ID: ROS_MASTER_URI=$ROS_MASTER_URI"

# 使用主机网络命名空间以访问 Qwen 服务（gpu011:5000）
echo "[INFO] Using host network namespace for Qwen service access"

# 构建环境变量参数
ENV_VARS="--env ROS_MASTER_URI=$ROS_MASTER_URI"
if [ -n "$QWEN_HOST" ]; then
    ENV_VARS="$ENV_VARS --env QWEN_HOST=$QWEN_HOST"
    echo "[INFO] QWEN_HOST=$QWEN_HOST"
fi
if [ -n "$QWEN_PORT" ]; then
    ENV_VARS="$ENV_VARS --env QWEN_PORT=$QWEN_PORT"
    echo "[INFO] QWEN_PORT=$QWEN_PORT"
fi

# 运行容器（移除 -n 标志以使用主机网络，这样可以访问 gpu011:5000）
singularity exec -i --nv -p \
    $ENV_VARS \
    -B `pwd`:/jackal_ws/src/ros_jackal \
    ${1} /bin/bash /jackal_ws/src/ros_jackal/entrypoint.sh ${@:2}
