#!/bin/bash

# 支持多实例（不同端口）
INSTANCE_ID=${INSTANCE_ID:-0}
ROS_MASTER_PORT=$((11311 + INSTANCE_ID))

export ROS_HOSTNAME=localhost
export ROS_MASTER_URI=http://localhost:${ROS_MASTER_PORT}

echo "[INFO] Instance $INSTANCE_ID: ROS_MASTER_URI=$ROS_MASTER_URI"

# 传递OpenAI API Key到容器
if [ -z "$OPENAI_API_KEY" ]; then
    echo "[WARNING] OPENAI_API_KEY not set. API calls will fail."
fi

# 尝试使用 --fakeroot (需要管理员配置)
# 如果失败，尝试 sudo
if singularity exec --fakeroot --network=bridge echo "test" &>/dev/null; then
    echo "[INFO] Using --fakeroot for network access"
    NETWORK_FLAGS="--fakeroot --network=bridge"
elif sudo -n singularity exec --network=bridge echo "test" &>/dev/null; then
    echo "[INFO] Using sudo for network access"
    USE_SUDO="sudo -E"
    NETWORK_FLAGS="--network=bridge"
else
    echo "[ERROR] Cannot enable network. Options:"
    echo "  1. Ask admin to run: sudo singularity config fakeroot --add $USER"
    echo "  2. Give yourself passwordless sudo"
    echo "  3. Run Python script on host instead of container"
    exit 1
fi

# 运行容器
$USE_SUDO singularity exec -i --nv -n $NETWORK_FLAGS -p \
    --env OPENAI_API_KEY="$OPENAI_API_KEY" \
    --env ROS_MASTER_URI="$ROS_MASTER_URI" \
    -B `pwd`:/jackal_ws/src/ros_jackal \
    ${1} /bin/bash /jackal_ws/src/ros_jackal/entrypoint.sh ${@:2}
