#!/bin/bash
# ============================================================
# 在 Hopper CPU 节点运行机器人测试（Singularity）
# 连接到 GPU 节点上的 Qwen 服务进行参数预测
# ============================================================

#SBATCH --partition=normal
#SBATCH --job-name=qwen_robot_test
#SBATCH --exclude=amd057,amd058,amd059,amd060
#SBATCH --output=cpu_report1/qwen-robot-test-%j.out
#SBATCH --error=cpu_report1/qwen-robot-test-%j.err
#SBATCH --time=4-23:59:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=1GB

# ============================================================
# 配置参数 (可以通过 --export 传入)
# ============================================================
# Qwen 服务配置
export QWEN_HOST=${QWEN_HOST:-gpu011}  # GPU 节点名
export QWEN_PORT=${QWEN_PORT:-5000}

# 测试配置
START_WORLD=${START_WORLD:-0}
END_WORLD=${END_WORLD:-299}
RUNS_PER_WORLD=${RUNS_PER_WORLD:-1}
POLICY_NAME=${POLICY_NAME:-ddp_qwen}
ALGORITHM=${ALGORITHM:-DDP}
NUM_PARAMS=${NUM_PARAMS:-6}

echo "========================================="
echo "Qwen Robot Navigation Test (Singularity)"
echo "========================================="
echo "SLURM Job ID: $SLURM_JOB_ID"
echo "Running on: $(hostname)"
echo "Algorithm: ${ALGORITHM}"
echo "Qwen Service: http://${QWEN_HOST}:${QWEN_PORT}"
echo "Test Worlds: ${START_WORLD} to ${END_WORLD}"
echo "Runs per world: ${RUNS_PER_WORLD}"
echo "Policy: ${POLICY_NAME}"
echo "Num Params: ${NUM_PARAMS}"
echo "========================================="
echo ""

# ============================================================
# Singularity 配置
# ============================================================
SINGULARITY_BASE=/scratch/ylu22/appvlm_ws/src/ros_jackal
SINGULARITY_IMG=$SINGULARITY_BASE/jackal.sif

# 使用 SLURM_JOB_ID 创建唯一的 ROS Master 端口，避免冲突
ROS_PORT=$((11311 + SLURM_JOB_ID % 10000))
export ROS_HOSTNAME=localhost
export ROS_MASTER_URI=http://localhost:${ROS_PORT}
export ROS_LOG_DIR=/tmp/ros_${SLURM_JOB_ID}

# 创建 ROS 日志目录（在宿主机上）
mkdir -p $ROS_LOG_DIR
chmod 777 $ROS_LOG_DIR

echo "ROS Configuration:"
echo "  ROS_MASTER_URI: $ROS_MASTER_URI"
echo "  ROS_LOG_DIR: $ROS_LOG_DIR"
echo ""

module load singularity

# 切换到项目根目录（在Hopper上使用scratch目录）
cd /scratch/ylu22/appvlm_ws/src/ros_jackal

# ============================================================
# 检查 Qwen 服务是否可达并获取 checkpoint 路径
# ============================================================
echo "Checking Qwen service at http://${QWEN_HOST}:${QWEN_PORT}..."
if ! curl -s --connect-timeout 5 http://${QWEN_HOST}:${QWEN_PORT}/health > /dev/null 2>&1; then
    echo "❌ Error: Cannot reach Qwen service"
    echo ""
    echo "Please check:"
    echo "  1. Is Qwen service running?"
    echo "     squeue -u \$USER | grep qwen"
    echo "  2. Is QWEN_HOST correct?"
    echo "     Current value: $QWEN_HOST"
    echo "  3. Submit with correct node:"
    echo "     sbatch --export=QWEN_HOST=gpu021 executable/hopper_robot_qwen_test.slurm"
    exit 1
fi

echo "✓ Qwen service is reachable"

# 🔧 获取 checkpoint 路径
CHECKPOINT_PATH=$(curl -s http://${QWEN_HOST}:${QWEN_PORT}/health | python3 -c "
import sys, json
data = json.load(sys.stdin)
print(data['current_checkpoint'])
" 2>/dev/null)

if [ -z "$CHECKPOINT_PATH" ]; then
    echo "❌ Error: Failed to get checkpoint path from Qwen service"
    exit 1
fi

echo "  Current checkpoint: $CHECKPOINT_PATH"
curl -s http://${QWEN_HOST}:${QWEN_PORT}/health | python3 -c "
import sys, json
data = json.load(sys.stdin)
print(f\"  Algorithm: {data['algorithm']}\")
" || echo "  (Failed to get algorithm)"

# 🔧 提取 checkpoint 编号作为 actor_id
if [[ $CHECKPOINT_PATH =~ checkpoint-([0-9]+) ]]; then
    ACTOR_ID="${BASH_REMATCH[1]}"
    echo "  Checkpoint number: $ACTOR_ID"
else
    ACTOR_ID="0"
    echo "  Warning: Could not extract checkpoint number, using default actor_id=0"
fi

echo ""

# ============================================================
# 运行机器人测试
# ============================================================
for i in $(seq $START_WORLD $END_WORLD) ; do
    for j in $(seq 1 $RUNS_PER_WORLD) ; do
        echo "========================================="
        echo "World: $i, Run: $j/$RUNS_PER_WORLD"
        echo "Time: $(date '+%Y-%m-%d %H:%M:%S')"
        echo "========================================="

        # 在 Singularity 容器内运行测试
        # 环境变量通过 singularity_run.sh 自动传递
        export QWEN_HOST=${QWEN_HOST}
        export QWEN_PORT=${QWEN_PORT}

        ./singularity_run.sh $SINGULARITY_IMG \
            python3 script/evaluate_qwen_hopper.py \
                --world_idx $i \
                --qwen_host ${QWEN_HOST} \
                --qwen_port ${QWEN_PORT} \
                --policy_name ${POLICY_NAME} \
                --id ${ACTOR_ID} \
                --buffer_path buffer/

        echo "World $i Run $j completed"
        echo ""

        sleep 4
    done
done

echo ""
echo "========================================="
echo "All tests completed!"
echo "Time: $(date '+%Y-%m-%d %H:%M:%S')"
echo "========================================="

# ============================================================
# 📝 写入完成 signal 到 checkpoint 目录
# ============================================================
if [ -n "$CHECKPOINT_PATH" ]; then
    SIGNAL_FILE="${CHECKPOINT_PATH}/evaluation_complete.signal"

    echo ""
    echo "Writing completion signal to checkpoint directory..."
    echo "Signal file: $SIGNAL_FILE"

    # 创建简单的 signal 文件（只记录 checkpoint 路径）
    echo "$CHECKPOINT_PATH" > "$SIGNAL_FILE"

    echo "✓ Signal file created successfully"
    echo "  Checkpoint: $CHECKPOINT_PATH"
else
    echo "⚠ Warning: CHECKPOINT_PATH not set, signal file not created"
fi
