#!/bin/bash
# ============================================================
# 在 Hopper CPU 节点运行机器人测试（Singularity）
# 连接到 GPU 节点上的 Qwen 服务进行参数预测
# ============================================================

#SBATCH --partition=normal
#SBATCH --job-name=qwen_robot_test
#SBATCH --exclude=amd057,amd058,amd059,amd060
#SBATCH --output=executable/cpu_report/qwen-robot-test-%j.out
#SBATCH --error=executable/cpu_report/qwen-robot-test-%j.err
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
END_WORLD=${END_WORLD:-10}
RUNS_PER_WORLD=${RUNS_PER_WORLD:-1}
POLICY_NAME=${POLICY_NAME:-ddp_qwen}

echo "========================================="
echo "Qwen Robot Navigation Test (Singularity)"
echo "========================================="
echo "SLURM Job ID: $SLURM_JOB_ID"
echo "Running on: $(hostname)"
echo "Qwen Service: http://${QWEN_HOST}:${QWEN_PORT}"
echo "Test Worlds: ${START_WORLD} to ${END_WORLD}"
echo "Runs per world: ${RUNS_PER_WORLD}"
echo "Policy: ${POLICY_NAME}"
echo "========================================="
echo ""

# ============================================================
# Singularity 配置
# ============================================================
SINGULARITY_BASE=/scratch/ylu22/appvlm/src/ros_jackal
SINGULARITY_IMG=$SINGULARITY_BASE/jackal.sif

export ROS_HOSTNAME=localhost
export ROS_MASTER_URI=http://localhost:11311
export ROS_LOG_DIR=/tmp

module load singularity

# 切换到项目根目录
cd /home/yuanjielu/robot_navigation/noetic/appvlm_ws/src/ros_jackal

# ============================================================
# 检查 Qwen 服务是否可达
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
curl -s http://${QWEN_HOST}:${QWEN_PORT}/health | python3 -c "
import sys, json
data = json.load(sys.stdin)
print(f\"  Current checkpoint: {data['current_checkpoint']}\")
print(f\"  Algorithm: {data['algorithm']}\")
" || echo "  (Failed to get details)"
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
                --policy_name ${POLICY_NAME}

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
