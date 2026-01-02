#!/bin/bash
# ============================================================
# 任务2: 运行机器人测试（Python 3.8，Singularity 容器内）
# 通过 HTTP 调用 Qwen 服务获取导航参数
# ============================================================

#SBATCH --partition=normal             # 可以用 CPU 节点（除非机器人系统也需要 GPU）
#SBATCH --job-name=robot_test
#SBATCH --exclude=amd057,amd058,amd059,amd060
#SBATCH --output=cpu_report/robot_test-%j.out
#SBATCH --error=cpu_report/robot_test-%j.err
#SBATCH --time=4-23:59:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=1GB

echo "=================================================="
echo "Robot Test Job"
echo "SLURM Job ID: $SLURM_JOB_ID"
echo "Running on node: $(hostname)"
echo "=================================================="

# 容器配置
SINGULARITY_BASE=/scratch/ylu22/appvlm_ws/src/ros_jackal
SINGULARITY_IMG=$SINGULARITY_BASE/jackal.sif

# ROS 环境变量
export ROS_HOSTNAME=localhost
export ROS_MASTER_URI=http://localhost:11311
export ROS_LOG_DIR=/tmp

# 加载 Singularity
module load singularity

# 创建日志目录
mkdir -p cpu_report

# ============================================================
# 💡 关键：设置 Qwen 服务地址
# ============================================================
# Hopper 使用完整域名格式：hopXXX.orc.gmu.edu
# 先运行 hopper_qwen_service.slurm，然后从日志中获取 hostname

# 方法1: 手动设置（推荐）
# 从 cpu_report/qwen_service-<JOB_ID>.out 中找到类似：
#   "Running on node: hop042.orc.gmu.edu"
# 然后填入下面：
QWEN_HOST="<REPLACE_WITH_QWEN_NODE>"  # 例如: hop042.orc.gmu.edu
QWEN_PORT=5000

# 方法2: 使用 IP 地址（如果 hostname 不工作）
# QWEN_HOST="172.16.145.42"  # 从 Qwen 日志中找到 IP

export QWEN_SERVICE_URL="http://${QWEN_HOST}:${QWEN_PORT}"

echo ""
echo "Qwen service URL: $QWEN_SERVICE_URL"
echo ""

# 检查 Qwen 服务是否可达
echo "🔍 Checking Qwen service availability..."
if curl -s --connect-timeout 5 ${QWEN_SERVICE_URL}/health > /dev/null 2>&1; then
    echo "✓ Qwen service is reachable!"
else
    echo "❌ ERROR: Cannot reach Qwen service at $QWEN_SERVICE_URL"
    echo "   Please check:"
    echo "   1. Qwen service job is running (squeue)"
    echo "   2. QWEN_HOST is set correctly"
    echo "   3. Firewall allows inter-node communication"
    exit 1
fi

# ============================================================
# 运行机器人测试（在容器内）
# ============================================================
cd /home/yuanjielu/robot_navigation/noetic/appvlm_ws/src/ros_jackal

echo ""
echo "🤖 Running robot test inside Singularity container..."
echo ""

# 将 QWEN_SERVICE_URL 传入容器
singularity exec \
    --nv \
    --bind /scratch:/scratch \
    --bind /home:/home \
    --env QWEN_SERVICE_URL=$QWEN_SERVICE_URL \
    $SINGULARITY_IMG \
    bash script/test.sh

TEST_EXIT_CODE=$?

echo ""
echo "=================================================="
echo "Robot test completed with exit code: $TEST_EXIT_CODE"
echo "=================================================="

exit $TEST_EXIT_CODE
