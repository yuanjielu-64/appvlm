#!/bin/bash

#SBATCH --partition=normal
#SBATCH --job-name=CSV2JSON
#SBATCH --exclude=amd057,amd058,amd059,amd060
#SBATCH --output=logs/csv2json-%j.out
#SBATCH --error=logs/csv2json-%j.err
#SBATCH --time=0-4:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=4GB

echo "=== 作业信息 ==="
echo "作业ID: $SLURM_JOB_ID"
echo "节点名称: $SLURM_NODELIST"
echo "CPU类型: $(cat /proc/cpuinfo | grep 'model name' | head -1 | cut -d':' -f2)"
echo "CPU核心数: $SLURM_CPUS_PER_TASK"
echo "内存: $(free -h | grep Mem | awk '{print $2}')"
echo "开始时间: $(date)"
echo "================="

# Default parameters
ALG=${1:-"ddp"}
ROOT_DIR=${2:-"/scratch/ylu22/app_data/ddp_heurstic"}
CSV_NAME=${3:-"data.csv"}
TRAJECTORY_NAME=${4:-"data_trajectory.csv"}

echo ""
echo "=== 转换参数 ==="
echo "算法: $ALG"
echo "根目录: $ROOT_DIR"
echo "CSV文件名: $CSV_NAME"
echo "轨迹文件名: $TRAJECTORY_NAME"
echo "================="
echo ""

module load python/3.8.6-ff

# Create logs directory if it doesn't exist
mkdir -p logs

# Run the conversion script
python csv_to_json.py \
    --alg "$ALG" \
    --root_dir "$ROOT_DIR" \
    --csv_name "$CSV_NAME" \
    --trajectory_name "$TRAJECTORY_NAME"

echo ""
echo "=== 完成 ==="
echo "结束时间: $(date)"
echo "============="
