#!/bin/bash
# 启动 Qwen2.5-VL 推理服务 (Hopper 版本)
#
# 用法:
#   bash start_qwen_service_hopper.sh [planner] [checkpoint_num] [port]
#
# 示例:
#   bash start_qwen_service_hopper.sh DDP 7500 5000
#   bash start_qwen_service_hopper.sh DWA 10000 5001

# 获取脚本目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# ============================================================
# 参数解析
# ============================================================
PLANNER=${1:-TEB}
CHECKPOINT_NUM=${2:-7500}
PORT=${3:-5000}

# 转换为大写
PLANNER=$(echo "$PLANNER" | tr '[:lower:]' '[:upper:]')
PLANNER_LOWER=$(echo "$PLANNER" | tr '[:upper:]' '[:lower:]')

# ============================================================
# 根据 planner 设置参数
# ============================================================
case $PLANNER in
    DWA)
        NUM_PARAMS=7
        ;;
    TEB)
        NUM_PARAMS=7
        ;;
    MPPI)
        NUM_PARAMS=8
        ;;
    DDP)
        NUM_PARAMS=6
        ;;
    *)
        echo "❌ Invalid planner: $PLANNER"
        echo "   Supported: DWA, TEB, MPPI, DDP"
        exit 1
        ;;
esac

# ============================================================
# 查找 checkpoint 路径
# ============================================================
BASE_DIR="/scratch/bwang25/appvlm_ws/src/ros_jackal/model/${PLANNER_LOWER}"

# 尝试多个可能的路径
POSSIBLE_PATHS=(
    "${BASE_DIR}/qwen2.5-vl-regression_lora-True_${PLANNER_LOWER}_regression_1/checkpoint-${CHECKPOINT_NUM}"
    "${BASE_DIR}/qwen2.5-vl-regression_lora-True_${PLANNER_LOWER}_regression/checkpoint-${CHECKPOINT_NUM}"
)

CHECKPOINT_PATH=""
for path in "${POSSIBLE_PATHS[@]}"; do
    if [ -d "$path" ]; then
        CHECKPOINT_PATH="$path"
        break
    fi
done

if [ -z "$CHECKPOINT_PATH" ]; then
    echo "❌ Checkpoint not found: $PLANNER checkpoint-$CHECKPOINT_NUM"
    echo ""
    echo "Searched in:"
    for path in "${POSSIBLE_PATHS[@]}"; do
        echo "  - $path"
    done
    exit 1
fi

# ============================================================
# 配置参数
# ============================================================
BASE_MODEL="Qwen/Qwen2.5-VL-7B-Instruct"
HEAD_TYPE="dpt"
DEVICE_MAP="auto"
LOAD_IN_4BIT=true

# Conda 环境 Python 解释器
CONDA_PYTHON="/home/ylu22/miniforge/envs/lmms-finetune-qwen/bin/python"

# Qwen 服务脚本路径
QWEN_SERVER="${SCRIPT_DIR}/qwen_server_flash_attn.py"

# 🔇 抑制不重要的警告（设置环境变量）
export TRANSFORMERS_VERBOSITY=error  # 只显示错误，隐藏警告
export TOKENIZERS_PARALLELISM=false  # 避免tokenizer警告
export PYTHONWARNINGS="ignore::FutureWarning,ignore::UserWarning,ignore::DeprecationWarning"

# 🛡️ 禁用PyTorch编译器优化（避免CUDA Graphs内存错误）
export TORCH_COMPILE_DISABLE=1
export TORCHDYNAMO_DISABLE=1
export CUDA_LAUNCH_BLOCKING=0  # 保持异步执行以提高性能

echo ""
echo "=================================================="
echo "Starting Qwen2.5-VL Navigation Service (Hopper)"
echo "=================================================="
echo "Configuration:"
echo "  Planner:       ${PLANNER}"
echo "  Checkpoint:    checkpoint-${CHECKPOINT_NUM}"
echo "  Full Path:     ${CHECKPOINT_PATH}"
echo "  Base Model:    ${BASE_MODEL}"
echo "  Head Type:     ${HEAD_TYPE}"
echo "  Num Params:    ${NUM_PARAMS}"
echo "  Device Map:    ${DEVICE_MAP}"
echo "  Port:          ${PORT}"
echo "  4-bit Quant:   ${LOAD_IN_4BIT}"
echo ""
echo "Python:"
echo "  Interpreter:   ${CONDA_PYTHON}"
echo "  Server Script: ${QWEN_SERVER}"
echo "=================================================="
echo ""

# 检查文件是否存在
if [ ! -f "${QWEN_SERVER}" ]; then
    echo "❌ Error: qwen_server_flash_attn.py not found at ${QWEN_SERVER}"
    echo ""
    echo "Available scripts in ${SCRIPT_DIR}:"
    ls -1 "${SCRIPT_DIR}"/*.py 2>/dev/null || echo "  (no python scripts found)"
    exit 1
fi

if [ ! -f "${CONDA_PYTHON}" ]; then
    echo "❌ Error: Python interpreter not found at ${CONDA_PYTHON}"
    echo ""
    echo "Please check conda environment:"
    echo "  conda env list | grep lmms-finetune-qwen"
    exit 1
fi

# 检查 checkpoint 是否包含必要文件
echo "Validating checkpoint..."
REQUIRED_FILES=(
    "${CHECKPOINT_PATH}/adapter_model.safetensors"
    "${CHECKPOINT_PATH}/regression_head/pytorch_model.bin"
)

for file in "${REQUIRED_FILES[@]}"; do
    if [ ! -f "$file" ]; then
        echo "⚠️  Warning: Missing file: $(basename $(dirname $file))/$(basename $file)"
    else
        echo "  ✓ $(basename $(dirname $file))/$(basename $file)"
    fi
done
echo ""

# 构建命令
CMD=(
    "${CONDA_PYTHON}" "${QWEN_SERVER}"
    --base_model "${BASE_MODEL}"
    --lora_path "${CHECKPOINT_PATH}"
    --head_type "${HEAD_TYPE}"
    --num_params ${NUM_PARAMS}
    --algorithm "${PLANNER}"
    --port ${PORT}
    --device_map "${DEVICE_MAP}"
)

if [ "${LOAD_IN_4BIT}" = true ]; then
  CMD+=( --load_in_4bit )
fi

echo "Launching service..."
echo "Command: ${CMD[*]}"
echo ""
echo "=================================================="
echo ""

# 运行服务
"${CMD[@]}"

echo ""
echo "=================================================="
echo "Qwen service stopped."
echo "=================================================="
