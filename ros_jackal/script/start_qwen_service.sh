#!/bin/bash
# 启动Qwen2.5-VL推理服务

# 获取脚本目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# 配置参数
BASE_MODEL="Qwen/Qwen2.5-VL-7B-Instruct"  # ✅ 使用7B模型（与训练时一致）
LORA_PATH="/home/yuanjielu/robot_navigation/noetic/appvlm_ws/src/ros_jackal/model/ddp/checkpoint-12500"  # ✅ checkpoint-12500模型
HEAD_TYPE="dpt"       # ✅ DPT head
NUM_PARAMS=6          # ✅ DDP有6个参数
DEVICE_MAP="auto"     # 使用auto让模型自动分配
ALGORITHM="DDP"       # ✅ 改为DDP（你训练的算法）
PORT=5000
STARTUP_WARMUP=true
STARTUP_TOKENS=16
LOAD_IN_4BIT=true
LOAD_IN_8BIT=false

# 🔍 性能分析配置
ENABLE_PROFILER=false  # 设为true启用详细profiler (会增加10-20%开销)
CUDA_TIMING=true       # 使用CUDA事件精确计时

# 🚀 性能优化配置（默认启用安全优化）
ENABLE_OPTIMIZATIONS=true      # 总开关
USE_FLASH_ATTENTION=true       # FlashAttention-2/SDPA (推荐)
OPTIMIZE_MEMORY=true           # 内存优化 (推荐)

# Conda环境Python解释器
CONDA_PYTHON="/home/yuanjielu/miniforge3/envs/lmms-finetune-qwen/bin/python"

# Qwen服务脚本路径
QWEN_SERVER="${SCRIPT_DIR}/qwen_server_flash_attn.py"

# 🔇 抑制不重要的警告（设置环境变量）
export TRANSFORMERS_VERBOSITY=error  # 只显示错误，隐藏警告
export TOKENIZERS_PARALLELISM=false  # 避免tokenizer警告
export PYTHONWARNINGS="ignore::FutureWarning,ignore::UserWarning,ignore::DeprecationWarning"

# 🛡️ 禁用PyTorch编译器优化（避免CUDA Graphs内存错误）
export TORCH_COMPILE_DISABLE=1
export TORCHDYNAMO_DISABLE=1
export CUDA_LAUNCH_BLOCKING=0  # 保持异步执行以提高性能

echo "=================================================="
echo "  Starting Qwen2.5-VL Navigation Service"
echo "=================================================="
echo "Base Model:    ${BASE_MODEL}"
echo "LoRA Path:     ${LORA_PATH}"
echo "Head Type:     ${HEAD_TYPE}"
echo "Num Params:    ${NUM_PARAMS}"
echo "Device Map:    ${DEVICE_MAP}"
echo "Algorithm:     ${ALGORITHM}"
echo "Port:          ${PORT}"
echo "4-bit Quant:   ${LOAD_IN_4BIT}"
echo "8-bit Quant:   ${LOAD_IN_8BIT}"
echo "Startup Warm:  ${STARTUP_WARMUP}"
echo "Profiler:      ${ENABLE_PROFILER}"
echo "CUDA Timing:   ${CUDA_TIMING}"
echo ""
echo "🚀 Performance Optimizations:"
echo "  FlashAttn:   ${USE_FLASH_ATTENTION}"
echo "  Memory Opt:  ${OPTIMIZE_MEMORY}"
echo "=================================================="

# 检查文件是否存在
if [ ! -f "${QWEN_SERVER}" ]; then
    echo "Error: qwen_server.py not found at ${QWEN_SERVER}"
    exit 1
fi

# 构建基础命令
CMD=(
    "${CONDA_PYTHON}" "${QWEN_SERVER}"
    --base_model "${BASE_MODEL}"
    --lora_path "${LORA_PATH}"
    --head_type "${HEAD_TYPE}"
    --num_params ${NUM_PARAMS}
    --algorithm "${ALGORITHM}"
    --port ${PORT}
    --max_new_tokens 30  # ✅ 优化：只需输出7个数字，30 tokens足够
)

# 添加可选参数
if [ -n "${DEVICE_MAP}" ]; then
  CMD+=( --device_map "${DEVICE_MAP}" )
fi

if [ "${LOAD_IN_4BIT}" = true ]; then
  CMD+=( --load_in_4bit )
fi

if [ "${LOAD_IN_8BIT}" = true ]; then
  CMD+=( --load_in_8bit )
fi

if [ "${STARTUP_WARMUP}" = true ]; then
  CMD+=( --startup_warmup --startup_tokens ${STARTUP_TOKENS} )
fi

# 🔍 性能分析选项
if [ "${ENABLE_PROFILER}" = true ]; then
  CMD+=( --enable_profiler )
fi

# 🚀 性能优化选项
if [ "${ENABLE_OPTIMIZATIONS}" = true ]; then
  # FlashAttention
  if [ "${USE_FLASH_ATTENTION}" = true ]; then
    CMD+=( --use_flash_attention )
  fi

  # 内存优化
  if [ "${OPTIMIZE_MEMORY}" = true ]; then
    CMD+=( --optimize_memory )
  fi
else
  CMD+=( --no_optimizations )
fi

if [ "${CUDA_TIMING}" = false ]; then
  CMD+=( --no_cuda_timing )
fi

echo "Launching: ${CMD[*]}"
"${CMD[@]}"

echo "Qwen service stopped."
