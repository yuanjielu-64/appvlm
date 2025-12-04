#!/bin/bash
# 启动Qwen2.5-VL推理服务

# 获取脚本目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# 配置参数
BASE_MODEL="Qwen/Qwen2.5-VL-7B-Instruct"  # ✅ 修正：你用的是3B模型
LORA_PATH=""
DEVICE_MAP="auto"     # 使用auto让模型自动分配
ALGORITHM="DWA"
PORT=5000
STARTUP_WARMUP=true
STARTUP_TOKENS=16
LOAD_IN_4BIT=true
LOAD_IN_8BIT=false

# 🔍 性能分析配置
ENABLE_PROFILER=false  # 设为true启用详细profiler (会增加10-20%开销)
CUDA_TIMING=true       # 使用CUDA事件精确计时

# Conda环境Python解释器
CONDA_PYTHON="/home/yuanjielu/miniforge3/envs/lmms-finetune-qwen/bin/python"

# Qwen服务脚本路径
QWEN_SERVER="${SCRIPT_DIR}/qwen_server.py"

echo "=================================================="
echo "  Starting Qwen2.5-VL Navigation Service"
echo "=================================================="
echo "Base Model:    ${BASE_MODEL}"
echo "LoRA Path:     ${LORA_PATH}"
echo "Device Map:    ${DEVICE_MAP}"
echo "Algorithm:     ${ALGORITHM}"
echo "Port:          ${PORT}"
echo "4-bit Quant:   ${LOAD_IN_4BIT}"
echo "8-bit Quant:   ${LOAD_IN_8BIT}"
echo "Startup Warm:  ${STARTUP_WARMUP}"
echo "Profiler:      ${ENABLE_PROFILER}"
echo "CUDA Timing:   ${CUDA_TIMING}"
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

if [ "${CUDA_TIMING}" = false ]; then
  CMD+=( --no_cuda_timing )
fi

echo "Launching: ${CMD[*]}"
"${CMD[@]}"

echo "Qwen service stopped."
