#!/bin/bash
# 启动基于 vLLM 的 Qwen2.5-VL 推理服务（支持 finetune 模型）

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# 配置参数（支持环境变量覆盖）
BASE_MODEL="${BASE_MODEL:-Qwen/Qwen2.5-VL-3B-Instruct}"  # 可替换为 finetune 后的本地路径
ALGORITHM="${ALGORITHM:-DWA}"
PORT="${PORT:-5003}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-80}"

# vLLM 相关
MAX_MODEL_LEN="${MAX_MODEL_LEN:-2048}"      # 上下文长度（导航 prompt 短，2048 足够）
GPU_MEMORY_UTIL="${GPU_MEMORY_UTIL:-0.85}"  # GPU 显存利用率
TP="${TP:-1}"                               # Tensor Parallel（单卡=1）

CONDA_PYTHON="/home/yuanjielu/miniforge3/envs/lmms-finetune-qwen/bin/python"
VLLM_SERVER="${SCRIPT_DIR}/qwen_server_vllm.py"

echo "=================================================="
echo "  Starting Qwen2.5-VL (vLLM) Service"
echo "=================================================="

# 端口检测（与 LMDeploy 版本相同逻辑）
is_port_in_use() {
  local p=$1
  if command -v ss >/dev/null 2>&1; then
    ss -ltn | grep -q ":${p} "
    return $?
  elif command -v lsof >/dev/null 2>&1; then
    lsof -iTCP:"${p}" -sTCP:LISTEN >/dev/null 2>&1
    return $?
  else
    return 1
  fi
}

TRY_COUNT=0
MAX_TRY=20
BASE_PORT=${PORT}
while is_port_in_use "${PORT}"; do
  echo "[WARN] Port ${PORT} is in use. Trying next..."
  PORT=$((BASE_PORT + TRY_COUNT + 1))
  TRY_COUNT=$((TRY_COUNT + 1))
  if [ ${TRY_COUNT} -ge ${MAX_TRY} ]; then
    echo "[ERROR] Failed to find a free port near ${BASE_PORT}."
    exit 1
  fi
done

echo "[INFO] Using free port: ${PORT}"
echo "Base Model:     ${BASE_MODEL}"
echo "Algorithm:      ${ALGORITHM}"
echo "Port:           ${PORT}"
echo "vLLM Config:    max_len=${MAX_MODEL_LEN}, gpu_util=${GPU_MEMORY_UTIL}, tp=${TP}"
echo "Max new tokens: ${MAX_NEW_TOKENS}"
echo "=================================================="

if [ ! -f "${VLLM_SERVER}" ]; then
  echo "Error: qwen_server_vllm.py not found at ${VLLM_SERVER}"
  exit 1
fi

# 检查 vLLM 是否安装
if ! "${CONDA_PYTHON}" -c "import vllm" 2>/dev/null; then
  echo ""
  echo "❌ vLLM not installed in conda env: lmms-finetune-qwen"
  echo ""
  echo "请先安装 vLLM："
  echo "  conda activate lmms-finetune-qwen"
  echo "  pip install vllm"
  echo ""
  echo "或使用官方推荐方式："
  echo "  pip install vllm --extra-index-url https://download.pytorch.org/whl/cu118"
  echo ""
  exit 2
fi

CMD=(
  "${CONDA_PYTHON}" "${VLLM_SERVER}"
  --base_model "${BASE_MODEL}"
  --algorithm "${ALGORITHM}"
  --port ${PORT}
  --max_new_tokens ${MAX_NEW_TOKENS}
  --max_model_len ${MAX_MODEL_LEN}
  --gpu_memory_util ${GPU_MEMORY_UTIL}
  --tp ${TP}
)

echo "Launching: ${CMD[*]}"
"${CMD[@]}"

echo "Qwen vLLM service stopped."
