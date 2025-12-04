#!/bin/bash
# 启动基于 LMDeploy 的 Qwen2.5-VL 推理服务（不替换现有服务，新增并行版本）

# 获取脚本目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# 配置参数（可根据需要调整，支持环境变量覆盖）
BASE_MODEL="${BASE_MODEL:-Qwen/Qwen2.5-VL-3B-Instruct}"  # 也可切换 7B
ALGORITHM="${ALGORITHM:-DWA}"                            # 默认算法，可在请求内覆盖
PORT="${PORT:-5001}"                                    # 与 transformers 版本区分开
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-80}"

# LMDeploy 相关（可选）
BACKEND="${BACKEND:-pytorch}"      # pytorch 后端（避免 TurboMind 转换 OOM，性能仍优于原生 HF）
SESSION_LEN="${SESSION_LEN:-2048}" # 会话长度（降低以减少显存，导航prompt较短）
TP="${TP:-1}"                      # tensor parallel（单卡保持 1）

# Conda环境Python解释器（与现有保持一致，可按需调整）
CONDA_PYTHON="/home/yuanjielu/miniforge3/envs/lmms-finetune-qwen/bin/python"

# 新的 LMDeploy 服务端脚本
LMDEPLOY_SERVER="${SCRIPT_DIR}/qwen_server_lmdeploy.py"

echo "=================================================="
echo "  Starting Qwen2.5-VL (LMDeploy) Service"
echo "=================================================="

# 若端口被占用，则自动寻找下一个可用端口（最多尝试 20 次）
is_port_in_use() {
  local p=$1
  if command -v ss >/dev/null 2>&1; then
    ss -ltn | grep -q ":${p} "
    return $?
  elif command -v lsof >/dev/null 2>&1; then
    lsof -iTCP:"${p}" -sTCP:LISTEN >/dev/null 2>&1
    return $?
  else
    # 保守：无法检测，认为未被占用
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
    echo "[ERROR] Failed to find a free port near ${BASE_PORT}. Please free the port or set PORT env var."
    exit 1
  fi
done
echo "[INFO] Using free port: ${PORT}"
echo "Base Model:     ${BASE_MODEL}"
echo "Algorithm:      ${ALGORITHM}"
echo "Port:           ${PORT}"
echo "LMDeploy:       backend=${BACKEND}, tp=${TP}, session_len=${SESSION_LEN}"
echo "Max new tokens: ${MAX_NEW_TOKENS}"
echo "=================================================="

if [ ! -f "${LMDEPLOY_SERVER}" ]; then
  echo "Error: qwen_server_lmdeploy.py not found at ${LMDEPLOY_SERVER}"
  exit 1
fi

CMD=(
  "${CONDA_PYTHON}" "${LMDEPLOY_SERVER}"
  --base_model "${BASE_MODEL}"
  --algorithm "${ALGORITHM}"
  --port ${PORT}
  --max_new_tokens ${MAX_NEW_TOKENS}
  --backend "${BACKEND}"
  --tp ${TP}
  --session_len ${SESSION_LEN}
)

echo "Launching: ${CMD[*]}"
"${CMD[@]}"

echo "Qwen LMDeploy service stopped."
