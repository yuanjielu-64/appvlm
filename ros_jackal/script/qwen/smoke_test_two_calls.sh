#!/usr/bin/env bash
set -euo pipefail

# Quick two-call smoke test against a running qwen_server
# - Verifies /health
# - Sends two /infer calls on the same image
# - Prints response success + inference_time and curl total time

# 🔍 智能检测 appvlm_ws 路径
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
APPVLM_WS="$(cd "${SCRIPT_DIR}" && while [[ "$PWD" != "/" ]]; do
  if [[ "$(basename "$PWD")" == "appvlm_ws" ]]; then
    echo "$PWD";
    break;
  fi;
  cd ..;
done)"

# 备用路径检测
if [[ -z "$APPVLM_WS" ]]; then
  for candidate in \
    "/home/yuanjielu/robot_navigation/noetic/appvlm_ws" \
    "/data/local/yl2832/appvlm_ws" \
    "$HOME/robot_navigation/noetic/appvlm_ws" \
    "$HOME/appvlm_ws"; do
    if [[ -d "$candidate" ]]; then
      APPVLM_WS="$candidate"
      break
    fi
  done
fi

# 默认使用检测到的路径
DEFAULT_IMAGE="${APPVLM_WS}/src/ros_jackal/buffer/ddp_gpt/actor_0/VLM_000000.png"

QWEN_URL=${QWEN_URL:-http://localhost:5001}
IMAGE_PATH=${1:-$DEFAULT_IMAGE}
ALGORITHM=${ALGORITHM:-DDP}  # ✅ 改为DDP测试你的新模型
TIMEOUT=${TIMEOUT:-120}

echo "[SMOKE] QWEN_URL=${QWEN_URL}"
echo "[SMOKE] IMAGE_PATH=${IMAGE_PATH}"
echo "[SMOKE] ALGORITHM=${ALGORITHM}"

if [ ! -f "$IMAGE_PATH" ]; then
  echo "[SMOKE] ERROR: Image not found: $IMAGE_PATH" >&2
  exit 1
fi

echo "[SMOKE] Waiting for health ok (timeout ${TIMEOUT}s)..."
start_ts=$(date +%s)
while true; do
  if curl -s "${QWEN_URL}/health" | grep -q '"status":"ok"'; then
    echo "[SMOKE] ✓ Service healthy"
    break
  fi
  now=$(date +%s)
  if [ $((now - start_ts)) -ge ${TIMEOUT} ]; then
    echo "[SMOKE] ERROR: Service not healthy within ${TIMEOUT}s" >&2
    exit 2
  fi
  sleep 2
done

payload() {
  cat <<JSON
{
  "image_path": "${IMAGE_PATH}",
  "linear_vel": 0.0,
  "angular_vel": 0.0,
  "algorithm": "${ALGORITHM}"
}
JSON
}

do_call() {
  idx=$1
  echo "[SMOKE] ---- Call #${idx} ----"
  # Capture curl timing and response body separately
  resp_file=$(mktemp)
  time_file=$(mktemp)

  # ✅ 使用 2>&1 确保所有输出完成，然后再解析
  curl -sS -X POST "${QWEN_URL}/infer" \
       -H 'Content-Type: application/json' \
       -d "$(payload)" \
       -w '%{time_total}\n' \
       -o "$resp_file" \
       > "$time_file" 2>&1

  # ✅ 等待文件写入完成
  sync

  total_time=$(cat "$time_file")
  success=$(jq -r '.success // false' "$resp_file" 2>/dev/null || echo false)
  inf_time=$(jq -r '.inference_time // null' "$resp_file" 2>/dev/null || echo null)

  echo "[SMOKE] curl_total=${total_time}s, success=${success}, inference_time=${inf_time}"

  # ✅ 显示完整响应以便调试
  if command -v jq >/dev/null 2>&1; then
    echo "[SMOKE] Response:"
    jq '.' "$resp_file" 2>/dev/null || cat "$resp_file"
  else
    cat "$resp_file"
  fi

  rm -f "$resp_file" "$time_file"
}

do_call 1
do_call 2

echo "[SMOKE] Done."

