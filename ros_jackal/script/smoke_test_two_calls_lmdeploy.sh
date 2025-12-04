#!/usr/bin/env bash
set -euo pipefail

# LMDeploy 服务两次调用冒烟测试（自动发现端口）
# 用法：
#   bash src/ros_jackal/script/smoke_test_two_calls_lmdeploy.sh [IMAGE_PATH]
# 可选环境变量：
#   QWEN_URL=http://localhost:5001  指定服务地址（若不指定则自动扫描端口）
#   ALGORITHM=DWA                   算法名（默认 DWA）
#   TIMEOUT=120                     健康检查超时秒数（默认 120）

IMAGE_PATH=${1:-/home/yuanjielu/robot_navigation/noetic/appvlm_ws/src/ros_jackal/buffer/dwa_qwen/actor_0/VLM_000000.png}
ALGORITHM=${ALGORITHM:-DWA}
TIMEOUT=${TIMEOUT:-120}

if [ ! -f "$IMAGE_PATH" ]; then
  echo "[SMOKE-LMD] ERROR: Image not found: $IMAGE_PATH" >&2
  exit 1
fi

detect_url() {
  if [ -n "${QWEN_URL:-}" ]; then
    echo "$QWEN_URL"
    return 0
  fi
  # 自动扫描常见端口（5001 起步，最多尝试 20 个）
  local base=5001
  local max=20
  for ((i=0;i<max;i++)); do
    local port=$((base+i))
    local url="http://localhost:${port}"
    # 健康检查并确认是 LMDeploy 版（health 中包含 backend 字段）
    echo "[SMOKE-LMD] Scanning ${url}/health ..."
    if curl -s --connect-timeout 0.3 --max-time 0.8 "${url}/health" | grep -q '"backend"'; then
      echo "$url"
      return 0
    fi
  done
  return 1
}

QWEN_URL=$(detect_url || true)
if [ -z "$QWEN_URL" ]; then
  echo "[SMOKE-LMD] ERROR: Failed to locate LMDeploy service on localhost:5001-5020. Set QWEN_URL env var." >&2
  exit 2
fi

echo "[SMOKE-LMD] QWEN_URL=${QWEN_URL}"
echo "[SMOKE-LMD] IMAGE_PATH=${IMAGE_PATH}"
echo "[SMOKE-LMD] ALGORITHM=${ALGORITHM}"

echo "[SMOKE-LMD] Waiting for health ok (timeout ${TIMEOUT}s)..."
start_ts=$(date +%s)
while true; do
  health_json=$(curl -s --connect-timeout 0.5 --max-time 1.5 "${QWEN_URL}/health" || true)
  if echo "$health_json" | grep -q '"status":"ok"'; then
    echo "[SMOKE-LMD] ✓ Service healthy"
    # 打印后端标识，便于确认是否 TurboMind/PyTorch/HF
    if command -v jq >/dev/null 2>&1; then
      echo "$health_json" | jq '.' || true
    else
      echo "$health_json"
    fi
    break
  fi
  now=$(date +%s)
  if [ $((now - start_ts)) -ge ${TIMEOUT} ]; then
    echo "[SMOKE-LMD] ERROR: Service not healthy within ${TIMEOUT}s" >&2
    exit 3
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
  echo "[SMOKE-LMD] ---- Call #${idx} ----"
  resp_file=$(mktemp)
  time_file=$(mktemp)
  curl -sS -X POST "${QWEN_URL}/infer" \
       -H 'Content-Type: application/json' \
       -d "$(payload)" \
       -w '%{time_total}\n' \
       -o "$resp_file" \
       > "$time_file"

  total_time=$(cat "$time_file")
  success=$(jq -r '.success // false' "$resp_file" 2>/dev/null || echo false)
  inf_time=$(jq -r '.inference_time // null' "$resp_file" 2>/dev/null || echo null)

  echo "[SMOKE-LMD] curl_total=${total_time}s, success=${success}, inference_time=${inf_time}"
  if command -v jq >/dev/null 2>&1; then
    jq '{success, inference_time, parameters_array}' "$resp_file" || true
  else
    cat "$resp_file"
  fi

  rm -f "$resp_file" "$time_file"
}

do_call 1
do_call 2

echo "[SMOKE-LMD] Done."
