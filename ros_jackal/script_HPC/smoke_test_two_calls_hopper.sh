#!/usr/bin/env bash
set -euo pipefail

# Quick two-call smoke test against a running qwen_server
# - Verifies /health
# - Sends two /infer calls on the same image
# - Prints response success + inference_time and curl total time

# ============================================================
# 配置参数
# ============================================================
QWEN_URL=${QWEN_URL:-}
QWEN_HOST=${QWEN_HOST:-}
QWEN_PORT=${QWEN_PORT:-5005}
IMAGE_PATH=${1:-}
ALGORITHM=${ALGORITHM:-DDP}
TIMEOUT=${TIMEOUT:-120}

# ============================================================
# 自动检测服务地址（如果未提供）
# ============================================================
if [ -z "$QWEN_URL" ]; then
    if [ -z "$QWEN_HOST" ]; then
        # 尝试从最近的日志中检测
        echo "[SMOKE] Auto-detecting Qwen service..."
        LATEST_LOG=$(ls -t /scratch/ylu22/appvlm_ws/src/ros_jackal/cpu_report*/qwen_*.out 2>/dev/null | head -1)

        if [ -n "$LATEST_LOG" ]; then
            QWEN_HOST=$(grep "QWEN_HOST=" "$LATEST_LOG" 2>/dev/null | tail -1 | sed 's/.*QWEN_HOST="\([^"]*\)".*/\1/' || echo "")
            DETECTED_PORT=$(grep "Port:" "$LATEST_LOG" 2>/dev/null | tail -1 | awk '{print $NF}' || echo "")
            [ -n "$DETECTED_PORT" ] && QWEN_PORT="$DETECTED_PORT"
        fi

        # 如果还是找不到，使用 localhost
        if [ -z "$QWEN_HOST" ]; then
            QWEN_HOST="localhost"
            echo "[SMOKE] ⚠️  Could not auto-detect, using localhost:$QWEN_PORT"
        else
            echo "[SMOKE] Detected from log: $QWEN_HOST:$QWEN_PORT"
        fi
    fi
    QWEN_URL="http://${QWEN_HOST}:${QWEN_PORT}"
fi

# ============================================================
# 查找测试图像（如果未提供）
# ============================================================
if [ -z "$IMAGE_PATH" ]; then
    echo "[SMOKE] Searching for test image..."

    # 尝试查找 buffer 目录中的图像
    SEARCH_DIRS=(
        "/scratch/ylu22/app_data/ddp_heurstic/actor_0/"
        "$(dirname "$0")/../buffer"
    )

    for dir in "${SEARCH_DIRS[@]}"; do
        if [ -d "$dir" ]; then
            IMAGE_PATH=$(find "$dir" -name "VLM_*.png" -type f 2>/dev/null | head -1)
            if [ -n "$IMAGE_PATH" ]; then
                echo "[SMOKE] Found test image: $IMAGE_PATH"
                break
            fi
        fi
    done

    if [ -z "$IMAGE_PATH" ]; then
        echo "[SMOKE] ERROR: No test image found" >&2
        echo "[SMOKE] Please provide image path as argument:" >&2
        echo "[SMOKE]   $0 /path/to/test_image.png" >&2
        exit 1
    fi
fi

# ============================================================
# 验证配置
# ============================================================
echo ""
echo "=========================================="
echo "Smoke Test Configuration"
echo "=========================================="
echo "Service URL:  ${QWEN_URL}"
echo "Image:        ${IMAGE_PATH}"
echo "Algorithm:    ${ALGORITHM}"
echo "Timeout:      ${TIMEOUT}s"
echo "=========================================="
echo ""

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
  # 转换图片为 base64
  local image_base64=$(base64 -w 0 "$IMAGE_PATH")
  cat <<JSON
{
  "image_base64": "${image_base64}",
  "linear_vel": 0.0,
  "angular_vel": 0.0,
  "algorithm": "${ALGORITHM}"
}
JSON
}

do_call() {
  idx=$1
  echo ""
  echo "=========================================="
  echo "Call #${idx}"
  echo "=========================================="

  # Capture curl timing and response body separately
  resp_file=$(mktemp)
  time_file=$(mktemp)

  # 记录开始时间
  call_start=$(date +%s.%N)

  # ✅ 使用 2>&1 确保所有输出完成，然后再解析
  curl -sS -X POST "${QWEN_URL}/infer" \
       -H 'Content-Type: application/json' \
       -d "$(payload)" \
       -w '%{time_total}\n' \
       -o "$resp_file" \
       > "$time_file" 2>&1

  curl_exit=$?
  call_end=$(date +%s.%N)

  # ✅ 等待文件写入完成
  sync

  # 检查 curl 是否成功
  if [ $curl_exit -ne 0 ]; then
    echo "❌ FAILED: curl request failed (exit code: $curl_exit)"
    cat "$resp_file" 2>/dev/null || echo "(no response)"
    rm -f "$resp_file" "$time_file"
    return 1
  fi

  total_time=$(cat "$time_file" 2>/dev/null || echo "0")
  wall_time=$(echo "$call_end - $call_start" | bc 2>/dev/null || echo "N/A")
  success=$(jq -r '.success // false' "$resp_file" 2>/dev/null || echo false)
  inf_time=$(jq -r '.inference_time // null' "$resp_file" 2>/dev/null || echo null)
  checkpoint=$(jq -r '.checkpoint // "unknown"' "$resp_file" 2>/dev/null | xargs basename 2>/dev/null || echo "unknown")

  # 显示结果
  echo "Results:"
  echo "  Success:         $success"
  echo "  Inference Time:  ${inf_time}s"
  echo "  Curl Time:       ${total_time}s"
  echo "  Wall Time:       ${wall_time}s"
  echo "  Checkpoint:      $checkpoint"

  # ✅ 显示预测的参数
  if command -v jq >/dev/null 2>&1; then
    echo ""
    echo "Predicted Parameters:"
    jq -r '.parameters_array // [] | to_entries | .[] | "  [\(.key)]: \(.value)"' "$resp_file" 2>/dev/null || echo "  (parsing failed)"
  fi

  # 如果失败，显示完整响应
  if [ "$success" != "true" ]; then
    echo ""
    echo "⚠️  Full Response (inference failed):"
    if command -v jq >/dev/null 2>&1; then
      jq '.' "$resp_file" 2>/dev/null || cat "$resp_file"
    else
      cat "$resp_file"
    fi
  fi

  rm -f "$resp_file" "$time_file"
}

# 执行两次调用
do_call 1
CALL1_RESULT=$?

do_call 2
CALL2_RESULT=$?

# 总结
echo ""
echo "=========================================="
echo "Smoke Test Summary"
echo "=========================================="
if [ $CALL1_RESULT -eq 0 ] && [ $CALL2_RESULT -eq 0 ]; then
  echo "✅ Both calls succeeded"
  echo "[SMOKE] Done."
  exit 0
else
  echo "❌ Some calls failed"
  [ $CALL1_RESULT -ne 0 ] && echo "  - Call #1 failed"
  [ $CALL2_RESULT -ne 0 ] && echo "  - Call #2 failed"
  exit 1
fi

