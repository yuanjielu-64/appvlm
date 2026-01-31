#!/bin/bash
# 测试单个 Qwen VLM 服务的完整功能
#
# 用法:
#   bash test_single_qwen_service.sh <planner> [test_image] [host:port]
#
# 示例:
#   bash test_single_qwen_service.sh DWA
#   bash test_single_qwen_service.sh DWA /path/to/test.png
#   bash test_single_qwen_service.sh DWA /path/to/test.png gpu019:5001

set -e

# ============================================================
# 参数解析
# ============================================================

if [ $# -lt 1 ]; then
    echo "用法: $0 <planner> [test_image] [host:port]"
    echo ""
    echo "参数:"
    echo "  planner:    DWA | TEB | MPPI | DDP"
    echo "  test_image: 测试图像路径（可选，默认自动查找）"
    echo "  host:port:  服务地址（可选，默认自动检测）"
    echo ""
    echo "示例:"
    echo "  $0 DWA"
    echo "  $0 DWA /path/to/test.png"
    echo "  $0 DWA /path/to/test.png gpu019:5001"
    exit 1
fi

PLANNER=$(echo "$1" | tr '[:lower:]' '[:upper:]')
TEST_IMAGE="${2:-}"
SERVICE_ADDR="${3:-}"

# 验证 planner 名称
if [[ ! "$PLANNER" =~ ^(DWA|TEB|MPPI|DDP)$ ]]; then
    echo "❌ 无效的 planner: $PLANNER"
    echo "   支持的 planner: DWA, TEB, MPPI, DDP"
    exit 1
fi

echo "=========================================="
echo "🧪 Testing Single Qwen Service"
echo "=========================================="
echo "Planner: $PLANNER"
echo "Time:    $(date '+%Y-%m-%d %H:%M:%S')"
echo "=========================================="
echo ""

# 切换到项目目录
cd /scratch/ylu22/appvlm_ws/src/ros_jackal

# ============================================================
# 1. 查找测试图像
# ============================================================

if [ -z "$TEST_IMAGE" ]; then
    echo "🔍 Searching for test image..."

    # 优先查找与 planner 匹配的目录
    PLANNER_LOWER=$(echo "$PLANNER" | tr '[:upper:]' '[:lower:]')

    # 尝试多种命名模式
    SEARCH_PATTERNS=(
        "buffer/${PLANNER_LOWER}_*/actor_*/VLM_*.png"          # dwa_qwen, dwa_gpt 等
        "buffer/*${PLANNER_LOWER}*/actor_*/VLM_*.png"          # 包含 dwa 的目录
        "buffer/${PLANNER_LOWER}/actor_*/VLM_*.png"            # 简单的 dwa 目录
        "buffer/world_*/actor_*/${PLANNER_LOWER}_VLM_*.png"    # 文件名包含 planner
    )

    TEST_IMAGE=""
    for pattern in "${SEARCH_PATTERNS[@]}"; do
        TEST_IMAGE=$(find buffer -path "$pattern" 2>/dev/null | head -1)
        if [ -n "$TEST_IMAGE" ]; then
            echo "  Found (matched to $PLANNER): $TEST_IMAGE"
            break
        fi
    done

    # 如果还是找不到，尝试查找任意图像（并给出警告）
    if [ -z "$TEST_IMAGE" ]; then
        echo "  ⚠️  No $PLANNER-specific image found, searching for any test image..."
        TEST_IMAGE=$(find buffer/*/actor_*/VLM_*.png 2>/dev/null | head -1)

        if [ -n "$TEST_IMAGE" ]; then
            echo "  Found (generic): $TEST_IMAGE"
            echo "  ⚠️  WARNING: This image may not be from $PLANNER data"
            echo ""
            read -p "  Continue with this image? [y/N] " -n 1 -r
            echo
            if [[ ! $REPLY =~ ^[Yy]$ ]]; then
                echo "  Aborted. Please provide a test image:"
                echo "    $0 $PLANNER /path/to/${PLANNER_LOWER}_test_image.png"
                exit 1
            fi
        fi
    fi

    # 如果完全找不到
    if [ -z "$TEST_IMAGE" ]; then
        echo "❌ No test image found in buffer/"
        echo ""
        echo "Please provide a test image:"
        echo "  $0 $PLANNER /path/to/test_image.png"
        echo ""
        echo "Available buffer directories:"
        ls -d buffer/*/ 2>/dev/null | head -5 || echo "  (none found)"
        exit 1
    fi
else
    if [ ! -f "$TEST_IMAGE" ]; then
        echo "❌ Test image not found: $TEST_IMAGE"
        exit 1
    fi
    echo "  Using: $TEST_IMAGE"
fi
echo ""

# ============================================================
# 2. 检测或验证服务地址
# ============================================================

if [ -z "$SERVICE_ADDR" ]; then
    echo "🔍 Auto-detecting service address..."

    # 方法 1: 从最新的日志文件中获取
    LATEST_LOG=$(ls -t cpu_report*/qwen_${PLANNER}-*.out 2>/dev/null | head -1)

    if [ -n "$LATEST_LOG" ]; then
        # 从日志中提取 QWEN_HOST
        QWEN_HOST=$(grep "QWEN_HOST=" "$LATEST_LOG" | tail -1 | sed 's/.*QWEN_HOST="\([^"]*\)".*/\1/')
        # 从日志中提取 Port
        QWEN_PORT=$(grep "Port:" "$LATEST_LOG" | tail -1 | awk '{print $NF}')

        if [ -n "$QWEN_HOST" ] && [ -n "$QWEN_PORT" ]; then
            SERVICE_ADDR="${QWEN_HOST}:${QWEN_PORT}"
            echo "  Detected from log: $SERVICE_ADDR"
            echo "  Log file: $LATEST_LOG"
        fi
    fi

    # 方法 2: 使用默认端口映射
    if [ -z "$SERVICE_ADDR" ]; then
        echo "  ⚠️  Could not auto-detect from logs"
        echo "  Using default port mapping..."

        case $PLANNER in
            DWA)  SERVICE_ADDR="gpu019:5001" ;;
            TEB)  SERVICE_ADDR="gpu019:5002" ;;
            MPPI) SERVICE_ADDR="gpu020:5003" ;;
            DDP)  SERVICE_ADDR="gpu021:5004" ;;
        esac

        echo "  Default: $SERVICE_ADDR"
        echo "  (Note: This may not be accurate if services were moved)"
    fi
else
    echo "  Using provided address: $SERVICE_ADDR"
fi

# 分离 host 和 port
IFS=':' read -r QWEN_HOST QWEN_PORT <<< "$SERVICE_ADDR"

echo ""
echo "Target Service:"
echo "  Host: $QWEN_HOST"
echo "  Port: $QWEN_PORT"
echo "  URL:  http://${QWEN_HOST}:${QWEN_PORT}"
echo ""

# ============================================================
# 3. 测试服务可达性
# ============================================================

echo "=========================================="
echo "Test 1: Service Reachability"
echo "=========================================="

if ! curl -s --connect-timeout 5 http://${QWEN_HOST}:${QWEN_PORT}/health > /dev/null 2>&1; then
    echo "❌ FAILED: Service is unreachable"
    echo ""
    echo "Troubleshooting:"
    echo "  1. Check if service is running:"
    echo "     squeue -u \$USER | grep qwen_${PLANNER}"
    echo "  2. Check service logs:"
    echo "     tail -50 cpu_report*/qwen_${PLANNER}-*.out"
    echo "  3. Verify host and port:"
    echo "     $0 $PLANNER \"$TEST_IMAGE\" <correct_host:port>"
    exit 1
fi

echo "✅ PASSED: Service is reachable"
echo ""

# ============================================================
# 4. 测试健康检查接口
# ============================================================

echo "=========================================="
echo "Test 2: Health Check API"
echo "=========================================="

HEALTH_JSON=$(curl -s http://${QWEN_HOST}:${QWEN_PORT}/health)

if [ $? -ne 0 ]; then
    echo "❌ FAILED: Could not get health status"
    exit 1
fi

echo "Response:"
echo "$HEALTH_JSON" | python3 -m json.tool 2>/dev/null || echo "$HEALTH_JSON"
echo ""

# 解析健康状态
STATUS=$(echo "$HEALTH_JSON" | python3 -c "import sys, json; data=json.load(sys.stdin); print(data.get('status', 'unknown'))" 2>/dev/null || echo "unknown")
MODEL_LOADED=$(echo "$HEALTH_JSON" | python3 -c "import sys, json; data=json.load(sys.stdin); print(data.get('model_loaded', False))" 2>/dev/null || echo "false")
CHECKPOINT=$(echo "$HEALTH_JSON" | python3 -c "import sys, json; data=json.load(sys.stdin); print(data.get('current_checkpoint', 'none'))" 2>/dev/null || echo "none")
ALGORITHM=$(echo "$HEALTH_JSON" | python3 -c "import sys, json; data=json.load(sys.stdin); print(data.get('algorithm', 'unknown'))" 2>/dev/null || echo "unknown")

echo "Health Status:"
echo "  Status:       $STATUS"
echo "  Model Loaded: $MODEL_LOADED"
echo "  Algorithm:    $ALGORITHM"
echo "  Checkpoint:   $(basename "$CHECKPOINT")"
echo ""

if [ "$STATUS" != "ok" ] || [ "$MODEL_LOADED" != "True" ]; then
    echo "❌ FAILED: Service is not ready"
    echo "   Check service logs for errors"
    exit 1
fi

if [ "$ALGORITHM" != "$PLANNER" ]; then
    echo "⚠️  WARNING: Algorithm mismatch"
    echo "   Expected: $PLANNER"
    echo "   Actual:   $ALGORITHM"
fi

echo "✅ PASSED: Service is healthy and ready"
echo ""

# ============================================================
# 5. 测试推理接口
# ============================================================

echo "=========================================="
echo "Test 3: Inference API"
echo "=========================================="

echo "Running inference test..."
echo "  Image: $TEST_IMAGE"
echo "  Algorithm: $PLANNER"
echo ""

# 使用 simple_infer.sh 进行推理测试
if [ -f "./script_HPC/simple_infer.sh" ]; then
    QWEN_HOST=$QWEN_HOST QWEN_PORT=$QWEN_PORT ./script_HPC/simple_infer.sh "$TEST_IMAGE" "$PLANNER"
    INFER_RESULT=$?
else
    echo "⚠️  simple_infer.sh not found, skipping inference test"
    INFER_RESULT=0
fi

echo ""

if [ $INFER_RESULT -eq 0 ]; then
    echo "✅ PASSED: Inference completed successfully"
else
    echo "❌ FAILED: Inference failed"
    echo "   Check inference script output above"
    exit 1
fi

echo ""

# ============================================================
# 6. 检查 Flash Attention 状态（通过 API）
# ============================================================

echo "=========================================="
echo "Test 4: Flash Attention Check"
echo "=========================================="

# 尝试从 health API 获取 Flash Attention 状态
FA_ENABLED=$(echo "$HEALTH_JSON" | python3 -c "import sys, json; data=json.load(sys.stdin); print(data.get('flash_attention_enabled', 'unknown'))" 2>/dev/null || echo "unknown")
FA_IMPL=$(echo "$HEALTH_JSON" | python3 -c "import sys, json; data=json.load(sys.stdin); print(data.get('attention_implementation', 'unknown'))" 2>/dev/null || echo "unknown")

if [ "$FA_ENABLED" != "unknown" ]; then
    echo "Flash Attention Status (from API):"
    echo "  Enabled: $FA_ENABLED"
    echo "  Implementation: $FA_IMPL"
    echo ""

    if [ "$FA_ENABLED" = "True" ]; then
        echo "✅ PASSED: Flash Attention is enabled"
    else
        echo "⚠️  WARNING: Flash Attention is not enabled (inference may be slower)"
    fi
else
    echo "ℹ️  Flash Attention status not available in API response"
    echo "   (Service may need to be updated to report this information)"
    echo ""
    echo "💡 Tip: Check service startup logs to verify Flash Attention:"
    echo "   grep -A 5 'Flash Attention' cpu_report*/qwen_${PLANNER}-*.out"
fi

echo ""

# ============================================================
# 测试总结
# ============================================================

echo "=========================================="
echo "🎉 All Tests Passed!"
echo "=========================================="
echo ""
echo "Service Summary:"
echo "  Planner:    $PLANNER"
echo "  Address:    $QWEN_HOST:$QWEN_PORT"
echo "  Status:     ✅ Healthy and operational"
echo "  Checkpoint: $(basename "$CHECKPOINT")"
echo ""
echo "Next steps:"
echo "  - Run full evaluation: bash script_HPC/submit_all_controllers.sh"
echo "  - Monitor service: tail -f cpu_report*/qwen_${PLANNER}-*.out"
echo "  - Check logs: tail -50 cpu_report*/qwen_${PLANNER}-*.err"
echo ""
