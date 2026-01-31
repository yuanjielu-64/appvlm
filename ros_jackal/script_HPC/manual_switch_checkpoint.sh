#!/bin/bash
# 手动切换 Qwen VLM 服务的 checkpoint
#
# 用法:
#   bash manual_switch_checkpoint.sh <checkpoint_path> [options]
#
# 示例:
#   # 使用完整路径
#   bash manual_switch_checkpoint.sh /scratch/.../checkpoint-7500 --host gpu019 --port 5001 --alg DWA
#
#   # 使用简化方式（planner + checkpoint 编号）
#   bash manual_switch_checkpoint.sh DWA 7500 --host gpu019 --port 5001
#
#   # 自动检测服务地址
#   bash manual_switch_checkpoint.sh DWA 10000

set -e

# ============================================================
# 默认配置
# ============================================================
QWEN_HOST="${QWEN_HOST:-}"
QWEN_PORT="${QWEN_PORT:-}"
ALGORITHM=""
NUM_PARAMS=""
HEAD_TYPE="dpt"

# ============================================================
# 帮助信息
# ============================================================
show_help() {
    cat << EOF
手动切换 Qwen VLM Checkpoint

用法:
  $0 <checkpoint_path> [options]
  $0 <planner> <checkpoint_number> [options]

参数:
  checkpoint_path     完整的 checkpoint 路径
  planner            算法名称 (DWA/TEB/MPPI/DDP)
  checkpoint_number  checkpoint 编号 (如 7500)

选项:
  --host HOST        Qwen 服务节点 (如 gpu019)
  --port PORT        服务端口 (如 5001)
  --alg ALGORITHM    算法名称 (DWA/TEB/MPPI/DDP，可选)
  --num NUM          参数数量 (默认: DWA=9, TEB=9, MPPI=10, DDP=8)
  --head TYPE        Head 类型 (默认: dpt)
  -h, --help         显示此帮助信息

示例:

  1. 完整路径方式:
     $0 /scratch/bwang25/.../checkpoint-7500 --host gpu019 --port 5001 --alg DWA

  2. 简化方式 (推荐):
     $0 DWA 7500 --host gpu019 --port 5001

  3. 自动检测服务地址:
     $0 DWA 10000

  4. 使用环境变量:
     QWEN_HOST=gpu019 QWEN_PORT=5001 $0 DWA 7500

可用的 checkpoint 目录:
  /scratch/bwang25/appvlm_ws/src/ros_jackal/model/dwa/
  /scratch/bwang25/appvlm_ws/src/ros_jackal/model/teb/
  /scratch/bwang25/appvlm_ws/src/ros_jackal/model/mppi/
  /scratch/bwang25/appvlm_ws/src/ros_jackal/model/ddp/

EOF
}

# ============================================================
# 参数解析
# ============================================================

if [ $# -eq 0 ]; then
    show_help
    exit 1
fi

# 检查是否是帮助请求
if [[ "$1" == "-h" ]] || [[ "$1" == "--help" ]]; then
    show_help
    exit 0
fi

# 第一个参数：checkpoint 路径或 planner 名称
FIRST_ARG="$1"
shift

# 判断是完整路径还是 planner 名称
if [[ "$FIRST_ARG" == /* ]] || [[ "$FIRST_ARG" == ./* ]]; then
    # 完整路径
    CHECKPOINT_PATH="$FIRST_ARG"
    echo "Using full checkpoint path: $CHECKPOINT_PATH"
elif [[ "$FIRST_ARG" =~ ^(DWA|TEB|MPPI|DDP|dwa|teb|mppi|ddp)$ ]]; then
    # Planner 名称 - 期望第二个参数是 checkpoint 编号
    if [ $# -lt 1 ]; then
        echo "❌ Error: Checkpoint number required after planner name"
        echo "Usage: $0 $FIRST_ARG <checkpoint_number> [options]"
        exit 1
    fi

    PLANNER=$(echo "$FIRST_ARG" | tr '[:lower:]' '[:upper:]')
    CHECKPOINT_NUM="$1"
    shift

    # 设置默认算法和参数数量
    ALGORITHM="$PLANNER"
    case $PLANNER in
        DWA)  NUM_PARAMS=9 ;;
        TEB)  NUM_PARAMS=9 ;;
        MPPI) NUM_PARAMS=10 ;;
        DDP)  NUM_PARAMS=8 ;;
    esac

    # 构建 checkpoint 路径 - 尝试多个可能的路径
    PLANNER_LOWER=$(echo "$PLANNER" | tr '[:upper:]' '[:lower:]')
    BASE_DIR="/scratch/bwang25/appvlm_ws/src/ros_jackal/model/${PLANNER_LOWER}"

    # 尝试的路径列表（按优先级）
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
        echo "❌ Error: Checkpoint not found for $PLANNER checkpoint-$CHECKPOINT_NUM"
        echo ""
        echo "Searched in:"
        for path in "${POSSIBLE_PATHS[@]}"; do
            echo "  - $path"
        done
        echo ""
        echo "Available model directories:"
        ls -d ${BASE_DIR}/qwen2.5-vl-regression_lora-True_${PLANNER_LOWER}_regression* 2>/dev/null || echo "  (none found)"
        exit 1
    fi

    echo "Using shorthand: $PLANNER checkpoint-$CHECKPOINT_NUM"
    echo "Resolved path: $CHECKPOINT_PATH"
else
    echo "❌ Error: Invalid first argument: $FIRST_ARG"
    echo "Expected: checkpoint path or planner name (DWA/TEB/MPPI/DDP)"
    show_help
    exit 1
fi

# 解析其他选项
while [ $# -gt 0 ]; do
    case $1 in
        --host)
            QWEN_HOST="$2"
            shift 2
            ;;
        --port)
            QWEN_PORT="$2"
            shift 2
            ;;
        --alg)
            ALGORITHM=$(echo "$2" | tr '[:lower:]' '[:upper:]')
            shift 2
            ;;
        --num)
            NUM_PARAMS="$2"
            shift 2
            ;;
        --head)
            HEAD_TYPE="$2"
            shift 2
            ;;
        *)
            echo "❌ Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

# ============================================================
# 验证 checkpoint 路径
# ============================================================

if [ ! -d "$CHECKPOINT_PATH" ]; then
    echo "❌ Error: Checkpoint directory not found: $CHECKPOINT_PATH"
    echo ""
    echo "Available checkpoints in model directory:"
    find /scratch/bwang25/appvlm_ws/src/ros_jackal/model -type d -name "checkpoint-*" | head -10
    exit 1
fi

echo "✓ Checkpoint exists: $CHECKPOINT_PATH"

# ============================================================
# 自动检测服务地址（如果未提供）
# ============================================================

if [ -z "$QWEN_HOST" ] || [ -z "$QWEN_PORT" ]; then
    echo ""
    echo "🔍 Auto-detecting service address..."

    # 从算法推断，查找对应的日志
    if [ -n "$ALGORITHM" ]; then
        LATEST_LOG=$(ls -t /scratch/ylu22/appvlm_ws/src/ros_jackal/cpu_report*/qwen_${ALGORITHM}-*.out 2>/dev/null | head -1)

        if [ -n "$LATEST_LOG" ]; then
            DETECTED_HOST=$(grep "QWEN_HOST=" "$LATEST_LOG" | tail -1 | sed 's/.*QWEN_HOST="\([^"]*\)".*/\1/')
            DETECTED_PORT=$(grep "Port:" "$LATEST_LOG" | tail -1 | awk '{print $NF}')

            [ -z "$QWEN_HOST" ] && QWEN_HOST="$DETECTED_HOST"
            [ -z "$QWEN_PORT" ] && QWEN_PORT="$DETECTED_PORT"

            echo "  Detected from $ALGORITHM log: $QWEN_HOST:$QWEN_PORT"
        fi
    fi

    # 如果还是没有，提示用户
    if [ -z "$QWEN_HOST" ] || [ -z "$QWEN_PORT" ]; then
        echo "❌ Error: Could not auto-detect service address"
        echo ""
        echo "Please specify manually:"
        echo "  $0 $CHECKPOINT_PATH --host <hostname> --port <port>"
        echo ""
        echo "Or check running services:"
        echo "  squeue -u \$USER | grep qwen"
        echo "  tail cpu_report*/qwen_*.out | grep QWEN_HOST"
        exit 1
    fi
fi

# ============================================================
# 从路径推断算法（如果未提供）
# ============================================================

if [ -z "$ALGORITHM" ]; then
    if [[ "$CHECKPOINT_PATH" =~ /dwa/ ]]; then
        ALGORITHM="DWA"
        NUM_PARAMS=7
    elif [[ "$CHECKPOINT_PATH" =~ /teb/ ]]; then
        ALGORITHM="TEB"
        NUM_PARAMS=7
    elif [[ "$CHECKPOINT_PATH" =~ /mppi/ ]]; then
        ALGORITHM="MPPI"
        NUM_PARAMS=8
    elif [[ "$CHECKPOINT_PATH" =~ /ddp/ ]]; then
        ALGORITHM="DDP"
        NUM_PARAMS=6
    else
        echo "❌ Error: Could not detect algorithm from path"
        echo "Please specify with --alg option"
        exit 1
    fi
    echo "  Detected algorithm from path: $ALGORITHM"
fi

# ============================================================
# 显示配置总结
# ============================================================

echo ""
echo "=========================================="
echo "🔄 Switching Checkpoint"
echo "=========================================="
echo "Service:    http://${QWEN_HOST}:${QWEN_PORT}"
echo "Checkpoint: $CHECKPOINT_PATH"
echo "Algorithm:  $ALGORITHM"
echo "Head Type:  $HEAD_TYPE"
echo "Num Params: $NUM_PARAMS"
echo "=========================================="
echo ""

# ============================================================
# 执行切换
# ============================================================

echo "Sending switch request..."
RESPONSE=$(curl -s -X POST http://${QWEN_HOST}:${QWEN_PORT}/switch_checkpoint \
  -H "Content-Type: application/json" \
  -d "{
    \"checkpoint_path\": \"${CHECKPOINT_PATH}\",
    \"algorithm\": \"${ALGORITHM}\",
    \"head_type\": \"${HEAD_TYPE}\",
    \"num_params\": ${NUM_PARAMS}
  }")

# 检查响应
if [ $? -ne 0 ]; then
    echo "❌ Failed to connect to service"
    echo "Please check if service is running:"
    echo "  squeue -u \$USER | grep qwen"
    exit 1
fi

echo ""
echo "Response:"
echo "$RESPONSE" | python3 -m json.tool 2>/dev/null || echo "$RESPONSE"

# 检查是否成功
SUCCESS=$(echo "$RESPONSE" | python3 -c "import sys, json; print(json.load(sys.stdin).get('success', False))" 2>/dev/null || echo "false")

echo ""
if [ "$SUCCESS" == "True" ]; then
    echo "✅ Checkpoint switched successfully!"

    # 显示当前状态
    echo ""
    echo "=========================================="
    echo "Current Service Status:"
    echo "=========================================="
    curl -s http://${QWEN_HOST}:${QWEN_PORT}/health | python3 -m json.tool 2>/dev/null
else
    echo "❌ Failed to switch checkpoint"
    MESSAGE=$(echo "$RESPONSE" | python3 -c "import sys, json; print(json.load(sys.stdin).get('message', 'Unknown error'))" 2>/dev/null || echo "Unknown error")
    echo "Error: $MESSAGE"
    exit 1
fi

echo ""
