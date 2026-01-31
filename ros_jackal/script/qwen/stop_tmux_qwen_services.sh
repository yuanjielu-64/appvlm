#!/bin/bash
# 关闭 Qwen VLM 服务的 tmux 会话
# 对应 tmux_qwen_services.sh 的关闭脚本
#
# 用法:
#   ./stop_tmux_qwen_services.sh           # 关闭所有服务
#   ./stop_tmux_qwen_services.sh ddp_2500  # 只关闭指定服务
#   ./stop_tmux_qwen_services.sh dwa       # 关闭所有dwa相关服务

# 定义所有服务的 tmux 会话名称（与启动脚本保持一致）
declare -a ALL_SESSIONS=(
    "ddp_2500"
    "ddp_5000"
    "dwa_2500"
    "dwa_5000"
    "teb_2500"
    "teb_5000"
    "mppi_2500"
    "mppi_5000"
    "mppi_7500"
)

# 显示用法
show_usage() {
    echo "用法: $0 [SESSION_NAME|PATTERN]"
    echo ""
    echo "参数:"
    echo "  无参数          - 关闭所有服务"
    echo "  SESSION_NAME    - 关闭指定的tmux会话 (如: ddp_2500)"
    echo "  PATTERN         - 关闭匹配的会话 (如: dwa, 会关闭dwa_2500和dwa_5000)"
    echo ""
    echo "可用的会话:"
    for session in "${ALL_SESSIONS[@]}"; do
        echo "  - ${session}"
    done
    echo ""
}

# 停止单个会话的函数
stop_session() {
    local session=$1

    # 检查 tmux 会话是否存在
    if tmux has-session -t "${session}" 2>/dev/null; then
        # 发送 Ctrl+C 停止服务（优雅关闭）
        tmux send-keys -t "${session}" C-c
        sleep 1

        # 杀死 tmux 会话
        tmux kill-session -t "${session}"
        echo "  [OK] Stopped and killed session '${session}'"
        return 0
    else
        echo "  [SKIP] Session '${session}' not found"
        return 1
    fi
}

# 解析参数
TARGET="${1:-all}"

if [ "$TARGET" == "-h" ] || [ "$TARGET" == "--help" ]; then
    show_usage
    exit 0
fi

echo "=================================================="
if [ "$TARGET" == "all" ]; then
    echo "  Stopping All Qwen VLM Services"
else
    echo "  Stopping Qwen VLM Services: ${TARGET}"
fi
echo "=================================================="

STOPPED_COUNT=0
NOT_FOUND_COUNT=0

# 根据参数决定关闭哪些会话
if [ "$TARGET" == "all" ]; then
    # 关闭所有服务
    SESSIONS_TO_STOP=("${ALL_SESSIONS[@]}")
else
    # 检查是精确匹配还是模式匹配
    SESSIONS_TO_STOP=()

    # 先检查是否是精确匹配
    EXACT_MATCH=false
    for session in "${ALL_SESSIONS[@]}"; do
        if [ "$session" == "$TARGET" ]; then
            EXACT_MATCH=true
            SESSIONS_TO_STOP+=("$session")
            break
        fi
    done

    # 如果不是精确匹配，尝试模式匹配
    if [ "$EXACT_MATCH" == false ]; then
        for session in "${ALL_SESSIONS[@]}"; do
            if [[ "$session" == *"$TARGET"* ]]; then
                SESSIONS_TO_STOP+=("$session")
            fi
        done
    fi

    # 如果没有匹配到任何会话
    if [ ${#SESSIONS_TO_STOP[@]} -eq 0 ]; then
        echo ""
        echo "  [ERROR] No sessions matching '${TARGET}'"
        echo ""
        show_usage
        exit 1
    fi
fi

# 停止所有匹配的会话
for session in "${SESSIONS_TO_STOP[@]}"; do
    echo ""
    echo "Checking tmux session: ${session}..."

    if stop_session "$session"; then
        ((STOPPED_COUNT++))
    else
        ((NOT_FOUND_COUNT++))
    fi
done

echo ""
echo "=================================================="
echo "  Summary"
echo "=================================================="
echo "  Stopped: ${STOPPED_COUNT}"
echo "  Not found: ${NOT_FOUND_COUNT}"
echo ""

# 额外清理：检查是否还有 qwen_server 进程残留
echo "Checking for any remaining qwen_server processes..."
REMAINING=$(ps aux | grep -E "qwen_server|start_qwen_service" | grep -v grep | wc -l)

if [ "$REMAINING" -gt 0 ]; then
    echo "  [WARNING] Found ${REMAINING} remaining processes:"
    ps aux | grep -E "qwen_server|start_qwen_service" | grep -v grep
    echo ""
    read -p "Kill these processes? [y/N] " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        pkill -f "qwen_server"
        pkill -f "start_qwen_service"
        echo "  [OK] Processes killed"
    fi
else
    echo "  [OK] No remaining processes found"
fi

echo ""
echo "All services stopped!"
echo ""
