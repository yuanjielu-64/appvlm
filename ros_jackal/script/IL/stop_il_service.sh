#!/bin/bash

# 停止所有 IL 相关的 tmux 会话
# 包括: IL_server_* (服务端) 和 il_* (评估客户端)

echo "=========================================="
echo "  停止 IL 相关会话"
echo "=========================================="

# 1. 停止 IL 服务端 (IL_server_*)
echo ""
echo "停止 IL 服务端..."
for planner in dwa teb mppi ddp; do
    session_name="IL_server_${planner}"
    if tmux has-session -t "$session_name" 2>/dev/null; then
        tmux kill-session -t "$session_name"
        echo "  [OK] $session_name"
    fi
done

# 2. 停止 IL 评估客户端 (il_*)
echo ""
echo "停止 IL 评估客户端..."
# 获取所有以 il_ 开头的会话
IL_CLIENTS=$(tmux ls -F '#{session_name}' 2>/dev/null | grep -E '^il_' || true)

if [ -n "$IL_CLIENTS" ]; then
    for session in $IL_CLIENTS; do
        tmux kill-session -t "$session"
        echo "  [OK] $session"
    done
else
    echo "  没有运行中的 IL 评估客户端"
fi

echo ""
echo "=========================================="
echo "  所有 IL 会话已停止"
echo "=========================================="
