#!/bin/bash
# 强制关闭所有匹配 s_*_* 模式的 tmux 会话（无需确认）
# 用法: bash kill_tmux_force.sh [pattern]
# 示例:
#   bash kill_tmux_force.sh           # 关闭所有 s_*_* 格式的会话
#   bash kill_tmux_force.sh teb       # 只关闭 s_teb_* 会话
#   bash kill_tmux_force.sh dwa       # 只关闭 s_dwa_* 会话

PATTERN="${1:-}"

echo "=================================================="
echo "  强制清理 tmux 会话 (模式: s_*_*)"
echo "=================================================="
echo ""

# 检查 tmux 是否安装
if ! command -v tmux &> /dev/null; then
    echo "❌ tmux 未安装"
    exit 1
fi

# 获取所有 tmux 会话列表
sessions=$(tmux list-sessions -F "#{session_name}" 2>/dev/null)

if [ -z "$sessions" ]; then
    echo "ℹ️  没有运行中的 tmux 会话"
    exit 0
fi

# 过滤出匹配 s_*_* 模式的会话 (例如: s_teb_2500, s_dwa_5000 等)
if [ -n "$PATTERN" ]; then
    # 如果提供了 pattern，只匹配特定 planner
    matching_sessions=$(echo "$sessions" | grep -E "^s_${PATTERN}_[0-9]+$")
    echo "过滤器: s_${PATTERN}_*"
else
    # 匹配所有 s_*_* 格式
    matching_sessions=$(echo "$sessions" | grep -E "^s_[a-z]+_[0-9]+$")
    echo "过滤器: s_*_*"
fi
echo ""

if [ -z "$matching_sessions" ]; then
    echo "ℹ️  没有找到匹配的会话"
    exit 0
fi

# 统计数量
count=$(echo "$matching_sessions" | wc -l)
echo "找到 $count 个匹配的会话，开始关闭..."
echo ""

# 直接关闭所有匹配的会话
killed_count=0
failed_count=0

while IFS= read -r session_name; do
    if [ -n "$session_name" ]; then
        echo "  ✗ $session_name"
        if tmux kill-session -t "$session_name" 2>/dev/null; then
            ((killed_count++))
        else
            ((failed_count++))
            echo "    ❌ 失败"
        fi
    fi
done <<< "$matching_sessions"

echo ""
echo "=================================================="
echo "✅ 完成: 成功关闭 $killed_count/$count 个会话"
if [ $failed_count -gt 0 ]; then
    echo "⚠️  失败: $failed_count 个会话"
fi
echo "=================================================="
