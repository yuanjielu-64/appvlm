#!/bin/bash

# cleanFile.sh - 清理训练日志和缓存文件

echo "=== 开始清理文件 ==="

# 获取当前目录
CURRENT_DIR=$(pwd)
echo "当前目录: $CURRENT_DIR"

# 清理 logging 目录中的 adaptive_dynamics_planning_v0
if [ -d "logging/adaptive_dynamics_planning_v0" ]; then
    echo "正在删除: logging/adaptive_dynamics_planning_v0"
    rm -rf logging/adaptive_dynamics_planning_v0
    echo "✓ 已删除 adaptive_dynamics_planning_v0 日志目录"
else
    echo "⚠ 未找到目录: logging/adaptive_dynamics_planning_v0"
fi

# 清理 buffer 目录中所有包含 "MPPI" 的目录
if [ -d "buffer" ]; then
    echo "正在搜索包含 'ddp' 的目录..."
    
    # 使用 find 命令查找所有包含 MPPI 的目录
    ddp_dirs=$(find buffer -type d -name "*ddp*" 2>/dev/null)
    
    if [ -n "$ddp_dirs" ]; then
        echo "找到以下包含 'ddp' 的目录:"
        echo "$ddp_dirs"
        
        # 删除所有找到的包含 MPPI 的目录
        echo "$ddp_dirs" | while read -r dir; do
            if [ -n "$dir" ]; then
                echo "正在删除: $dir"
                rm -rf "$dir"
                echo "✓ 已删除 $dir"
            fi
        done
    else
        echo "⚠ 在 buffer 目录中未找到包含 'ddp' 的目录"
    fi
else
    echo "⚠ 未找到目录: buffer"
fi

# 清理 report 目录中的所有文件
if [ -d "executable/cpu_report" ]; then
    echo "正在清理: report 目录"
    rm -rf executable/cpu_report/*
    echo "✓ 已清理 report 目录"
else
    echo "⚠ 未找到目录: report"
fi

if [ -d "executable/gpu_report" ]; then
    echo "正在清理: report 目录"
    rm -rf executable/gpu_report/*
    echo "✓ 已清理 report 目录"
else
    echo "⚠ 未找到目录: report"
fi
# 显示清理后的目录状态
if [ -d "logging" ]; then
    echo "正在清理 logging 目录中的 adaptive_dynamics_planning 相关目录..."
    
    # 要删除的目录列表
    dirs_to_delete=(
        "dwa_param-v0"
        "mppi_param-v0"
        "teb_param-v0"
    )
    
    for dir in "${dirs_to_delete[@]}"; do
        if [ -d "logging/$dir" ]; then
            echo "正在删除: logging/$dir"
            rm -rf "logging/$dir"
            echo "✓ 已删除 $dir"
        else
            echo "⚠ 未找到目录: logging/$dir"
        fi
    done
else
    echo "⚠ 未找到目录: logging"
fi

echo ""
if [ -d "buffer" ]; then
    echo "buffer 目录内容:"
    ls -la buffer/ 2>/dev/null || echo "  (空目录)"
else
    echo "buffer 目录不存在"
fi

echo ""
echo "=== 清理完成 ==="