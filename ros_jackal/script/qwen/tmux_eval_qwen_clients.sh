#!/bin/bash
# 根据 planner 类型启动评估 tmux 会话
#
# 用法:
#   bash start_eval_sessions.sh --planner ddp    # 启动 DDP 的 2 个评估会话
#   bash start_eval_sessions.sh --planner mppi   # 启动 MPPI 的 3 个评估会话
#   bash start_eval_sessions.sh --planner dwa    # 启动 DWA 的 2 个评估会话
#   bash start_eval_sessions.sh --planner teb    # 启动 TEB 的 2 个评估会话
#   bash start_eval_sessions.sh --planner all    # 启动所有评估会话
#   bash start_eval_sessions.sh --planner ddp --start 0 --end 99 --runs 3



set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROS_JACKAL_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

# 解析参数
PLANNER=""
START_WORLD=0
END_WORLD=299
RUNS=3

while [[ $# -gt 0 ]]; do
  case $1 in
    --planner)
      PLANNER="$2"
      shift 2
      ;;
    --start)
      START_WORLD="$2"
      shift 2
      ;;
    --end)
      END_WORLD="$2"
      shift 2
      ;;
    --runs)
      RUNS="$2"
      shift 2
      ;;
    *)
      echo "未知参数: $1"
      echo ""
      echo "用法: $0 --planner PLANNER [--start N] [--end N] [--runs N]"
      echo ""
      echo "Planner 选项:"
      echo "  ddp   - 启动 2 个会话 (ddp_2500, ddp_5000)"
      echo "  dwa   - 启动 2 个会话 (dwa_2500, dwa_5000)"
      echo "  teb   - 启动 2 个会话 (teb_2500, teb_5000)"
      echo "  mppi  - 启动 3 个会话 (mppi_2500, mppi_5000, mppi_7500)"
      echo "  all   - 启动所有 9 个会话"
      echo ""
      echo "示例:"
      echo "  $0 --planner ddp"
      echo "  $0 --planner mppi --start 0 --end 99"
      exit 1
      ;;
  esac
done

if [ -z "$PLANNER" ]; then
    echo "错误: 必须指定 --planner 参数"
    echo "用法: $0 --planner {ddp|dwa|teb|mppi|all}"
    exit 1
fi

# 定义所有评估配置
# 格式: "tmux名称:容器ID:Qwen端口:策略名称"
declare -a DDP_SESSIONS=(
    "s_ddp_2500:1:5000:ddp_qwen"
    "s_ddp_5000:2:5001:ddp_qwen"
)

declare -a DWA_SESSIONS=(
    "s_dwa_2500:3:5002:dwa_qwen"
    "s_dwa_5000:4:5003:dwa_qwen"
)

declare -a TEB_SESSIONS=(
    "s_teb_2500:5:5004:teb_qwen"
    "s_teb_5000:6:5005:teb_qwen"
)

declare -a MPPI_SESSIONS=(
    #"s_mppi_2500:7:5006:mppi_qwen"
    #"s_mppi_5000:8:5007:mppi_qwen"
    "s_mppi_7500:9:5008:mppi_qwen"
)

# 根据 planner 选择要启动的会话
declare -a SESSIONS_TO_START=()

case "${PLANNER,,}" in  # 转小写
    ddp)
        SESSIONS_TO_START=("${DDP_SESSIONS[@]}")
        ;;
    dwa)
        SESSIONS_TO_START=("${DWA_SESSIONS[@]}")
        ;;
    teb)
        SESSIONS_TO_START=("${TEB_SESSIONS[@]}")
        ;;
    mppi)
        SESSIONS_TO_START=("${MPPI_SESSIONS[@]}")
        ;;
    all)
        SESSIONS_TO_START=(
            "${DDP_SESSIONS[@]}"
            "${DWA_SESSIONS[@]}"
            "${TEB_SESSIONS[@]}"
            "${MPPI_SESSIONS[@]}"
        )
        ;;
    *)
        echo "错误: 未知的 planner '$PLANNER'"
        echo "可选值: ddp, dwa, teb, mppi, all"
        exit 1
        ;;
esac

echo "=================================================="
echo "  启动评估会话 - Planner: ${PLANNER^^}"
echo "=================================================="
echo "环境范围:    $START_WORLD - $END_WORLD"
echo "每环境运行:  $RUNS 次"
echo "会话数量:    ${#SESSIONS_TO_START[@]}"
echo "=================================================="
echo ""

# 启动所有会话
for session in "${SESSIONS_TO_START[@]}"; do
    IFS=':' read -r TMUX_NAME CONTAINER_ID QWEN_PORT POLICY_NAME <<< "$session"

    echo "启动 ${TMUX_NAME}..."
    echo "  容器ID: ${CONTAINER_ID}, Qwen端口: ${QWEN_PORT}, 策略: ${POLICY_NAME}"

    # 检查 tmux 会话是否已存在
    if tmux has-session -t "${TMUX_NAME}" 2>/dev/null; then
        echo "  [SKIP] tmux 会话 '${TMUX_NAME}' 已存在"
        continue
    fi

    # 创建 tmux 会话并运行批量评估
    tmux new-session -d -s "${TMUX_NAME}" \
        "cd ${ROS_JACKAL_DIR} && bash script/qwen/eval_batch_worlds_singularity.sh \
            --id ${CONTAINER_ID} \
            --qwen_port ${QWEN_PORT} \
            --policy ${POLICY_NAME} \
            --start ${START_WORLD} \
            --end ${END_WORLD} \
            --runs ${RUNS}; \
         echo ''; \
         echo '评估完成。按 Enter 退出。'; \
         read"

    echo "  [OK] 已在 tmux '${TMUX_NAME}' 中启动"

    # 等待 Gazebo 完成初始化，避免资源竞争
    # 第一个 Gazebo 启动需要约 30 秒加载
    echo "  等待 10 秒让 Gazebo 初始化..."
    sleep 10
done

echo ""
echo "=================================================="
echo "  所有评估会话已启动!"
echo "=================================================="
echo ""
echo "查看所有 tmux 会话:  tmux ls"
echo "进入某个会话:        tmux attach -t s_ddp_2500"
echo "退出会话(不停止):    Ctrl+b d"
echo ""
echo "停止所有评估会话:"
echo "  tmux kill-session -t s_ddp_2500"
echo "  或批量: for s in \$(tmux ls -F '#{session_name}' | grep '^s_'); do tmux kill-session -t \$s; done"
echo ""
