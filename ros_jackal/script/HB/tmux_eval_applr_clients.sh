#!/bin/bash
# 根据 planner 类型启动 HB 评估客户端
#
# 用法:
#   bash tmux_eval_applr_clients.sh                                    # 使用tmux启动所有评估客户端 (默认)
#   bash tmux_eval_applr_clients.sh --planner ddp                      # 使用tmux启动 DDP 评估客户端
#   bash tmux_eval_applr_clients.sh --planner ddp --num 3              # 启动 3 个 DDP 评估客户端
#   bash tmux_eval_applr_clients.sh --planner all --num 2              # 每个planner启动 2 个客户端
#   bash tmux_eval_applr_clients.sh --planner ddp --tmux false         # 前台直接运行 DDP (不使用tmux)
#   bash tmux_eval_applr_clients.sh --start 0 --end 99                 # 所有planner评估世界 0-99

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROS_JACKAL_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"  # script/HB/ -> ros_jackal/

# 解析参数
PLANNER="all"  # 默认启动所有 planner
START_WORLD=0
END_WORLD=299
RUNS_PER_WORLD=3
USE_TMUX=true
NUM_INSTANCES=1  # 每个planner启动的实例数

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
      RUNS_PER_WORLD="$2"
      shift 2
      ;;
    --tmux)
      USE_TMUX="$2"
      shift 2
      ;;
    --num)
      NUM_INSTANCES="$2"
      shift 2
      ;;
    *)
      echo "未知参数: $1"
      echo ""
      echo "用法: $0 [--planner PLANNER] [--num N] [--start N] [--end N] [--runs N] [--tmux true/false]"
      echo ""
      echo "Planner 选项 (默认: all):"
      echo "  ddp   - 启动 DDP 评估客户端"
      echo "  dwa   - 启动 DWA 评估客户端"
      echo "  teb   - 启动 TEB 评估客户端"
      echo "  mppi  - 启动 MPPI 评估客户端"
      echo "  all   - 启动所有评估客户端 (默认，仅tmux模式)"
      echo ""
      echo "实例数量 (默认: 1):"
      echo "  --num N  每个planner启动N个并行实例"
      echo "           例如 --planner ddp --num 3 启动3个DDP实例"
      echo "           例如 --planner all --num 2 每个planner启动2个实例"
      echo ""
      echo "Tmux 选项 (默认: true):"
      echo "  true  - 在tmux会话中运行 (默认)"
      echo "  false - 在当前终端前台运行 (需要指定单个planner且num=1)"
      echo ""
      echo "示例:"
      echo "  $0                                      # tmux模式，启动所有planner"
      echo "  $0 --planner ddp                        # tmux模式，只启动DDP"
      echo "  $0 --planner ddp --num 3                # tmux模式，启动3个DDP实例"
      echo "  $0 --planner all --num 2                # tmux模式，每个planner启动2个实例"
      echo "  $0 --planner ddp --tmux false           # 前台运行DDP，不使用tmux"
      echo "  $0 --start 0 --end 99                   # tmux模式，所有planner评估世界 0-99"
      exit 1
      ;;
  esac
done

# 验证参数
if [[ "$USE_TMUX" == "false" ]]; then
    if [[ "$PLANNER" == "all" ]]; then
        echo "❌ 错误: --tmux false 模式必须指定单个 planner"
        echo "请使用: --planner {ddp|dwa|teb|mppi}"
        exit 1
    fi
    if [[ "$NUM_INSTANCES" -gt 1 ]]; then
        echo "❌ 错误: --tmux false 模式不支持多实例 (--num > 1)"
        echo "请使用 tmux 模式启动多个实例"
        exit 1
    fi
fi

if [[ "$NUM_INSTANCES" -lt 1 ]]; then
    echo "❌ 错误: --num 必须 >= 1"
    exit 1
fi

# 定义 planner 基础配置
# 格式: "基础tmux名称:基础容器ID:策略名称"
# 容器ID分配: DDP=10, DWA=11, TEB=12, MPPI=13
# 多实例时: ID = 基础ID + 4*实例索引 (避免冲突)
declare -A PLANNER_CONFIG=(
    ["ddp"]="applr_ddp:10:ddp_heurstic"
    ["dwa"]="applr_dwa:11:dwa_heurstic"
    ["teb"]="applr_teb:12:teb_heurstic"
    ["mppi"]="applr_mppi:13:mppi_heurstic"
)

# 生成指定 planner 的多实例会话
# 参数: planner名称
# 输出: 添加到 SESSIONS_TO_START 数组
generate_sessions() {
    local planner="$1"
    local config="${PLANNER_CONFIG[$planner]}"

    if [[ -z "$config" ]]; then
        echo "错误: 未知的 planner '$planner'"
        return 1
    fi

    IFS=':' read -r BASE_TMUX_NAME BASE_CONTAINER_ID POLICY_NAME <<< "$config"

    for ((i=0; i<NUM_INSTANCES; i++)); do
        local container_id=$((BASE_CONTAINER_ID + 4 * i))
        if [[ "$NUM_INSTANCES" -eq 1 ]]; then
            # 单实例：不加后缀
            local tmux_name="${BASE_TMUX_NAME}"
        else
            # 多实例：加数字后缀
            local tmux_name="${BASE_TMUX_NAME}_${i}"
        fi
        SESSIONS_TO_START+=("${tmux_name}:${container_id}:${POLICY_NAME}")
    done
}

# 根据 planner 选择要启动的会话
declare -a SESSIONS_TO_START=()

case "${PLANNER,,}" in  # 转小写
    ddp)
        generate_sessions "ddp"
        ;;
    dwa)
        generate_sessions "dwa"
        ;;
    teb)
        generate_sessions "teb"
        ;;
    mppi)
        generate_sessions "mppi"
        ;;
    all)
        generate_sessions "ddp"
        generate_sessions "dwa"
        generate_sessions "teb"
        generate_sessions "mppi"
        ;;
    *)
        echo "错误: 未知的 planner '$PLANNER'"
        echo "可选值: ddp, dwa, teb, mppi, all"
        exit 1
        ;;
esac

echo "=================================================="
echo "  启动 HB 评估客户端 - Planner: ${PLANNER^^}"
echo "=================================================="
echo "运行模式:    $([ "$USE_TMUX" == "true" ] && echo "tmux" || echo "前台直接运行")"
echo "实例数量:    每个planner ${NUM_INSTANCES} 个实例"
echo "会话总数:    ${#SESSIONS_TO_START[@]}"
echo "评估范围:    世界 ${START_WORLD}-${END_WORLD} x ${RUNS_PER_WORLD}次"
echo "=================================================="
echo ""

# 非tmux模式：直接在当前终端前台运行
if [[ "$USE_TMUX" == "false" ]]; then
    session="${SESSIONS_TO_START[0]}"
    IFS=':' read -r TMUX_NAME CONTAINER_ID POLICY_NAME <<< "$session"

    echo "前台运行评估..."
    echo "  容器ID: ${CONTAINER_ID}, 策略: ${POLICY_NAME}"
    echo "  评估范围: 世界 ${START_WORLD}-${END_WORLD} x ${RUNS_PER_WORLD}次"
    echo ""

    cd "${ROS_JACKAL_DIR}"
    exec bash script/HB/eval_batch_worlds_singularity.sh \
        --id ${CONTAINER_ID} \
        --policy ${POLICY_NAME} \
        --start ${START_WORLD} \
        --end ${END_WORLD} \
        --runs ${RUNS_PER_WORLD}

    # exec 会替换当前进程，所以下面的代码不会执行
fi

# tmux模式：在tmux会话中运行
for session in "${SESSIONS_TO_START[@]}"; do
    IFS=':' read -r TMUX_NAME CONTAINER_ID POLICY_NAME <<< "$session"

    echo "启动 ${TMUX_NAME}..."
    echo "  容器ID: ${CONTAINER_ID}, 策略: ${POLICY_NAME}"
    echo "  评估范围: 世界 ${START_WORLD}-${END_WORLD} x ${RUNS_PER_WORLD}次"

    # 检查 tmux 会话是否已存在
    if tmux has-session -t "${TMUX_NAME}" 2>/dev/null; then
        echo "  [SKIP] tmux 会话 '${TMUX_NAME}' 已存在"
        continue
    fi

    # 创建 tmux 会话并运行 eval_batch_worlds_singularity.sh
    tmux new-session -d -s "${TMUX_NAME}" \
        "cd ${ROS_JACKAL_DIR} && bash script/HB/eval_batch_worlds_singularity.sh \
            --id ${CONTAINER_ID} \
            --policy ${POLICY_NAME} \
            --start ${START_WORLD} \
            --end ${END_WORLD} \
            --runs ${RUNS_PER_WORLD}; \
         echo ''; \
         echo '批量评估完成。按 Enter 退出。'; \
         read"

    echo "  [OK] 已在 tmux '${TMUX_NAME}' 中启动"
done

echo ""
echo "=================================================="
echo "  所有 HB 评估客户端已启动!"
echo "=================================================="
echo ""
echo "查看所有 tmux 会话:  tmux ls"
echo "进入某个会话:        tmux attach -t applr_ddp"
echo "退出会话(不停止):    Ctrl+b d"
echo ""
echo "停止评估客户端:"
echo "  tmux kill-session -t applr_ddp"
echo "  或使用脚本: bash script/HB/kill_tmux_sessions.sh"
echo ""
