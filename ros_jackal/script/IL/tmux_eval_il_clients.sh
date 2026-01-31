#!/bin/bash
# 根据 planner 类型启动 IL 评估 tmux 会话
#
# 用法:
#   bash tmux_eval_il_clients.sh --planner ddp    # 启动 DDP 的评估会话
#   bash tmux_eval_il_clients.sh --planner mppi   # 启动 MPPI 的评估会话
#   bash tmux_eval_il_clients.sh --planner dwa    # 启动 DWA 的评估会话
#   bash tmux_eval_il_clients.sh --planner teb    # 启动 TEB 的评估会话
#   bash tmux_eval_il_clients.sh --planner all    # 启动所有评估会话
#   bash tmux_eval_il_clients.sh --planner ddp --start 0 --end 99 --runs 3

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROS_JACKAL_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

# 解析参数
PLANNER=""
START_WORLD=0
END_WORLD=299
RUNS=3
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
      RUNS="$2"
      shift 2
      ;;
    --num)
      NUM_INSTANCES="$2"
      shift 2
      ;;
    *)
      echo "未知参数: $1"
      echo ""
      echo "用法: $0 --planner PLANNER [--num N] [--start N] [--end N] [--runs N]"
      echo ""
      echo "Planner 选项:"
      echo "  ddp   - 启动 DDP 评估会话 (端口 6003)"
      echo "  dwa   - 启动 DWA 评估会话 (端口 6000)"
      echo "  teb   - 启动 TEB 评估会话 (端口 6001)"
      echo "  mppi  - 启动 MPPI 评估会话 (端口 6002)"
      echo "  all   - 启动所有评估会话"
      echo ""
      echo "实例数量 (默认: 1):"
      echo "  --num N  每个planner启动N个并行实例"
      echo ""
      echo "示例:"
      echo "  $0 --planner ddp"
      echo "  $0 --planner ddp --num 2"
      echo "  $0 --planner all --num 2"
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

# IL 服务端口配置 (与 start_il_service.sh 一致)
# DWA=6000, TEB=6001, MPPI=6002, DDP=6003
declare -A IL_PORTS
IL_PORTS["dwa"]=6000
IL_PORTS["teb"]=6001
IL_PORTS["mppi"]=6002
IL_PORTS["ddp"]=6003

# 定义 planner 基础配置
# 格式: "基础tmux名称:基础容器ID:策略名称"
declare -A PLANNER_CONFIG
PLANNER_CONFIG["ddp"]="il_ddp:1:ddp_il"
PLANNER_CONFIG["dwa"]="il_dwa:5:dwa_il"
PLANNER_CONFIG["teb"]="il_teb:9:teb_il"
PLANNER_CONFIG["mppi"]="il_mppi:13:mppi_il"

# 生成指定 planner 的多实例会话
generate_sessions() {
    local planner="$1"
    local config="${PLANNER_CONFIG[$planner]}"
    local il_port="${IL_PORTS[$planner]}"

    if [[ -z "$config" ]]; then
        echo "错误: 未知的 planner '$planner'"
        return 1
    fi

    IFS=':' read -r BASE_TMUX_NAME BASE_CONTAINER_ID POLICY_NAME <<< "$config"

    for ((i=0; i<NUM_INSTANCES; i++)); do
        local container_id=$((BASE_CONTAINER_ID + i))
        if [[ "$NUM_INSTANCES" -eq 1 ]]; then
            local tmux_name="${BASE_TMUX_NAME}"
        else
            local tmux_name="${BASE_TMUX_NAME}_${i}"
        fi
        # 格式: "tmux名称:容器ID:IL端口:策略名称"
        SESSIONS_TO_START+=("${tmux_name}:${container_id}:${il_port}:${POLICY_NAME}")
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
echo "  启动 IL 评估会话 - Planner: ${PLANNER^^}"
echo "=================================================="
echo "实例数量:    每个planner ${NUM_INSTANCES} 个实例"
echo "会话总数:    ${#SESSIONS_TO_START[@]}"
echo "环境范围:    $START_WORLD - $END_WORLD"
echo "每环境运行:  $RUNS 次"
echo ""
echo "会话列表:"
for session in "${SESSIONS_TO_START[@]}"; do
    IFS=':' read -r _TMUX_NAME _CONTAINER_ID _IL_PORT _POLICY_NAME <<< "$session"
    printf "  %-15s  ID: %-3s  IL端口: %-5s  策略: %s\n" "$_TMUX_NAME" "$_CONTAINER_ID" "$_IL_PORT" "$_POLICY_NAME"
done
echo "=================================================="
echo ""

# 启动所有会话
for session in "${SESSIONS_TO_START[@]}"; do
    IFS=':' read -r TMUX_NAME CONTAINER_ID IL_PORT POLICY_NAME <<< "$session"

    echo "启动 ${TMUX_NAME}..."
    echo "  容器ID: ${CONTAINER_ID}, IL端口: ${IL_PORT}, 策略: ${POLICY_NAME}"

    # 检查 tmux 会话是否已存在
    if tmux has-session -t "${TMUX_NAME}" 2>/dev/null; then
        echo "  [SKIP] tmux 会话 '${TMUX_NAME}' 已存在"
        continue
    fi

    # 创建 tmux 会话并运行批量评估
    tmux new-session -d -s "${TMUX_NAME}" \
        "cd ${ROS_JACKAL_DIR} && bash script/IL/eval_batch_worlds_singularity.sh \
            --id ${CONTAINER_ID} \
            --il_port ${IL_PORT} \
            --policy ${POLICY_NAME} \
            --start ${START_WORLD} \
            --end ${END_WORLD} \
            --runs ${RUNS}; \
         echo ''; \
         echo '评估完成。按 Enter 退出。'; \
         read"

    echo "  [OK] 已在 tmux '${TMUX_NAME}' 中启动"

    # 等待 Gazebo 完成初始化，避免资源竞争
    echo "  等待 10 秒让 Gazebo 初始化..."
    sleep 10
done

echo ""
echo "=================================================="
echo "  所有 IL 评估会话已启动!"
echo "=================================================="
echo ""
echo "IL 服务端口映射:"
echo "  DWA:  http://localhost:6000"
echo "  TEB:  http://localhost:6001"
echo "  MPPI: http://localhost:6002"
echo "  DDP:  http://localhost:6003"
echo ""
echo "查看所有 tmux 会话:  tmux ls"
echo "进入某个会话:        tmux attach -t il_ddp"
echo "退出会话(不停止):    Ctrl+b d"
echo ""
echo "停止所有评估会话:"
echo "  for s in \$(tmux ls -F '#{session_name}' | grep '^il_'); do tmux kill-session -t \$s; done"
echo ""
