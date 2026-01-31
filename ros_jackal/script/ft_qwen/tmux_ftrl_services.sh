#!/bin/bash
# 批量启动 FTRL VLM+DPT 服务（使用 Qwen2.5-VL + DPT）
# 支持启动多个实例，每个绑定到不同GPU和端口

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# 🔍 智能检测 appvlm_ws 路径
APPVLM_WS="$(cd "${SCRIPT_DIR}" && while [[ "$PWD" != "/" ]]; do
  if [[ "$(basename "$PWD")" == "appvlm_ws" ]]; then
    echo "$PWD";
    break;
  fi;
  cd ..;
done)"

# 备用路径检测
if [[ -z "$APPVLM_WS" ]]; then
  for candidate in \
    "/home/yuanjielu/robot_navigation/noetic/appvlm_ws" \
    "/data/local/yl2832/appvlm_ws" \
    "$HOME/robot_navigation/noetic/appvlm_ws" \
    "$HOME/appvlm_ws"; do
    if [[ -d "$candidate" ]]; then
      APPVLM_WS="$candidate"
      break
    fi
  done
fi

if [[ -z "$APPVLM_WS" ]]; then
  echo "❌ Error: Cannot find appvlm_ws directory!"
  exit 1
fi

MODEL_BASE="${APPVLM_WS}/src/ros_jackal/model"
BASE_MODEL="Qwen/Qwen2.5-VL-3B-Instruct"

# 默认参数
PLANNER="ddp"
NUM_SERVICES=1
GPU_STRATEGY="0"  # 默认所有服务共享GPU 0
START_PORT=5000
LORA_PATH=""

# 帮助信息
show_help() {
    cat << EOF
Usage: $0 [OPTIONS]

启动FTRL VLM+DPT推理服务（基于 Qwen2.5-VL）

Options:
    --planner PLANNER      规划器类型 (ddp/dwa/teb/mppi, 默认: ddp)
    --num NUM             启动服务数量 (默认: 1)
    --gpu GPU             GPU分配策略:
                           - 单个数字: 所有服务共享该GPU (例如: --gpu 0)
                           - 逗号分隔: 手动指定每个服务的GPU (例如: --gpu 0,1,0,2)
                           - "auto": 自动递增GPU (--num 3 -> GPU 0,1,2)
                           默认: 0 (所有服务共享GPU 0)
    --port PORT           起始端口号 (默认: 5000, 自动递增)
    --lora_path PATH      LoRA checkpoint路径 (默认: model/{planner}/checkpoint-5000)
    --base_model PATH     基础VLM模型路径 (默认: Qwen/Qwen2.5-VL-3B-Instruct)
    -h, --help            显示帮助信息

Examples:
    # 启动1个DDP服务在GPU 0，端口5000
    $0

    # 启动3个服务，都在GPU 0（共享显存），端口5000/5001/5002
    $0 --num 3 --gpu 0

    # 启动3个服务，自动分配到GPU 0/1/2，端口5000/5001/5002
    $0 --num 3 --gpu auto

    # 启动4个服务，手动指定GPU分配（服务0和2共享GPU 0）
    $0 --num 4 --gpu 0,1,0,2

    # 启动2个DWA服务在GPU 3，从端口6000开始
    $0 --planner dwa --num 2 --gpu 3 --port 6000

    # 指定LoRA checkpoint
    $0 --lora_path /path/to/checkpoint-2500
EOF
}

# 解析参数
while [[ $# -gt 0 ]]; do
    case $1 in
        --planner)
            PLANNER="$2"
            shift 2
            ;;
        --num)
            NUM_SERVICES="$2"
            shift 2
            ;;
        --gpu)
            GPU_STRATEGY="$2"
            shift 2
            ;;
        --port)
            START_PORT="$2"
            shift 2
            ;;
        --lora_path|--checkpoint)  # 兼容旧参数名
            LORA_PATH="$2"
            shift 2
            ;;
        --base_model)
            BASE_MODEL="$2"
            shift 2
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

# 解析GPU分配策略
declare -a GPU_ARRAY
if [[ "$GPU_STRATEGY" == "auto" ]]; then
    # 自动递增: 0, 1, 2, ...
    for ((i=0; i<NUM_SERVICES; i++)); do
        GPU_ARRAY[$i]=$i
    done
elif [[ "$GPU_STRATEGY" == *,* ]]; then
    # 逗号分隔: 手动指定每个服务的GPU
    IFS=',' read -ra GPU_ARRAY <<< "$GPU_STRATEGY"
    if [ ${#GPU_ARRAY[@]} -ne $NUM_SERVICES ]; then
        echo "[ERROR] GPU count (${#GPU_ARRAY[@]}) must match NUM_SERVICES ($NUM_SERVICES)"
        echo "        You provided: --gpu $GPU_STRATEGY"
        exit 1
    fi
else
    # 单个数字: 所有服务共享该GPU
    for ((i=0; i<NUM_SERVICES; i++)); do
        GPU_ARRAY[$i]=$GPU_STRATEGY
    done
fi

# 如果没有指定lora_path，使用默认路径
if [ -z "$LORA_PATH" ]; then
    LORA_PATH="${MODEL_BASE}/${PLANNER}/checkpoint-5000"
fi

echo "=================================================="
echo "  Starting FTRL VLM+DPT Services (Qwen2.5-VL)"
echo "=================================================="
echo "Planner: ${PLANNER}"
echo "Number of services: ${NUM_SERVICES}"
echo "GPU Strategy: ${GPU_STRATEGY}"
echo "GPU Mapping: ${GPU_ARRAY[*]}"
echo "Start Port: ${START_PORT}"
echo "Base Model: ${BASE_MODEL}"
echo "LoRA Path: ${LORA_PATH}"
echo "=================================================="

# 检查 LoRA checkpoint 是否存在
if [ ! -d "${LORA_PATH}" ]; then
    echo "[ERROR] LoRA checkpoint directory not found: ${LORA_PATH}"
    exit 1
fi

# 启动多个服务
for ((i=0; i<NUM_SERVICES; i++)); do
    GPU=${GPU_ARRAY[$i]}
    PORT=$((START_PORT + i))
    NAME="ftrl_${PLANNER}_${i}"

    echo ""
    echo "Starting service $((i+1))/${NUM_SERVICES}..."
    echo "  Name: ${NAME}"
    echo "  GPU: ${GPU}"
    echo "  Port: ${PORT}"

    # 检查并清理端口占用
    echo "  Checking port ${PORT}..."
    if lsof -ti:${PORT} >/dev/null 2>&1; then
        echo "  [WARN] Port ${PORT} is in use, killing process..."
        lsof -ti:${PORT} | xargs kill -9 2>/dev/null
        sleep 1
    fi

    # 检查 tmux 会话是否已存在
    if tmux has-session -t "${NAME}" 2>/dev/null; then
        echo "  [WARN] tmux session '${NAME}' already exists, killing it..."
        tmux kill-session -t "${NAME}"
        sleep 1
    fi

    # 创建 tmux 会话并启动服务
    CONDA_PYTHON="/home/yuanjielu/miniforge3/envs/lmms-finetune-qwen/bin/python"
    SERVER_SCRIPT="${SCRIPT_DIR}/qwen_server.py"

    # 构建启动命令（与 qwen_server.py 参数一致）
    tmux new-session -d -s "${NAME}" \
        "export CUDA_VISIBLE_DEVICES=${GPU} && \
         ${CONDA_PYTHON} ${SERVER_SCRIPT} \
            --base_model ${BASE_MODEL} \
            --lora_path ${LORA_PATH} \
            --algorithm ${PLANNER^^} \
            --port ${PORT} \
            --device cuda:0 \
            --load_in_4bit; \
         echo 'Service stopped. Press Enter to exit.'; read"

    echo "  [OK] Started in tmux session '${NAME}'"

    # 等待一小段时间，避免同时启动太多进程
    sleep 3
done

echo ""
echo "=================================================="
echo "  All ${NUM_SERVICES} service(s) started!"
echo "=================================================="
echo ""
echo "查看所有 tmux 会话:    tmux ls"
echo "进入某个会话:          tmux attach -t ftrl_${PLANNER}_0"
echo "退出会话(不停止):      Ctrl+b d"
echo "停止某个会话:          tmux kill-session -t ftrl_${PLANNER}_0"
echo "停止所有${PLANNER}会话:      ./kill_ftrl_services.sh ${PLANNER}"
echo ""
echo "测试服务健康状态:"
for ((i=0; i<NUM_SERVICES; i++)); do
    PORT=$((START_PORT + i))
    echo "  curl http://localhost:${PORT}/health"
done
echo ""
