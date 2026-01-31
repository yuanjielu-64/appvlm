#!/bin/bash
# FTRL训练启动脚本

# 配置路径
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
ROS_JACKAL_DIR="$( cd "$SCRIPT_DIR/../.." && pwd )"

# 默认参数
CONFIG_FILE="ftrl_vlm_dwa"
BUFFER_PATH="$ROS_JACKAL_DIR/buffer/ftrl_vlm"
LOGGING_PATH="$ROS_JACKAL_DIR/logging/ftrl_vlm"
BUFFER_SIZE=350

# 解析命令行参数
while [[ $# -gt 0 ]]; do
  case $1 in
    --config)
      CONFIG_FILE="$2"
      shift 2
      ;;
    --buffer_path)
      BUFFER_PATH="$2"
      shift 2
      ;;
    --logging_path)
      LOGGING_PATH="$2"
      shift 2
      ;;
    --buffer_size)
      BUFFER_SIZE="$2"
      shift 2
      ;;
    *)
      echo "Unknown option: $1"
      echo "Usage: $0 [--config CONFIG_FILE] [--buffer_path PATH] [--logging_path PATH] [--buffer_size SIZE]"
      exit 1
      ;;
  esac
done

echo "========================================"
echo "FTRL Training Launcher"
echo "========================================"
echo "Config file: $CONFIG_FILE"
echo "Buffer path: $BUFFER_PATH"
echo "Logging path: $LOGGING_PATH"
echo "Buffer size: $BUFFER_SIZE"
echo "========================================"

# 创建目录
mkdir -p "$BUFFER_PATH"
mkdir -p "$LOGGING_PATH"

# 启动训练
cd "$ROS_JACKAL_DIR"

python rlft/train.py \
  --config_path script/ft_qwen/configs/ \
  --config_file "$CONFIG_FILE" \
  --buffer_path "$BUFFER_PATH" \
  --logging_path "$LOGGING_PATH" \
  --buffer_size "$BUFFER_SIZE"
