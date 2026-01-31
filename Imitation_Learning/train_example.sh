#!/bin/bash

# 训练 Transformer Imitation Learning 模型
# 示例脚本 - 现在只需要指定 planner，路径和参数会自动配置！

# ============================================================
# GPU 配置
# ============================================================
export CUDA_VISIBLE_DEVICES=0  # 指定使用哪个 GPU (0, 1, 2, ... 或 0,1 多卡)

# ============================================================
# 主要配置：只需修改这个！
# ============================================================
PLANNER="ddp"  # 选择: dwa, teb, mppi, ddp

# ============================================================
# 数据路径配置（可选，有默认值）
# ============================================================
DATA_ROOT="/data/local/yl2832/appvlm/"

# 如果需要覆盖默认路径，取消下面的注释
# TRAIN_JSON="/custom/path/to/train.json"
# EVAL_JSON="/custom/path/to/eval.json"
# IMAGE_FOLDER="/custom/path/to/images"
# OUTPUT_DIR="/custom/path/to/output"

# ============================================================
# 训练配置
# ============================================================
NUM_HISTORY_FRAMES=2            # 历史帧数量
BATCH_SIZE=16                   # 批次大小
NUM_EPOCHS=10                   # 训练轮数
LEARNING_RATE=1e-4              # 学习率
EVAL_SAMPLES=2000               # 评估时使用的样本数 (0=全部)

# ============================================================
# 保存配置
# ============================================================
SAVE_STEPS=5000                 # 每N步保存一次checkpoint (0表示禁用)
SAVE_TOTAL_LIMIT=20              # 最多保留N个step checkpoint
SAVE_INTERVAL=5                 # 每N个epoch保存一次

# ============================================================
# 模型配置
# ============================================================
# 小模型 (115M): vit_base_patch16_224 + 4层 (默认)
# 大模型 (330M): vit_large_patch16_224 + 2层 (与Qwen LoRA相当)
VISION_MODEL="vit_large_patch16_224"   # Vision Transformer 模型
NUM_TRANSFORMER_LAYERS=2               # Transformer 层数

# ============================================================
# 开始训练
# ============================================================
echo "Training Transformer IL for planner: $PLANNER"
echo "Data root: $DATA_ROOT"
echo "=========================================="

python train.py \
    --planner $PLANNER \
    --data_root $DATA_ROOT \
    --output_dir ./output/${PLANNER}_transformer_il \
    --num_history_frames $NUM_HISTORY_FRAMES \
    --batch_size $BATCH_SIZE \
    --num_epochs $NUM_EPOCHS \
    --learning_rate $LEARNING_RATE \
    --eval_samples $EVAL_SAMPLES \
    --vision_model $VISION_MODEL \
    --num_transformer_layers $NUM_TRANSFORMER_LAYERS \
    --normalize_params \
    --use_velocity \
    --vision_pretrained \
    --mixed_precision \
    --num_workers 4 \
    --log_interval 50 \
    --eval_interval 1 \
    --save_interval $SAVE_INTERVAL \
    --save_steps $SAVE_STEPS \
    --save_total_limit $SAVE_TOTAL_LIMIT \
    --save_best \
    --device cuda

# ============================================================
# 使用说明
# ============================================================
# 1. 基本用法（使用默认路径）：
#    只需修改 PLANNER="dwa" 或 "teb" 或 "mppi" 或 "ddp"
#    然后运行: bash train_example.sh
#
# 2. 自定义路径：
#    取消注释并修改 TRAIN_JSON, EVAL_JSON 等变量
#
# 3. 路径自动生成规则：
#    train_json: {DATA_ROOT}/{planner}_heurstic/splits_200k/chunk_000.json
#    eval_json:  {DATA_ROOT}/{planner}_heurstic/splits_200k/chunk_000.json
#    image_folder: {DATA_ROOT}/{planner}_heurstic/
#    output_dir: ./output/{planner}_transformer_il
#
# 4. 参数数量自动检测：
#    DWA: 9 params
#    TEB: 9 params
#    MPPI: 10 params
#    DDP: 8 params
#
# 5. Checkpoint 保存策略：
#    - 每 SAVE_STEPS 步保存一次 checkpoint_step_XXXXX.pth
#    - 每个 epoch 结束保存 latest.pth (始终是最新的)
#    - 每 SAVE_INTERVAL 个 epoch 保存 checkpoint_epoch_X.pth
#    - 验证集最佳模型保存为 best_model.pth
#    - step checkpoint 最多保留 SAVE_TOTAL_LIMIT 个
#
# 6. 输出目录结构：
#    ./output/{planner}_transformer_il/
#    ├── config.json              # 训练配置
#    ├── param_stats.json.npz     # 参数归一化统计
#    ├── latest.pth               # 最新 checkpoint
#    ├── best_model.pth           # 最佳模型
#    ├── checkpoint_epoch_5.pth   # epoch checkpoint
#    ├── checkpoint_step_5000.pth # step checkpoint
#    └── ...
