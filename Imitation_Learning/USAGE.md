# Transformer Imitation Learning 使用指南

## 快速开始

### 1. 安装依赖

```bash
cd src/Imitation_Learning
pip install -r requirements.txt
```

### 2. 测试模型

在训练之前，先测试模型是否正确构建：

```bash
python test_model.py
```

如果看到 "All tests passed! ✓"，说明模型构建正确。

### 3. 准备数据

你的数据应该已经是 VLM 的 JSON 格式：

```json
{
    "id": "HB_003741",
    "images": ["actor_0/HB_003741.png"],
    "parameters": [1.9143, 0.2285, 797, 0.0951, 0.0975, 0.1391, 1.1163, -0.1447],
    "conversations": ["... Linear velocity: 1.475 m/s ... Angular velocity: -0.067 rad/s ..."],
    "history_images": ["actor_0/HB_003741.png", "actor_0/HB_003740.png"]
}
```

**数据集分析**：

```bash
python utils.py /path/to/your/train.json
```

这会输出：
- 数据集大小
- 参数统计（均值、标准差、范围）
- 参数分布图
- 参数相关性矩阵

### 4. 训练模型

**方法 1：使用配置脚本**

编辑 `train_example.sh`，修改数据路径：

```bash
# 修改这些路径
TRAIN_JSON="/path/to/your/train.json"
EVAL_JSON="/path/to/your/eval.json"
IMAGE_FOLDER="/path/to/your/images"
```

然后运行：

```bash
bash train_example.sh
```

**方法 2：直接运行命令**

```bash
python train.py \
    --train_json /path/to/train.json \
    --eval_json /path/to/eval.json \
    --image_folder /path/to/images \
    --output_dir ./output/my_experiment \
    --num_params 8 \
    --num_history_frames 2 \
    --batch_size 16 \
    --num_epochs 50 \
    --learning_rate 1e-4 \
    --vision_model vit_base_patch16_224 \
    --normalize_params \
    --use_velocity \
    --vision_pretrained \
    --mixed_precision \
    --device cuda
```

**关键参数说明**：

- `--num_params`: 参数数量（根据你的算法：DWA=8, TEB=8, DDP=8）
- `--num_history_frames`: 使用几帧历史图像（推荐 2）
- `--vision_model`: Vision Transformer 模型
  - `vit_base_patch16_224` (86M 参数，推荐)
  - `vit_small_patch16_224` (22M 参数，更快)
  - `vit_large_patch16_224` (304M 参数，更强但慢)
- `--vision_freeze`: 是否冻结 vision encoder（冻结会更快，但可能效果稍差）
- `--normalize_params`: 是否归一化参数（强烈推荐）
- `--use_velocity`: 是否使用速度信息（推荐开启）

### 5. 监控训练

训练过程中会打印：

```
Epoch 1/50
==================================================
Epoch 1 [100/500] Loss: 0.125678, LR: 0.000100
...

Train metrics - Loss: 0.123456, MAE: 0.234567, MSE: 0.098765
Per-param MAE: [0.12, 0.08, 15.2, 0.05, 0.03, 0.04, 0.25, 0.11]
R^2: [0.85, 0.92, 0.78, 0.95, 0.88, 0.91, 0.82, 0.87]

Eval metrics - Loss: 0.135678, MAE: 0.256789, MSE: 0.109876
Per-param MAE: [0.13, 0.09, 16.1, 0.06, 0.04, 0.05, 0.27, 0.12]
R^2: [0.83, 0.90, 0.75, 0.93, 0.86, 0.89, 0.80, 0.85]

New best model! Eval loss: 0.135678
Saved best model to ./output/my_experiment/best_model.pth
```

**输出文件**：

```
output/my_experiment/
├── config.json                    # 训练配置
├── param_stats.json.npz           # 参数统计（用于推理）
├── best_model.pth                 # 最佳模型
├── checkpoint_epoch_5.pth         # 定期保存的 checkpoint
├── checkpoint_epoch_10.pth
└── ...
```

### 6. 评估模型

在测试集上评估：

```bash
python evaluate.py \
    --checkpoint ./output/my_experiment/best_model.pth \
    --test_json /path/to/test.json \
    --image_folder /path/to/images \
    --output_dir ./eval_results \
    --batch_size 32
```

输出：

```
Evaluation Results
============================================================
MSE:  0.098765
MAE:  0.234567
RMSE: 0.314159
MAPE: 12.34%

Per-parameter metrics:

Parameter 0:
  MAE:  0.120000
  RMSE: 0.150000
  R^2:  0.850000
  Max Error: 0.500000

...
```

生成的文件：

```
eval_results/
├── metrics.json                   # 详细指标
├── predictions.csv                # 每个样本的预测结果
└── prediction_comparison.png      # 预测 vs 真实值对比图
```

### 7. 推理

**单张图像推理**：

```bash
python inference.py \
    --checkpoint ./output/my_experiment/best_model.pth \
    --image_path /path/to/test_image.png \
    --linear_vel 1.5 \
    --angular_vel 0.1 \
    --device cuda
```

输出：

```
Predicted parameters:
[0.5  0.8  16.  20.  1.0  0.5  0.2  0.3]
```

**在 Python 代码中使用**：

```python
from inference import NavigationPredictor

# 初始化
predictor = NavigationPredictor(
    checkpoint_path='./output/my_experiment/best_model.pth',
    device='cuda'
)

# 预测
params = predictor.predict(
    current_image_path='test_image.png',
    history_image_paths=['hist1.png', 'hist2.png'],
    linear_vel=1.5,
    angular_vel=0.1
)

print(f"Predicted parameters: {params}")
```

## 高级用法

### 1. 不使用历史帧

如果你的数据没有历史帧，或者想测试不使用历史的效果：

```bash
python train.py \
    --num_history_frames 0 \
    ...其他参数
```

### 2. 不使用速度信息

如果想测试纯视觉的效果：

```bash
python train.py \
    --use_velocity False \
    ...其他参数
```

注意：这样可能会降低性能，因为速度是重要的状态信息。

### 3. 使用更小的模型

如果 GPU 内存不够或想要更快的训练：

```bash
python train.py \
    --vision_model vit_small_patch16_224 \
    --num_transformer_layers 2 \
    --batch_size 32 \
    ...其他参数
```

### 4. 冻结 Vision Encoder

只训练 Transformer 和回归头，加速训练：

```bash
python train.py \
    --vision_freeze \
    ...其他参数
```

### 5. 从 checkpoint 恢复训练

如果训练中断，可以恢复：

```bash
python train.py \
    --resume ./output/my_experiment/checkpoint_epoch_10.pth \
    ...其他参数（必须与之前一致）
```

## 与 VLM 的对比

### 数据格式

两者使用**完全相同**的 JSON 数据，所以你可以直接用 VLM 的数据训练 Transformer IL。

### 输入

| 模型 | 输入 |
|------|------|
| **VLM** | Costmap 图像 + Prompt (包含速度) + History images |
| **Transformer IL** | Costmap 图像 + 速度 (数值) + History images |

### 输出

两者都输出相同的参数向量（如 8 维 DWA 参数）。

### 性能对比建议

建议对比以下几个维度：

1. **预测精度**（MAE, RMSE, R²）
2. **训练样本效率**（1k, 5k, 10k, 50k 样本时的性能）
3. **推理速度**（Transformer IL 应该快 10-50 倍）
4. **泛化能力**（测试集性能下降幅度）
5. **模型大小**（Transformer IL ~100M vs VLM ~7B）

## 常见问题

### Q: 训练很慢怎么办？

A: 尝试以下方法：
1. 减小 batch_size
2. 使用更小的 vision_model（vit_small）
3. 减少 num_transformer_layers
4. 冻结 vision encoder（--vision_freeze）
5. 使用混合精度训练（--mixed_precision，默认开启）

### Q: GPU 内存不够？

A:
1. 减小 batch_size
2. 减小图像尺寸（--image_size 224 → 128）
3. 使用 vision_freeze
4. 使用 vit_small 模型

### Q: 模型过拟合？

A:
1. 增加 dropout（--dropout 0.2）
2. 增加 weight_decay（--weight_decay 1e-3）
3. 收集更多数据
4. 使用数据增强（需要修改 dataset.py）

### Q: 预测的参数不合理（太大或太小）？

A:
1. 确保使用了 --normalize_params
2. 检查 param_stats.json.npz 是否正确加载
3. 检查训练数据中的参数范围是否合理

### Q: 如何确定最佳超参数？

A: 建议的实验流程：

1. **Baseline** (默认配置)
   - vision_model: vit_base_patch16_224
   - num_history_frames: 2
   - use_velocity: True

2. **消融实验**
   - 不用历史帧（num_history_frames=0）
   - 不用速度（use_velocity=False）
   - 冻结 vision encoder（vision_freeze=True）

3. **模型大小对比**
   - vit_small (快但可能弱)
   - vit_base (平衡)
   - vit_large (慢但可能强)

## 实验建议

为了验证 "用 VLM 数据训练普通监督模型" 的效果，建议：

### 实验设置

```bash
# 1. 训练 Transformer IL (完整版)
python train.py \
    --train_json $VLM_TRAIN_JSON \
    --eval_json $VLM_EVAL_JSON \
    --num_history_frames 2 \
    --use_velocity True \
    --output_dir ./output/transformer_full

# 2. 训练无历史版本
python train.py \
    --train_json $VLM_TRAIN_JSON \
    --eval_json $VLM_EVAL_JSON \
    --num_history_frames 0 \
    --use_velocity True \
    --output_dir ./output/transformer_no_history

# 3. 训练无速度版本
python train.py \
    --train_json $VLM_TRAIN_JSON \
    --eval_json $VLM_EVAL_JSON \
    --num_history_frames 2 \
    --use_velocity False \
    --output_dir ./output/transformer_no_velocity

# 4. 对比 VLM
# (使用你已有的 VLM 推理脚本)
```

### 对比指标

| 模型 | MAE | RMSE | R² | 推理时间 | 参数量 |
|------|-----|------|----|---------|----|
| APPLR | - | - | - | ~10ms | ~5M |
| Transformer IL (Full) | ? | ? | ? | ~20ms | ~100M |
| Transformer IL (No History) | ? | ? | ? | ~15ms | ~100M |
| VLM + DPT | ? | ? | ? | ~200ms | ~7B |

希望这能帮助你快速上手！如有问题随时问我。
