# Qwen2.5-VL + DPT Head for Navigation Parameter Regression

## 概述

本实现将 Qwen2.5-VL 与 DPT (Dense Prediction Transformer) head 结合，用于从 costmap/场景图像预测导航算法的 7 个连续参数。

**任务**: Costmap/场景图像 → 7 个导航参数（MSE 回归）

**参考**:
- DPT head 设计参考 [DUSt3R](https://github.com/naver/dust3r)
- 基于 lmms-finetune-qwen 框架

## 架构

```
Input: Costmap/RGB Image
         ↓
Qwen2.5-VL (Vision Encoder + LLM)
         ↓
Extract multi-layer hidden states (最后4层)
         ↓
DPT Head (渐进式特征融合)
  - Layer projections
  - Feature refinement blocks
  - Progressive fusion (bottom-up)
  - Global pooling
         ↓
Output: 7 navigation parameters
```

## 核心组件

### 1. 模型 (`models/qwen2_5_vl_dpt_regression.py`)

- **`Qwen2_5_VLForRegression`**: 主模型类
  - 包装 Qwen2.5-VL base model
  - 提取多层 hidden states
  - 通过 DPT head 预测参数

- **`DPTRegressionHead`**: DPT 风格的回归头
  - 从多层提取特征并投影到统一空间
  - 使用 FeatureRefinementBlock 渐进融合
  - 全局池化后预测 7 个参数

- **`FeatureRefinementBlock`**: 特征细化模块
  - 类似 DUSt3R 的 refinenet
  - 自顶向下融合多层特征

### 2. 数据处理 (`collators/qwen2_5_vl_regression.py`)

- **`Qwen2_5_VLRegressionDataCollator`**: 回归任务专用 collator
  - 处理图像和文本
  - 返回参数标签 (shape: [B, 7])

### 3. 训练器 (`trainers/regression_trainer.py`)

- **`RegressionTrainer`**: 自定义 Trainer
  - 使用 MSE loss
  - 计算回归指标 (MAE, RMSE, R²)
  - 支持参数归一化

### 4. Loader (`loaders/qwen2_5_vl_regression.py`)

- **`Qwen2_5_VLRegressionModelLoader`**: 模型加载器
  - 加载 base Qwen2.5-VL
  - 添加 DPT regression head
  - 配置多层提取参数

## 快速开始

### 1. 准备数据

参考 `DATA_FORMAT_REGRESSION.md` 准备数据：

```json
[
  {
    "images": ["costmap_001.png"],
    "parameters": [0.5, 0.8, 16, 20, 1.0, 0.5, 0.2],
    "conversations": [
      "<image>\nPredict navigation parameters.",
      "Parameters predicted."
    ]
  }
]
```

**参数归一化**（强烈推荐）：

```python
import numpy as np
import json

with open('raw_data.json') as f:
    data = json.load(f)

all_params = np.array([d['parameters'] for d in data])
param_mean = all_params.mean(axis=0)
param_std = all_params.std(axis=0)

for d in data:
    d['parameters'] = ((np.array(d['parameters']) - param_mean) / param_std).tolist()

with open('normalized_data.json', 'w') as f:
    json.dump(data, f)

np.save('param_mean.npy', param_mean)
np.save('param_std.npy', param_std)
```

### 2. 训练

```bash
# 修改配置
vim configs/qwen2_5_vl_regression_example.sh

# 设置模型路径、数据路径等
MODEL_PATH="/path/to/Qwen2.5-VL-7B"
DATA_PATH="./data/navigation_regression_train.json"
IMAGE_FOLDER="./data/costmaps"

# 执行训练
bash configs/qwen2_5_vl_regression_example.sh
```

**关键参数**:
- `--model_family_id qwen2.5-vl-regression`: 使用回归模型
- `--num_params 7`: 预测 7 个参数
- `--feature_dim 256`: DPT 特征维度
- `--num_layers_to_extract 4`: 提取最后 4 层

### 3. 推理

```bash
python inference_regression.py \
  --model_path ./output/qwen2_5_vl_dpt_regression \
  --image_path ./test_costmap.png \
  --param_mean param_mean.npy \
  --param_std param_std.npy
```

输出示例：
```
Predicted parameters (original scale):
  max_vel_x           : 0.5234
  max_vel_theta       : 0.7891
  vx_samples          : 15.2341
  vtheta_samples      : 19.5672
  occdist_scale       : 1.0234
  pdist_scale         : 0.4891
  gdist_scale         : 0.2123
```

## 训练细节

### LoRA 配置

默认只对 LLM 部分使用 LoRA，vision encoder 冻结：

```bash
--use_lora true \
--lora_r 64 \
--lora_alpha 128 \
--lora_dropout 0.05
```

### DPT Head 配置

- **特征维度**: 256 (可调整)
- **提取层数**: 4 (最后4层)
- **融合方式**: 自顶向下渐进融合
- **池化**: Masked average pooling

### 损失函数

```python
loss = MSELoss(predictions, labels)
```

### 评估指标

- **MAE** (Mean Absolute Error): 平均绝对误差
- **RMSE** (Root Mean Squared Error): 均方根误差
- **R²** (Coefficient of Determination): 决定系数

## 与 baseline 的对比

### APPLR (RL-based)
- 方法: TD3 强化学习
- 训练: 需要 5M samples, 500 并行环境, 6小时
- 样本效率: 低
- 可解释性: 弱

### APPLLLM (本实现, VLM-based)
- 方法: Qwen2.5-VL + DPT head (监督学习)
- 训练: 直接从 (costmap, parameters) pairs 学习
- 样本效率: 高（少样本即可）
- 可解释性: 强（VLM 可以理解场景语义）

## 文件结构

```
lmms-finetune-qwen/
├── models/
│   ├── __init__.py
│   └── qwen2_5_vl_dpt_regression.py  # 主模型
├── loaders/
│   └── qwen2_5_vl_regression.py      # 模型加载器
├── collators/
│   └── qwen2_5_vl_regression.py      # 数据 collator
├── trainers/
│   ├── __init__.py
│   └── regression_trainer.py         # 自定义 Trainer
├── configs/
│   └── qwen2_5_vl_regression_example.sh  # 训练配置
├── train_regression.py               # 训练脚本
├── inference_regression.py           # 推理脚本
├── DATA_FORMAT_REGRESSION.md         # 数据格式说明
└── README_REGRESSION.md              # 本文档
```

## 超参数调优建议

### DPT Head

```python
# 小数据集
feature_dim = 128
num_layers = 2

# 中等数据集（推荐）
feature_dim = 256
num_layers = 4

# 大数据集
feature_dim = 512
num_layers = 6
```

### 学习率

```bash
# Base model (LoRA)
--learning_rate 2e-5

# DPT head (全训练)
--learning_rate 5e-4
```

### Batch Size

```bash
# 建议根据 GPU 内存调整
--per_device_train_batch_size 4 \
--gradient_accumulation_steps 4  # 有效 batch_size = 16
```

## 常见问题

### Q1: OOM (Out of Memory)

**解决方案**:
- 减小 `batch_size`
- 增加 `gradient_accumulation_steps`
- 减小 `feature_dim`
- 启用 `gradient_checkpointing`
- 使用 `--use_lora true` 只训练部分参数

### Q2: 训练不收敛

**检查**:
- 参数是否归一化？
- 学习率是否合适？(尝试 1e-5 ~ 1e-4)
- 数据质量如何？ground truth 是否准确？

**尝试**:
- 增加 warmup_ratio (0.1)
- 使用 cosine learning rate scheduler
- 检查数据分布是否平衡

### Q3: 推理速度慢

**优化**:
- 使用 `--use_flash_attn true`
- 量化模型 (4-bit, 8-bit)
- 使用更小的 base model (Qwen2.5-VL-3B)

### Q4: 如何可视化预测结果？

```python
import matplotlib.pyplot as plt

# 预测 vs Ground Truth
plt.figure(figsize=(10, 6))
for i in range(7):
    plt.subplot(2, 4, i+1)
    plt.scatter(labels[:, i], predictions[:, i], alpha=0.5)
    plt.plot([labels[:, i].min(), labels[:, i].max()],
             [labels[:, i].min(), labels[:, i].max()], 'r--')
    plt.xlabel('Ground Truth')
    plt.ylabel('Prediction')
    plt.title(f'Parameter {i+1}')
plt.tight_layout()
plt.savefig('predictions.png')
```

## 下一步

1. **数据收集**: 从 BARN 环境收集 (costmap, optimal_params) pairs
2. **训练**: 使用本实现进行训练
3. **评估**: 在 BARN 测试集上评估导航性能
4. **对比**: 与 APPLR baseline 对比
5. **部署**: 集成到 ROS 导航系统

## 参考

- [DUSt3R](https://github.com/naver/dust3r): DPT head 设计参考
- [Qwen2.5-VL](https://github.com/QwenLM/Qwen2.5-VL): 基础视觉语言模型
- [APPLR Paper](../../../applr.pdf): Baseline 方法

## License

本实现遵循 lmms-finetune 和 Qwen2.5-VL 的许可证。
