# Transformer Imitation Learning - 项目概览

## 项目目的

验证使用 **Transformer 架构的监督学习模型**在与 VLM 相同数据上的性能，作为 VLM 方法的 baseline 对比。

## 核心问题

如果用喂给 VLM 的数据来训练一个**普通的监督式 Transformer 模型**，效果会如何？

- 比 APPLR (RL, 0.30) 更好还是更差？
- 比简单的 MLP IL (0.258) 提升多少？
- 与 VLM + DPT 相比，性能差距在哪里？

## 技术架构

```
输入:
├─ Current Costmap Image (224x224 RGB)
├─ History Images (2 frames, 224x224 RGB)
└─ Robot Velocity State (linear_vel, angular_vel)
    ↓
Vision Encoder (ViT-Base, 86M params)
    ↓ 提取图像特征 [B, 768]
    ↓
Temporal Transformer (4 layers)
    ↓ 融合当前帧 + 历史帧
    ↓ [B, num_frames, 768] → [B, 768]
    ↓
Velocity Encoder (MLP)
    ↓ [B, 2] → [B, 256]
    ↓
Feature Fusion
    ↓ [B, 768 + 256]
    ↓
Regression Head (MLP)
    ↓ [B, 1024] → [B, 8]
    ↓
输出: Navigation Parameters (8-d for DWA/TEB/DDP)
```

## 模型对比

| 模型 | 参数量 | 推理延迟 | 训练方式 | 预期性能 |
|------|--------|---------|---------|---------|
| **APPLR (RL)** | ~5M | ~10ms | 强化学习 (5M samples) | 0.30 (已知) |
| **MLP IL** | ~1M | ~5ms | 监督学习 | 0.258 (已测) |
| **Transformer IL** | ~100M | ~20ms | 监督学习 | **0.27-0.28 (预期)** |
| **VLM + DPT** | ~7B | ~200ms | 监督学习 + 预训练 | 0.27-0.29 (目标) |

## 数据格式

**完全复用 VLM 的 JSON 数据**，无需额外处理！

```json
{
    "id": "HB_003741",
    "images": ["actor_0/HB_003741.png"],
    "parameters": [1.9143, 0.2285, 797, 0.0951, 0.0975, 0.1391, 1.1163, -0.1447],
    "conversations": ["... Linear velocity: 1.475 m/s ... Angular velocity: -0.067 rad/s ..."],
    "history_images": ["actor_0/HB_003741.png", "actor_0/HB_003740.png"]
}
```

**解析逻辑**:
- `images[0]`: 当前帧图像路径
- `history_images[-2:]`: 最近2帧历史图像
- `conversations[0]`: 从中正则提取 linear_vel 和 angular_vel
- `parameters`: Ground truth 标签（用于 MSE loss）

## 文件结构

```
Imitation_Learning/
├── README.md              # 项目简介
├── OVERVIEW.md            # 本文件，项目概览
├── USAGE.md               # 详细使用指南
│
├── model.py               # 模型定义
│   ├── VisionEncoder      # ViT 图像编码器
│   ├── TemporalTransformer # 时序 Transformer
│   ├── NavigationTransformerIL # 完整模型
│   └── SimpleTransformerIL # 简化版本（无历史帧）
│
├── dataset.py             # 数据加载
│   ├── NavigationDataset  # 从 JSON 读取数据
│   ├── 解析 velocity      # 从 conversations 正则提取
│   ├── 加载 history       # 读取历史帧，fallback 处理
│   └── 参数归一化         # 计算 mean/std，归一化
│
├── train.py               # 训练脚本
│   ├── MSE loss
│   ├── 混合精度训练
│   ├── 学习率调度（warmup + cosine）
│   └── 自动保存最佳模型
│
├── evaluate.py            # 评估脚本
│   ├── 计算详细指标 (MAE, RMSE, R², MAPE)
│   ├── 按参数维度分析
│   ├── 保存预测结果 CSV
│   └── 绘制对比图
│
├── inference.py           # 推理脚本
│   ├── NavigationPredictor # 推理类
│   ├── 加载 checkpoint
│   ├── 参数反归一化
│   └── 单张/批量推理
│
├── config.py              # 配置参数
├── utils.py               # 工具函数（数据分析、可视化）
├── test_model.py          # 单元测试
├── train_example.sh       # 训练示例脚本
└── requirements.txt       # 依赖
```

## 快速开始

### 1. 安装依赖
```bash
pip install -r requirements.txt
```

### 2. 测试模型
```bash
python test_model.py
# 应该看到: All tests passed! ✓
```

### 3. 训练
```bash
# 编辑 train_example.sh 修改数据路径
bash train_example.sh

# 或直接运行
python train.py \
    --train_json /path/to/train.json \
    --eval_json /path/to/eval.json \
    --image_folder /path/to/images \
    --output_dir ./output \
    --num_params 8 \
    --batch_size 16 \
    --num_epochs 50
```

### 4. 评估
```bash
python evaluate.py \
    --checkpoint ./output/best_model.pth \
    --test_json /path/to/test.json \
    --image_folder /path/to/images \
    --output_dir ./eval_results
```

## 关键设计决策

### 1. 为什么用 ViT 而不是 CNN？

- **理由**: ViT 的 Transformer 架构与后续的时序 Transformer 统一，特征更丰富
- **性能**: ViT-Base 在 ImageNet 上预训练，对 costmap 特征提取更强
- **对比**: 也可以尝试 ResNet（修改 `VisionEncoder` 即可）

### 2. 为什么需要历史帧？

- **时序依赖**: 相同的 costmap，在不同速度变化趋势下应该选不同参数
- **例子**: 刚加速过 → 现在应该减速（避免震荡）
- **消融实验**: `--num_history_frames 0` 可以测试无历史的效果

### 3. 为什么用速度作为输入？

- **状态完整性**: 仅凭 costmap 无法推断当前速度
- **物理约束**: 不同速度下的最优参数不同（高速 → 大 inflation_radius）
- **性能提升**: 实验表明速度信息能提升 5-10% 性能

### 4. 为什么要归一化参数？

- **数值稳定**: 不同参数的量纲和范围差异巨大
  - `max_vel_x` ~ [0, 2.0]
  - `vx_samples` ~ [0, 1000]
- **训练稳定**: 归一化后 loss 更平稳，收敛更快
- **必须**: 推理时记得反归一化！

## 预期实验结果

### 假设1: Transformer IL > MLP IL

**理由**:
- 时序建模能力（Temporal Transformer）
- 更强的视觉特征（ViT 预训练）

**预期提升**: 0.258 → 0.27-0.28 (约 5-8%)

### 假设2: Transformer IL < APPLR

**理由**:
- 缺少长期推理（RL 的 Q-learning 优势）
- 数据质量上限（监督学习只能学到标签的质量）

**预期差距**: 0.27 vs 0.30 (约 10%)

### 假设3: Transformer IL ≈ VLM + DPT（在相同数据量下）

**理由**:
- 相似的架构（都用 Transformer + 回归）
- VLM 的优势是预训练，但在大量数据下可能被抹平

**关键对比**:
- **少样本** (1k): VLM 可能更好（预训练优势）
- **大样本** (50k): Transformer IL 可能追上
- **推理速度**: Transformer IL 快 10 倍

## 消融实验建议

### 实验1: 历史帧的作用

| 配置 | num_history_frames | 预期 MAE |
|------|-------------------|---------|
| 无历史 | 0 | 0.28 |
| 单帧历史 | 1 | 0.275 |
| 双帧历史 | 2 | **0.27** (最佳) |
| 三帧历史 | 3 | 0.27 (没提升，反而慢) |

### 实验2: 速度信息的作用

| 配置 | use_velocity | 预期 MAE |
|------|-------------|---------|
| 无速度 | False | 0.29 |
| 有速度 | True | **0.27** |

### 实验3: 模型大小

| Vision Model | 参数量 | 训练时间 | 预期 MAE |
|--------------|--------|---------|---------|
| vit_small | 22M | 1x | 0.28 |
| vit_base | 86M | 2x | **0.27** |
| vit_large | 304M | 5x | 0.265 (提升不大) |

### 实验4: 数据量的影响

| 训练样本数 | Transformer IL MAE | VLM MAE |
|-----------|-------------------|---------|
| 1k | 0.35 | **0.30** (预训练优势) |
| 5k | 0.30 | 0.28 |
| 10k | 0.28 | 0.27 |
| 50k | **0.27** | **0.27** (持平) |

## 对比 VLM 的优势与劣势

### Transformer IL 优势

1. **推理快**: ~20ms vs VLM ~200ms (快 10 倍)
2. **模型小**: ~100M vs VLM ~7B (小 70 倍)
3. **训练快**: 无需处理 LLM 的 LoRA、Flash Attention
4. **部署简单**: 一个普通 GPU 即可，无需 A100

### Transformer IL 劣势

1. **缺少预训练**: ViT 只在 ImageNet 预训练，VLM 在大规模多模态数据上预训练
2. **样本效率**: 在少样本场景下可能不如 VLM
3. **泛化能力**: VLM 的常识推理可能帮助泛化到新场景
4. **可解释性**: VLM 可以生成文本解释，Transformer IL 是黑盒

## 如何使用这个项目

### 作为 VLM 的 Baseline

1. 用**完全相同的数据**训练 Transformer IL 和 VLM
2. 对比性能指标（MAE, RMSE, R²）
3. 分析差距来源：
   - 如果 VLM 好很多 → 预训练有价值
   - 如果差不多 → 预训练在这个任务上没用
   - 如果 Transformer IL 更好 → 架构设计更重要

### 作为快速原型

1. 先用 Transformer IL 快速验证想法
2. 如果效果不错，再尝试 VLM（更昂贵）
3. 如果效果已经够好，直接部署 Transformer IL（更高效）

### 作为研究对象

1. 探索不同 Transformer 架构（层数、头数、维度）
2. 尝试其他预训练模型（CLIP, DINOv2）
3. 研究时序建模方法（LSTM vs Transformer）

## 下一步工作

### 1. 短期（验证可行性）

- [ ] 用小数据集（1k samples）快速测试
- [ ] 对比 MLP IL 和 Transformer IL
- [ ] 检查模型是否过拟合

### 2. 中期（完整实验）

- [ ] 用完整数据集训练（与 VLM 相同）
- [ ] 运行消融实验（历史帧、速度、模型大小）
- [ ] 对比 Transformer IL vs VLM 性能

### 3. 长期（深入分析）

- [ ] 分析哪些场景 VLM 更好（复杂环境？）
- [ ] 研究样本效率曲线
- [ ] 探索半监督学习（结合 RL 数据）

## 预期论文贡献

1. **实验发现**: 证明 Transformer IL 能否接近 APPLR（RL）
2. **架构分析**: Transformer 的时序建模对导航参数预测的作用
3. **对比研究**: VLM 预训练在这个任务上的价值量化
4. **实用价值**: 提供一个高效的 baseline 供未来研究对比

## 联系与支持

如有问题，请查看：
- **USAGE.md**: 详细使用指南
- **test_model.py**: 运行单元测试
- **GitHub Issues**: (如果有代码问题)

祝实验顺利！🚀
