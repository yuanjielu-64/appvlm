# Transformer Imitation Learning for Robot Navigation

## 项目目的
验证使用 Transformer 架构的监督学习模型，在与 VLM 相同数据集上的性能表现。

## 模型架构
- **输入**:
  - Current costmap image (224x224 RGB)
  - History images (2 frames, 224x224 RGB)
  - Robot velocity state (linear_vel, angular_vel)
- **输出**:
  - Navigation parameters (7-8 维，取决于算法)
- **损失函数**: MSE Loss

## 数据格式
从 VLM 训练数据的 JSON 格式读取：
```json
{
    "id": "HB_003741",
    "images": ["actor_0/HB_003741.png"],
    "parameters": [1.9143, 0.2285, 797, ...],
    "conversations": ["... Linear velocity: 1.475 m/s\n... Angular velocity: -0.067 rad/s ..."],
    "history_images": ["actor_0/HB_003741.png", "actor_0/HB_003740.png"]
}
```

## 使用方法

### 训练
```bash
python train.py \
    --train_json /path/to/train.json \
    --eval_json /path/to/eval.json \
    --image_folder /path/to/images \
    --output_dir ./output \
    --num_epochs 50 \
    --batch_size 16
```

### 推理
```bash
python inference.py \
    --checkpoint ./output/best_model.pth \
    --image_path test_image.png \
    --linear_vel 1.5 \
    --angular_vel 0.1
```

## 目录结构
```
Imitation_Learning/
├── README.md
├── model.py              # Transformer 模型定义
├── dataset.py            # 数据加载器
├── train.py              # 训练脚本
├── inference.py          # 推理脚本
├── config.py             # 配置文件
└── utils.py              # 工具函数
```
