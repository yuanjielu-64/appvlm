# 导航参数回归数据格式说明

## 数据格式

训练数据应为 JSON 格式，每条数据包含：

```json
[
  {
    "id": "sample_001",
    "images": ["costmap_001.png"],
    "parameters": [0.5, 0.8, 16, 20, 1.0, 0.5, 0.2],
    "conversations": [
      "<image>\nPredict navigation parameters for this costmap.",
      "Navigation parameters predicted."
    ],
    "system_prompt": "You are a navigation parameter prediction assistant."
  },
  {
    "id": "sample_002",
    "images": ["costmap_002.png"],
    "parameters": [0.4, 0.6, 12, 16, 1.5, 0.3, 0.3],
    "conversations": [
      "<image>\nWhat are the optimal parameters?",
      "Parameters calculated."
    ]
  }
]
```

## 字段说明

### 必需字段

- **`images`** (list): 图像文件名列表（相对于 `image_folder`）
  - 通常包含一张 costmap 图像
  - 也可以是场景RGB图像

- **`parameters`** (list of float): 7个导航参数的ground truth值
  - 长度必须为 7
  - 按顺序对应：
    1. `max_vel_x` - 最大线速度
    2. `max_vel_theta` - 最大角速度
    3. `vx_samples` - 线速度采样数
    4. `vtheta_samples` - 角速度采样数
    5. `occdist_scale` - 障碍物距离权重
    6. `pdist_scale` - 路径距离权重
    7. `gdist_scale` - 目标距离权重
  - 或者根据你的具体任务定义

### 可选字段

- **`conversations`** (list): 用户-助手对话
  - 第一个元素：用户输入（包含 `<image>` 标记）
  - 第二个元素：助手响应（dummy，回归任务不需要）
  - 如果未提供，将使用默认提示

- **`system_prompt`** (str): 系统提示词
  - 如果未提供，使用默认：`"You are a navigation parameter prediction assistant."`

- **`id`** (str): 样本唯一标识符

## 参数归一化

### 建议归一化方法

由于7个参数的量纲和范围不同，强烈建议进行归一化：

```python
import numpy as np
import json

# 读取原始数据
with open('raw_data.json') as f:
    data = json.load(f)

# 提取所有参数
all_params = np.array([d['parameters'] for d in data])

# 计算均值和标准差
param_mean = all_params.mean(axis=0)
param_std = all_params.std(axis=0)

# 归一化
for d in data:
    d['parameters'] = ((np.array(d['parameters']) - param_mean) / param_std).tolist()

# 保存归一化数据
with open('normalized_data.json', 'w') as f:
    json.dump(data, f, indent=2)

# 保存统计信息（用于反归一化）
np.save('param_mean.npy', param_mean)
np.save('param_std.npy', param_std)
```

### 推理时反归一化

```python
import numpy as np

# 加载统计信息
param_mean = np.load('param_mean.npy')
param_std = np.load('param_std.npy')

# 模型预测（归一化值）
predictions_normalized = model.predict(...)

# 反归一化
predictions_original = predictions_normalized * param_std + param_mean
```

## 数据准备示例

### 从 BARN 导航数据生成

```python
import json
import os
from PIL import Image

def create_regression_dataset(
    costmap_dir: str,
    parameter_csv: str,
    output_json: str
):
    """
    从 costmap 图像和参数CSV创建回归数据集

    Args:
        costmap_dir: costmap图像目录
        parameter_csv: 参数CSV文件 (columns: image_name, p1, p2, ..., p7)
        output_json: 输出JSON文件路径
    """
    import pandas as pd

    # 读取参数
    df = pd.read_csv(parameter_csv)

    dataset = []
    for idx, row in df.iterrows():
        image_name = row['image_name']
        params = row[['p1', 'p2', 'p3', 'p4', 'p5', 'p6', 'p7']].tolist()

        # 检查图像是否存在
        image_path = os.path.join(costmap_dir, image_name)
        if not os.path.exists(image_path):
            print(f"Warning: {image_path} not found, skipping...")
            continue

        dataset.append({
            "id": f"sample_{idx:05d}",
            "images": [image_name],
            "parameters": params,
            "conversations": [
                "<image>\nPredict navigation parameters.",
                "Parameters predicted."
            ]
        })

    # 保存
    with open(output_json, 'w') as f:
        json.dump(dataset, f, indent=2)

    print(f"Created dataset with {len(dataset)} samples")
    print(f"Saved to {output_json}")

# 使用示例
create_regression_dataset(
    costmap_dir='./data/costmaps',
    parameter_csv='./data/parameters.csv',
    output_json='./data/navigation_regression_train.json'
)
```

## 数据增强建议

### 图像增强

```python
from torchvision import transforms

augmentation = transforms.Compose([
    transforms.RandomRotation(10),
    transforms.RandomHorizontalFlip(0.5),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.RandomResizedCrop(size=(224, 224), scale=(0.9, 1.0))
])
```

### 参数扰动（可选）

```python
import numpy as np

def add_parameter_noise(params, noise_std=0.05):
    """添加高斯噪声进行数据增强"""
    noise = np.random.normal(0, noise_std, len(params))
    return params + noise
```

## 训练/验证分割

```python
from sklearn.model_selection import train_test_split

# 读取数据
with open('all_data.json') as f:
    all_data = json.load(f)

# 80/20 分割
train_data, eval_data = train_test_split(
    all_data, test_size=0.2, random_state=42
)

# 保存
with open('train.json', 'w') as f:
    json.dump(train_data, f, indent=2)

with open('eval.json', 'w') as f:
    json.dump(eval_data, f, indent=2)
```

## 完整示例数据

```json
[
  {
    "id": "barn_world_001_optimal",
    "images": ["barn_001_costmap.png"],
    "parameters": [0.5, 0.8, 16, 20, 1.0, 0.5, 0.2],
    "conversations": [
      "<image>\nAnalyze this costmap and predict optimal DWA parameters for safe navigation.",
      "Navigation parameters optimized for the given environment."
    ],
    "system_prompt": "You are an expert in robot navigation parameter tuning."
  },
  {
    "id": "barn_world_002_tight",
    "images": ["barn_002_costmap.png"],
    "parameters": [0.3, 0.6, 12, 16, 1.5, 0.3, 0.3],
    "conversations": [
      "<image>\nThis is a tight corridor. What parameters should be used?",
      "Parameters adjusted for narrow passage navigation."
    ]
  },
  {
    "id": "barn_world_003_open",
    "images": ["barn_003_costmap.png"],
    "parameters": [0.8, 1.0, 20, 24, 0.8, 0.6, 0.1],
    "conversations": [
      "<image>\nOpen space detected. Suggest navigation parameters.",
      "Parameters set for fast open-space navigation."
    ]
  }
]
```

## 注意事项

1. **参数范围检查**：确保所有参数值在合理范围内
2. **图像格式**：支持 PNG, JPG 等常见格式
3. **路径一致性**：`images` 中的文件名应与 `image_folder` 中的文件对应
4. **归一化**：强烈建议对参数进行归一化，提高训练稳定性
5. **数据质量**：参数的 ground truth 应来自实际导航性能最优的配置
