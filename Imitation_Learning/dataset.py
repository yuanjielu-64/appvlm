"""
Dataset for Transformer Imitation Learning
从 VLM 的 JSON 格式数据中读取
"""

import json
import os
import re
from typing import List, Dict, Any, Optional
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
import numpy as np


class NavigationDataset(Dataset):
    """
    导航数据集，从 JSON 文件加载

    JSON 格式:
    {
        "id": "HB_003741",
        "images": ["actor_0/HB_003741.png"],
        "parameters": [1.9143, 0.2285, ...],
        "conversations": ["... Linear velocity: 1.475 m/s ... Angular velocity: -0.067 rad/s ..."],
        "history_images": ["actor_0/HB_003741.png", "actor_0/HB_003740.png"]
    }
    """

    def __init__(
        self,
        json_path: str,
        image_folder: str,
        num_history_frames: int = 2,
        image_size: int = 224,
        normalize_params: bool = True,
        param_stats: Optional[Dict[str, np.ndarray]] = None
    ):
        """
        Args:
            json_path: JSON 数据文件路径
            image_folder: 图像文件夹根目录
            num_history_frames: 使用的历史帧数量
            image_size: 图像resize大小
            normalize_params: 是否归一化参数
            param_stats: 参数统计信息 {'mean': array, 'std': array}
        """
        self.image_folder = image_folder
        self.num_history_frames = num_history_frames
        self.image_size = image_size
        self.normalize_params = normalize_params

        # 加载 JSON 数据
        with open(json_path, 'r') as f:
            self.data = json.load(f)

        print(f"Loaded {len(self.data)} samples from {json_path}")

        # 图像预处理
        self.transform = T.Compose([
            T.Resize((image_size, image_size)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # 计算或加载参数统计
        if normalize_params:
            if param_stats is None:
                self.param_mean, self.param_std = self._compute_param_stats()
            else:
                self.param_mean = param_stats['mean']
                self.param_std = param_stats['std']

            print(f"Parameter mean: {[f'{x:.3f}' for x in self.param_mean]}")
            print(f"Parameter std: {[f'{x:.3f}' for x in self.param_std]}")
        else:
            self.param_mean = None
            self.param_std = None

    def _compute_param_stats(self):
        """计算参数的均值和标准差"""
        all_params = []
        for item in self.data:
            all_params.append(item['parameters'])

        all_params = np.array(all_params)
        mean = np.mean(all_params, axis=0)
        std = np.std(all_params, axis=0)

        # 避免除以0
        std = np.where(std < 1e-6, 1.0, std)

        return mean, std

    def get_param_stats(self):
        """返回参数统计信息（用于保存）"""
        return {
            'mean': self.param_mean,
            'std': self.param_std
        }

    def _parse_velocity_from_conversation(self, conversation: str):
        """从 conversation 文本中解析速度信息"""
        # 匹配 "Linear velocity: 1.475 m/s"
        linear_match = re.search(r'Linear velocity:\s*([-+]?\d*\.?\d+)', conversation)
        # 匹配 "Angular velocity: -0.067 rad/s"
        angular_match = re.search(r'Angular velocity:\s*([-+]?\d*\.?\d+)', conversation)

        linear_vel = float(linear_match.group(1)) if linear_match else 0.0
        angular_vel = float(angular_match.group(1)) if angular_match else 0.0

        return linear_vel, angular_vel

    def _load_image(self, image_path: str) -> Image.Image:
        """加载并返回 PIL 图像，如果不存在返回 None"""
        full_path = os.path.join(self.image_folder, image_path)

        if not os.path.exists(full_path):
            return None

        try:
            return Image.open(full_path).convert('RGB')
        except Exception as e:
            print(f"Warning: Failed to load image {full_path}: {e}")
            return None

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # 尝试最多3次找到有效样本
        max_attempts = 3
        for attempt in range(max_attempts):
            try:
                item = self.data[(idx + attempt) % len(self.data)]

                # 1. 加载当前帧图像
                current_image_path = item['images'][0]
                current_image = self._load_image(current_image_path)

                if current_image is None:
                    print(f"Warning: Current image not found: {current_image_path}, trying next sample...")
                    continue

                current_image = self.transform(current_image)  # [3, H, W]

                # 2. 加载历史帧图像
                history_images = []
                if 'history_images' in item and len(item['history_images']) > 0:
                    # 取最后 num_history_frames 帧（不包括当前帧）
                    history_paths = item['history_images'][:-1]  # 去掉最后一帧（通常是当前帧的重复）
                    history_paths = history_paths[-self.num_history_frames:]  # 取最近的N帧

                    for hist_path in history_paths:
                        hist_img = self._load_image(hist_path)
                        if hist_img is not None:
                            hist_img = self.transform(hist_img)
                            history_images.append(hist_img)
                        else:
                            # 如果历史帧不存在，用当前帧填充
                            history_images.append(current_image)

                # 如果历史帧不够，用当前帧填充
                while len(history_images) < self.num_history_frames:
                    history_images.append(current_image)

                # Stack: [num_history, 3, H, W]
                history_images = torch.stack(history_images[:self.num_history_frames])

                # 3. 解析速度状态
                conversation = item['conversations'][0] if item['conversations'] else ""
                linear_vel, angular_vel = self._parse_velocity_from_conversation(conversation)
                velocity_state = torch.tensor([linear_vel, angular_vel], dtype=torch.float32)

                # 4. 获取参数标签
                parameters = torch.tensor(item['parameters'], dtype=torch.float32)

                # 5. 归一化参数（如果需要）
                if self.normalize_params and self.param_mean is not None:
                    parameters = (parameters - torch.from_numpy(self.param_mean).float()) / \
                                torch.from_numpy(self.param_std).float()

                # 成功加载，返回数据
                return {
                    'id': item['id'],
                    'current_image': current_image,        # [3, H, W]
                    'history_images': history_images,      # [num_history, 3, H, W]
                    'velocity_state': velocity_state,      # [2]
                    'parameters': parameters               # [num_params]
                }

            except Exception as e:
                print(f"Warning: Error loading sample {idx + attempt}: {e}, trying next sample...")
                continue

        # 如果所有尝试都失败了，抛出错误
        raise RuntimeError(f"Failed to load valid sample after {max_attempts} attempts starting from index {idx}")


def collate_fn(batch):
    """自定义 collate 函数"""
    ids = [item['id'] for item in batch]
    current_images = torch.stack([item['current_image'] for item in batch])
    history_images = torch.stack([item['history_images'] for item in batch])
    velocity_states = torch.stack([item['velocity_state'] for item in batch])
    parameters = torch.stack([item['parameters'] for item in batch])

    return {
        'ids': ids,
        'current_images': current_images,
        'history_images': history_images,
        'velocity_states': velocity_states,
        'parameters': parameters
    }


class InferenceDataset(Dataset):
    """推理时使用的数据集（单张图像）"""
    def __init__(self, image_size=224):
        self.transform = T.Compose([
            T.Resize((image_size, image_size)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def transform_image(self, image_path: str):
        """转换单张图像"""
        image = Image.open(image_path).convert('RGB')
        return self.transform(image)


if __name__ == "__main__":
    # 测试数据集
    import sys

    # 测试路径（需要根据实际情况修改）
    json_path = "path/to/train.json"
    image_folder = "path/to/images"

    if not os.path.exists(json_path):
        print(f"Test skipped: {json_path} not found")
        sys.exit(0)

    dataset = NavigationDataset(
        json_path=json_path,
        image_folder=image_folder,
        num_history_frames=2,
        normalize_params=True
    )

    print(f"\nDataset size: {len(dataset)}")

    # 测试加载一个样本
    sample = dataset[0]
    print(f"\nSample keys: {sample.keys()}")
    print(f"ID: {sample['id']}")
    print(f"Current image shape: {sample['current_image'].shape}")
    print(f"History images shape: {sample['history_images'].shape}")
    print(f"Velocity state: {sample['velocity_state']}")
    print(f"Parameters shape: {sample['parameters'].shape}")
    print(f"Parameters: {sample['parameters']}")

    # 测试 DataLoader
    from torch.utils.data import DataLoader

    dataloader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=True,
        num_workers=2,
        collate_fn=collate_fn
    )

    batch = next(iter(dataloader))
    print(f"\nBatch current_images shape: {batch['current_images'].shape}")
    print(f"Batch history_images shape: {batch['history_images'].shape}")
    print(f"Batch velocity_states shape: {batch['velocity_states'].shape}")
    print(f"Batch parameters shape: {batch['parameters'].shape}")
