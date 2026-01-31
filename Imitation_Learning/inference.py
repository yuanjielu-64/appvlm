"""
推理脚本
"""

import os
import json
import argparse
import numpy as np
import torch
from PIL import Image
import torchvision.transforms as T

from model import NavigationTransformerIL


class NavigationPredictor:
    """导航参数预测器"""

    def __init__(self, checkpoint_path, device='cuda'):
        """
        Args:
            checkpoint_path: 模型 checkpoint 路径
            device: 设备
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')

        # 加载 checkpoint
        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        # 获取模型配置（从 checkpoint 或配置文件）
        config_path = os.path.join(os.path.dirname(checkpoint_path), 'config.json')
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                self.config = json.load(f)
        else:
            # 使用默认配置
            self.config = {
                'num_params': 8,
                'num_history_frames': 2,
                'vision_model': 'vit_base_patch16_224',
                'image_size': 224
            }

        # 创建模型
        self.model = NavigationTransformerIL(
            num_params=self.config.get('num_params', 8),
            num_history_frames=self.config.get('num_history_frames', 2),
            vision_model=self.config.get('vision_model', 'vit_base_patch16_224'),
            vision_pretrained=False,  # 推理时不需要
            d_model=self.config.get('d_model', 768),
            nhead=self.config.get('nhead', 8),
            num_transformer_layers=self.config.get('num_transformer_layers', 4),
            use_velocity=self.config.get('use_velocity', True)
        )

        # 加载权重
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()

        # 加载参数统计（用于反归一化）
        param_stats_path = os.path.join(os.path.dirname(checkpoint_path), 'param_stats.json.npz')
        if os.path.exists(param_stats_path):
            param_stats = np.load(param_stats_path)
            self.param_mean = torch.from_numpy(param_stats['mean']).float().to(self.device)
            self.param_std = torch.from_numpy(param_stats['std']).float().to(self.device)
        else:
            print("Warning: param_stats not found, using identity normalization")
            self.param_mean = None
            self.param_std = None

        # 图像预处理
        image_size = self.config.get('image_size', 224)
        self.transform = T.Compose([
            T.Resize((image_size, image_size)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        print("Model loaded successfully!")

    def preprocess_image(self, image_path):
        """预处理图像"""
        image = Image.open(image_path).convert('RGB')
        return self.transform(image)

    @torch.no_grad()
    def predict(self, current_image_path, history_image_paths=None,
                linear_vel=0.0, angular_vel=0.0):
        """
        预测导航参数

        Args:
            current_image_path: 当前帧图像路径
            history_image_paths: 历史帧图像路径列表 (可选)
            linear_vel: 线速度
            angular_vel: 角速度

        Returns:
            params: 预测的参数 (numpy array)
        """
        # 1. 加载当前帧
        current_image = self.preprocess_image(current_image_path).unsqueeze(0).to(self.device)

        # 2. 加载历史帧
        num_history = self.config.get('num_history_frames', 2)

        if history_image_paths and len(history_image_paths) > 0:
            history_images = []
            for hist_path in history_image_paths[-num_history:]:  # 取最后 N 帧
                if os.path.exists(hist_path):
                    hist_img = self.preprocess_image(hist_path)
                else:
                    hist_img = current_image.squeeze(0)  # fallback
                history_images.append(hist_img)
        else:
            # 没有历史帧，用当前帧填充
            history_images = [current_image.squeeze(0)] * num_history

        # Padding
        while len(history_images) < num_history:
            history_images.append(current_image.squeeze(0))

        history_images = torch.stack(history_images[:num_history]).unsqueeze(0).to(self.device)

        # 3. 速度状态
        velocity_state = torch.tensor([[linear_vel, angular_vel]], dtype=torch.float32).to(self.device)

        # 4. 前向传播
        predictions = self.model(current_image, history_images, velocity_state)

        # 5. 反归一化
        if self.param_mean is not None:
            predictions = predictions * self.param_std + self.param_mean

        # 转换为 numpy
        params = predictions.cpu().numpy().flatten()

        return params

    def predict_batch(self, image_paths, velocities):
        """批量预测"""
        results = []
        for img_path, (lin_vel, ang_vel) in zip(image_paths, velocities):
            params = self.predict(img_path, linear_vel=lin_vel, angular_vel=ang_vel)
            results.append(params)
        return np.array(results)


def main():
    parser = argparse.ArgumentParser(description="Inference for Navigation Transformer IL")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint")
    parser.add_argument("--image_path", type=str, required=True, help="Path to current image")
    parser.add_argument("--history_paths", type=str, nargs='+', default=None,
                       help="Paths to history images")
    parser.add_argument("--linear_vel", type=float, default=0.0, help="Linear velocity")
    parser.add_argument("--angular_vel", type=float, default=0.0, help="Angular velocity")
    parser.add_argument("--device", type=str, default="cuda", help="Device")

    args = parser.parse_args()

    # 创建预测器
    predictor = NavigationPredictor(args.checkpoint, device=args.device)

    # 预测
    params = predictor.predict(
        current_image_path=args.image_path,
        history_image_paths=args.history_paths,
        linear_vel=args.linear_vel,
        angular_vel=args.angular_vel
    )

    # 打印结果
    print("\nPredicted parameters:")
    print(params)

    # 保存结果
    output_path = "prediction_result.json"
    result = {
        'image_path': args.image_path,
        'velocity': {
            'linear': args.linear_vel,
            'angular': args.angular_vel
        },
        'parameters': params.tolist()
    }

    with open(output_path, 'w') as f:
        json.dump(result, f, indent=2)

    print(f"\nResult saved to {output_path}")


if __name__ == "__main__":
    main()
