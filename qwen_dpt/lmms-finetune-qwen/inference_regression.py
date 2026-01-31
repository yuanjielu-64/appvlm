"""
Inference script for Qwen2.5-VL + DPT Head Regression

用于推理：从 costmap/图像预测导航参数
"""

import torch
from PIL import Image
from transformers import AutoProcessor
import numpy as np
import argparse
from pathlib import Path
import sys

# 添加 models 目录到 path
sys.path.insert(0, str(Path(__file__).parent))
from models import Qwen2_5_VLForRegression


def load_model(model_path: str, device: str = "cuda"):
    """
    加载训练好的回归模型

    Args:
        model_path: 模型保存路径
        device: 设备

    Returns:
        model, processor
    """
    print(f"Loading model from {model_path}...")

    # 加载模型
    model = Qwen2_5_VLForRegression.from_pretrained(model_path)
    model = model.to(device)
    model.eval()

    # 加载 processor
    processor = AutoProcessor.from_pretrained(
        Path(model_path) / "base_model"
    )

    print("Model loaded successfully!")
    return model, processor


def predict_parameters(
    model,
    processor,
    image_path: str,
    prompt: str = "Predict navigation parameters for this costmap.",
    device: str = "cuda"
):
    """
    预测导航参数

    Args:
        model: 回归模型
        processor: 图像和文本处理器
        image_path: 输入图像路径
        prompt: 文本提示
        device: 设备

    Returns:
        parameters: [7] 预测的参数
    """
    # 加载图像
    image = Image.open(image_path).convert('RGB')

    # 构建输入文本
    text = f"<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>\n{prompt}\n<|im_end|>\n"

    # 处理输入
    inputs = processor(
        text=[text],
        images=[image],
        return_tensors="pt"
    )

    # 移到设备
    inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v
              for k, v in inputs.items()}

    # 推理
    with torch.no_grad():
        predictions = model.predict(**inputs)

    # 转为 numpy
    parameters = predictions.cpu().numpy()[0]

    return parameters


def denormalize_parameters(
    parameters: np.ndarray,
    param_mean: np.ndarray,
    param_std: np.ndarray
) -> np.ndarray:
    """
    反归一化参数

    Args:
        parameters: 归一化后的参数
        param_mean: 训练时的均值
        param_std: 训练时的标准差

    Returns:
        原始尺度的参数
    """
    return parameters * param_std + param_mean


def main():
    parser = argparse.ArgumentParser(description="Navigation Parameter Regression Inference")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to trained model")
    parser.add_argument("--image_path", type=str, required=True,
                        help="Path to input costmap/image")
    parser.add_argument("--param_mean", type=str, default=None,
                        help="Path to param_mean.npy (for denormalization)")
    parser.add_argument("--param_std", type=str, default=None,
                        help="Path to param_std.npy (for denormalization)")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to use (cuda/cpu)")
    parser.add_argument("--prompt", type=str,
                        default="Predict navigation parameters for this costmap.",
                        help="Text prompt")

    args = parser.parse_args()

    # 加载模型
    model, processor = load_model(args.model_path, args.device)

    # 预测
    print(f"\nPredicting parameters for: {args.image_path}")
    parameters = predict_parameters(
        model, processor, args.image_path, args.prompt, args.device
    )

    print("\nPredicted parameters (normalized):")
    param_names = [
        "max_vel_x",
        "max_vel_theta",
        "vx_samples",
        "vtheta_samples",
        "occdist_scale",
        "pdist_scale",
        "gdist_scale"
    ]
    for name, value in zip(param_names, parameters):
        print(f"  {name:20s}: {value:.4f}")

    # 反归一化（如果提供了统计信息）
    if args.param_mean and args.param_std:
        param_mean = np.load(args.param_mean)
        param_std = np.load(args.param_std)
        parameters_original = denormalize_parameters(
            parameters, param_mean, param_std
        )

        print("\nPredicted parameters (original scale):")
        for name, value in zip(param_names, parameters_original):
            print(f"  {name:20s}: {value:.4f}")

        return parameters_original
    else:
        print("\nNote: No normalization stats provided. Showing normalized values.")
        return parameters


if __name__ == "__main__":
    main()
