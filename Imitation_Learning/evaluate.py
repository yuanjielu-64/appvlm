"""
评估脚本：在整个数据集上评估模型性能
"""

import os
import json
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt

from model import NavigationTransformerIL
from dataset import NavigationDataset, collate_fn
from utils import compare_predictions, save_predictions_to_csv


def compute_detailed_metrics(predictions, targets):
    """计算详细的评估指标"""
    # predictions, targets: [N, num_params]

    # MSE, MAE
    mse = np.mean((predictions - targets) ** 2)
    mae = np.mean(np.abs(predictions - targets))
    rmse = np.sqrt(mse)

    # 按参数维度计算
    per_param_mse = np.mean((predictions - targets) ** 2, axis=0)
    per_param_mae = np.mean(np.abs(predictions - targets), axis=0)
    per_param_rmse = np.sqrt(per_param_mse)

    # R^2 score
    ss_res = np.sum((targets - predictions) ** 2, axis=0)
    ss_tot = np.sum((targets - np.mean(targets, axis=0, keepdims=True)) ** 2, axis=0)
    r2 = 1 - ss_res / (ss_tot + 1e-8)

    # MAPE (Mean Absolute Percentage Error)
    # 避免除以0
    mape = np.mean(np.abs((targets - predictions) / (np.abs(targets) + 1e-8))) * 100

    # Max error
    max_error = np.max(np.abs(predictions - targets), axis=0)

    return {
        'mse': mse,
        'mae': mae,
        'rmse': rmse,
        'mape': mape,
        'per_param_mse': per_param_mse,
        'per_param_mae': per_param_mae,
        'per_param_rmse': per_param_rmse,
        'per_param_r2': r2,
        'per_param_max_error': max_error
    }


@torch.no_grad()
def evaluate_model(model, dataloader, device, param_mean=None, param_std=None):
    """在数据集上评估模型"""
    model.eval()

    all_predictions = []
    all_targets = []
    all_ids = []

    for batch in tqdm(dataloader, desc="Evaluating"):
        current_images = batch['current_images'].to(device)
        history_images = batch['history_images'].to(device)
        velocity_states = batch['velocity_states'].to(device)
        parameters = batch['parameters'].to(device)

        # 前向传播
        predictions = model(current_images, history_images, velocity_states)

        all_predictions.append(predictions.cpu())
        all_targets.append(parameters.cpu())
        all_ids.extend(batch['ids'])

    # 合并所有批次
    all_predictions = torch.cat(all_predictions, dim=0).numpy()
    all_targets = torch.cat(all_targets, dim=0).numpy()

    # 反归一化
    if param_mean is not None and param_std is not None:
        all_predictions = all_predictions * param_std + param_mean
        all_targets = all_targets * param_std + param_mean

    # 计算指标
    metrics = compute_detailed_metrics(all_predictions, all_targets)

    return metrics, all_predictions, all_targets, all_ids


def main():
    parser = argparse.ArgumentParser(description="Evaluate Transformer IL Model")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint")
    parser.add_argument("--test_json", type=str, required=True, help="Path to test JSON")
    parser.add_argument("--image_folder", type=str, required=True, help="Path to image folder")
    parser.add_argument("--output_dir", type=str, default="./eval_results", help="Output directory")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of workers")
    parser.add_argument("--device", type=str, default="cuda", help="Device")

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # 设置设备
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 加载 checkpoint
    print(f"Loading checkpoint from {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device)

    # 加载配置
    config_path = os.path.join(os.path.dirname(args.checkpoint), 'config.json')
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = json.load(f)
    else:
        config = {
            'num_params': 8,
            'num_history_frames': 2,
            'vision_model': 'vit_base_patch16_224',
            'image_size': 224,
            'd_model': 768,
            'nhead': 8,
            'num_transformer_layers': 4,
            'use_velocity': True
        }

    # 加载参数统计
    param_stats_path = os.path.join(os.path.dirname(args.checkpoint), 'param_stats.json.npz')
    if os.path.exists(param_stats_path):
        param_stats = np.load(param_stats_path)
        param_mean = param_stats['mean']
        param_std = param_stats['std']
        print(f"Loaded parameter statistics")
    else:
        param_mean = None
        param_std = None
        print("Warning: parameter statistics not found")

    # 创建数据集
    print("Loading test dataset...")
    test_dataset = NavigationDataset(
        json_path=args.test_json,
        image_folder=args.image_folder,
        num_history_frames=config.get('num_history_frames', 2),
        image_size=config.get('image_size', 224),
        normalize_params=True,
        param_stats={'mean': param_mean, 'std': param_std} if param_mean is not None else None
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )

    # 创建模型
    print("Creating model...")
    model = NavigationTransformerIL(
        num_params=config.get('num_params', 8),
        num_history_frames=config.get('num_history_frames', 2),
        vision_model=config.get('vision_model', 'vit_base_patch16_224'),
        vision_pretrained=False,
        d_model=config.get('d_model', 768),
        nhead=config.get('nhead', 8),
        num_transformer_layers=config.get('num_transformer_layers', 4),
        use_velocity=config.get('use_velocity', True)
    )

    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)

    # 评估
    print("Evaluating...")
    metrics, predictions, targets, ids = evaluate_model(
        model, test_loader, device, param_mean, param_std
    )

    # 打印结果
    print("\n" + "="*60)
    print("Evaluation Results")
    print("="*60)
    print(f"MSE:  {metrics['mse']:.6f}")
    print(f"MAE:  {metrics['mae']:.6f}")
    print(f"RMSE: {metrics['rmse']:.6f}")
    print(f"MAPE: {metrics['mape']:.2f}%")

    print("\nPer-parameter metrics:")
    num_params = len(metrics['per_param_mae'])
    for i in range(num_params):
        print(f"\nParameter {i}:")
        print(f"  MAE:  {metrics['per_param_mae'][i]:.6f}")
        print(f"  RMSE: {metrics['per_param_rmse'][i]:.6f}")
        print(f"  R^2:  {metrics['per_param_r2'][i]:.6f}")
        print(f"  Max Error: {metrics['per_param_max_error'][i]:.6f}")

    # 保存指标
    metrics_path = os.path.join(args.output_dir, 'metrics.json')
    metrics_dict = {
        'mse': float(metrics['mse']),
        'mae': float(metrics['mae']),
        'rmse': float(metrics['rmse']),
        'mape': float(metrics['mape']),
        'per_param_mae': metrics['per_param_mae'].tolist(),
        'per_param_rmse': metrics['per_param_rmse'].tolist(),
        'per_param_r2': metrics['per_param_r2'].tolist(),
        'per_param_max_error': metrics['per_param_max_error'].tolist()
    }

    with open(metrics_path, 'w') as f:
        json.dump(metrics_dict, f, indent=2)
    print(f"\nMetrics saved to {metrics_path}")

    # 保存预测结果
    csv_path = os.path.join(args.output_dir, 'predictions.csv')
    save_predictions_to_csv(predictions, targets, ids, csv_path)

    # 绘制对比图
    plot_path = os.path.join(args.output_dir, 'prediction_comparison.png')
    compare_predictions(predictions, targets, save_path=plot_path)

    print(f"\nEvaluation completed! Results saved to {args.output_dir}")


if __name__ == "__main__":
    main()
