"""
工具函数
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


def analyze_dataset(json_path):
    """分析数据集统计信息"""
    with open(json_path, 'r') as f:
        data = json.load(f)

    print(f"Dataset: {json_path}")
    print(f"Total samples: {len(data)}")

    # 参数统计
    all_params = [item['parameters'] for item in data]
    all_params = np.array(all_params)

    print(f"\nParameter statistics:")
    print(f"Shape: {all_params.shape}")
    print(f"Mean: {np.mean(all_params, axis=0)}")
    print(f"Std: {np.std(all_params, axis=0)}")
    print(f"Min: {np.min(all_params, axis=0)}")
    print(f"Max: {np.max(all_params, axis=0)}")

    # 检查是否有历史帧
    has_history = sum(1 for item in data if 'history_images' in item and len(item['history_images']) > 0)
    print(f"\nSamples with history images: {has_history} / {len(data)}")

    # ID 前缀分布
    id_prefixes = {}
    for item in data:
        prefix = item['id'].split('_')[0]
        id_prefixes[prefix] = id_prefixes.get(prefix, 0) + 1

    print(f"\nID prefix distribution:")
    for prefix, count in sorted(id_prefixes.items()):
        print(f"  {prefix}: {count}")

    return all_params


def plot_parameter_distribution(params, save_path=None):
    """绘制参数分布"""
    num_params = params.shape[1]
    fig, axes = plt.subplots(2, (num_params + 1) // 2, figsize=(15, 8))
    axes = axes.flatten()

    for i in range(num_params):
        axes[i].hist(params[:, i], bins=50, alpha=0.7)
        axes[i].set_title(f'Parameter {i}')
        axes[i].set_xlabel('Value')
        axes[i].set_ylabel('Count')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        print(f"Saved plot to {save_path}")
    else:
        plt.show()


def plot_training_curves(log_path, save_path=None):
    """绘制训练曲线"""
    # 假设 log 是 JSON 格式
    with open(log_path, 'r') as f:
        logs = json.load(f)

    epochs = [log['epoch'] for log in logs]
    train_loss = [log['train_loss'] for log in logs]
    eval_loss = [log.get('eval_loss', None) for log in logs]

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_loss, label='Train Loss')

    if any(eval_loss):
        eval_epochs = [e for e, l in zip(epochs, eval_loss) if l is not None]
        eval_loss_clean = [l for l in eval_loss if l is not None]
        plt.plot(eval_epochs, eval_loss_clean, label='Eval Loss')

    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Curves')
    plt.legend()
    plt.grid(True)

    if save_path:
        plt.savefig(save_path)
        print(f"Saved plot to {save_path}")
    else:
        plt.show()


def compare_predictions(predictions, targets, param_names=None, save_path=None):
    """对比预测值和真实值"""
    num_params = predictions.shape[1]

    if param_names is None:
        param_names = [f'Param {i}' for i in range(num_params)]

    fig, axes = plt.subplots(2, (num_params + 1) // 2, figsize=(15, 8))
    axes = axes.flatten()

    for i in range(num_params):
        axes[i].scatter(targets[:, i], predictions[:, i], alpha=0.5, s=10)
        axes[i].plot([targets[:, i].min(), targets[:, i].max()],
                    [targets[:, i].min(), targets[:, i].max()],
                    'r--', lw=2)
        axes[i].set_xlabel('Ground Truth')
        axes[i].set_ylabel('Prediction')
        axes[i].set_title(param_names[i])
        axes[i].grid(True)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        print(f"Saved plot to {save_path}")
    else:
        plt.show()


def save_predictions_to_csv(predictions, targets, ids, output_path):
    """保存预测结果到 CSV"""
    import pandas as pd

    num_params = predictions.shape[1]

    data = {
        'id': ids
    }

    for i in range(num_params):
        data[f'pred_param_{i}'] = predictions[:, i]
        data[f'true_param_{i}'] = targets[:, i]
        data[f'error_param_{i}'] = np.abs(predictions[:, i] - targets[:, i])

    df = pd.DataFrame(data)
    df.to_csv(output_path, index=False)
    print(f"Saved predictions to {output_path}")


def compute_correlation_matrix(params, save_path=None):
    """计算参数相关性矩阵"""
    corr = np.corrcoef(params.T)

    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', center=0,
                square=True, linewidths=1)
    plt.title('Parameter Correlation Matrix')

    if save_path:
        plt.savefig(save_path)
        print(f"Saved plot to {save_path}")
    else:
        plt.show()


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python utils.py <json_path>")
        sys.exit(1)

    json_path = sys.argv[1]

    if not Path(json_path).exists():
        print(f"File not found: {json_path}")
        sys.exit(1)

    # 分析数据集
    params = analyze_dataset(json_path)

    # 绘制分布
    plot_parameter_distribution(params, save_path='param_distribution.png')

    # 计算相关性
    compute_correlation_matrix(params, save_path='param_correlation.png')
