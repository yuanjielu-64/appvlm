"""
训练脚本
"""

import os
import json
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
from collections import deque
import wandb
from pathlib import Path

from model import NavigationTransformerIL
from dataset import NavigationDataset, collate_fn
from config import get_config
from planner_configs import get_param_names


class MetricsTracker:
    """滑动窗口指标跟踪器"""

    def __init__(self, param_names, window_size=100):
        self.param_names = param_names
        self.window_size = window_size
        self.num_params = len(param_names)

        # 整体指标滑动窗口
        self.loss_window = deque(maxlen=window_size)
        self.mae_window = deque(maxlen=window_size)
        self.rmse_window = deque(maxlen=window_size)
        self.r2_window = deque(maxlen=window_size)

        # 每个参数的 MAE 滑动窗口
        self.per_param_mae_windows = [deque(maxlen=window_size) for _ in range(self.num_params)]

    def update(self, loss, mae, rmse, r2, per_param_mae):
        """更新指标"""
        self.loss_window.append(loss)
        self.mae_window.append(mae)
        self.rmse_window.append(rmse)
        self.r2_window.append(r2)

        for i, param_mae in enumerate(per_param_mae):
            self.per_param_mae_windows[i].append(param_mae)

    def get_smoothed_metrics(self):
        """获取平滑后的指标"""
        metrics = {
            'loss': np.mean(self.loss_window) if self.loss_window else 0,
            'mae': np.mean(self.mae_window) if self.mae_window else 0,
            'rmse': np.mean(self.rmse_window) if self.rmse_window else 0,
            'r2': np.mean(self.r2_window) if self.r2_window else 0,
        }

        # 每个参数的平滑 MAE
        per_param_mae = {}
        for i, name in enumerate(self.param_names):
            if self.per_param_mae_windows[i]:
                per_param_mae[name] = np.mean(self.per_param_mae_windows[i])
            else:
                per_param_mae[name] = 0
        metrics['per_param_mae'] = per_param_mae

        return metrics

    def reset(self):
        """重置所有窗口"""
        self.loss_window.clear()
        self.mae_window.clear()
        self.rmse_window.clear()
        self.r2_window.clear()
        for window in self.per_param_mae_windows:
            window.clear()


def set_seed(seed):
    """设置随机种子"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def create_optimizer(model, config):
    """创建优化器"""
    # 区分 vision encoder 和其他部分，使用不同学习率
    vision_params = []
    other_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if 'vision_encoder' in name:
            vision_params.append(param)
        else:
            other_params.append(param)

    param_groups = [
        {'params': other_params, 'lr': config.learning_rate},
        {'params': vision_params, 'lr': config.learning_rate * 0.1}  # vision encoder 学习率降低
    ]

    optimizer = optim.AdamW(param_groups, weight_decay=config.weight_decay)
    return optimizer


def create_scheduler(optimizer, config, num_training_steps):
    """创建学习率调度器"""
    warmup_steps = config.warmup_epochs * num_training_steps

    if config.lr_scheduler == "cosine":
        from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

        warmup_scheduler = LinearLR(optimizer, start_factor=0.01, total_iters=warmup_steps)
        cosine_scheduler = CosineAnnealingLR(optimizer, T_max=num_training_steps - warmup_steps)
        scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, cosine_scheduler],
                                milestones=[warmup_steps])

    elif config.lr_scheduler == "step":
        from torch.optim.lr_scheduler import StepLR
        scheduler = StepLR(optimizer, step_size=10 * num_training_steps, gamma=0.5)

    elif config.lr_scheduler == "plateau":
        from torch.optim.lr_scheduler import ReduceLROnPlateau
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    else:
        scheduler = None

    return scheduler


def compute_metrics(predictions, targets, param_names=None):
    """计算评估指标

    Args:
        predictions: [B, num_params] 预测值
        targets: [B, num_params] 目标值
        param_names: 参数名称列表（可选，用于生成命名指标）

    Returns:
        dict: 包含 mse, mae, rmse, r2, per_param_mae 等指标
    """
    # predictions, targets: [B, num_params]
    mse = torch.mean((predictions - targets) ** 2)
    mae = torch.mean(torch.abs(predictions - targets))
    rmse = torch.sqrt(mse)

    # 按参数维度计算 MAE 和 RMSE
    per_param_mae = torch.mean(torch.abs(predictions - targets), dim=0)
    per_param_rmse = torch.sqrt(torch.mean((predictions - targets) ** 2, dim=0))

    # R^2 score (整体)
    ss_res = torch.sum((targets - predictions) ** 2)
    ss_tot = torch.sum((targets - torch.mean(targets, dim=0, keepdim=True)) ** 2)
    r2 = 1 - ss_res / (ss_tot + 1e-8)

    # 每个参数的 R^2
    per_param_ss_res = torch.sum((targets - predictions) ** 2, dim=0)
    per_param_ss_tot = torch.sum((targets - torch.mean(targets, dim=0, keepdim=True)) ** 2, dim=0)
    per_param_r2 = 1 - per_param_ss_res / (per_param_ss_tot + 1e-8)

    metrics = {
        'mse': mse.item(),
        'mae': mae.item(),
        'rmse': rmse.item(),
        'r2': r2.item(),
        'per_param_mae': per_param_mae.cpu().numpy(),
        'per_param_rmse': per_param_rmse.cpu().numpy(),
        'per_param_r2': per_param_r2.cpu().numpy()
    }

    # 如果提供了参数名称，生成命名指标
    if param_names is not None:
        metrics['per_param_mae_dict'] = {
            name: per_param_mae[i].item()
            for i, name in enumerate(param_names)
        }
        metrics['per_param_rmse_dict'] = {
            name: per_param_rmse[i].item()
            for i, name in enumerate(param_names)
        }
        metrics['per_param_r2_dict'] = {
            name: per_param_r2[i].item()
            for i, name in enumerate(param_names)
        }

    return metrics


def train_epoch(model, dataloader, optimizer, scheduler, criterion, device, scaler, config, epoch,
                global_step_start=0, param_stats=None, param_names=None, metrics_tracker=None):
    """训练一个 epoch

    Args:
        global_step_start: 该 epoch 开始时的全局步数
        param_stats: 参数统计信息（用于保存 checkpoint）
        param_names: 参数名称列表
        metrics_tracker: MetricsTracker 实例（用于滑动窗口平滑）

    Returns:
        metrics: 该 epoch 的训练指标
        global_step: 该 epoch 结束时的全局步数
    """
    model.train()
    total_loss = 0
    all_predictions = []
    all_targets = []
    global_step = global_step_start

    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")

    for batch_idx, batch in enumerate(pbar):
        current_images = batch['current_images'].to(device)
        history_images = batch['history_images'].to(device)
        velocity_states = batch['velocity_states'].to(device)
        parameters = batch['parameters'].to(device)

        optimizer.zero_grad()

        # 混合精度训练
        if config.mixed_precision:
            with autocast():
                predictions = model(current_images, history_images, velocity_states)
                loss = criterion(predictions, parameters)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            predictions = model(current_images, history_images, velocity_states)
            loss = criterion(predictions, parameters)
            loss.backward()
            optimizer.step()

        if scheduler is not None and config.lr_scheduler != "plateau":
            scheduler.step()

        total_loss += loss.item()
        all_predictions.append(predictions.detach())
        all_targets.append(parameters.detach())
        global_step += 1

        # 计算当前 batch 的指标并更新滑动窗口
        with torch.no_grad():
            batch_metrics = compute_metrics(predictions.detach(), parameters.detach(), param_names)
            if metrics_tracker is not None:
                metrics_tracker.update(
                    loss=loss.item(),
                    mae=batch_metrics['mae'],
                    rmse=batch_metrics['rmse'],
                    r2=batch_metrics['r2'],
                    per_param_mae=batch_metrics['per_param_mae']
                )

        # 更新进度条（显示平滑后的指标）
        if metrics_tracker is not None:
            smoothed = metrics_tracker.get_smoothed_metrics()
            pbar.set_postfix({
                'loss': f"{smoothed['loss']:.4f}",
                'mae': f"{smoothed['mae']:.4f}",
                'step': global_step
            })
        else:
            pbar.set_postfix({'loss': loss.item(), 'step': global_step})

        # 日志（每 log_interval 步打印详细指标）
        if (batch_idx + 1) % config.log_interval == 0:
            avg_loss = total_loss / (batch_idx + 1)
            lr = optimizer.param_groups[0]['lr']

            # 基本信息
            print(f"\nEpoch {epoch} [{batch_idx + 1}/{len(dataloader)}] Step {global_step}")
            print(f"  Loss: {avg_loss:.6f}, LR: {lr:.6f}")

            # 打印每个参数的 MAE（当前 batch）
            if param_names is not None and 'per_param_mae_dict' in batch_metrics:
                print("  Per-param MAE (current batch):")
                for name, mae_val in batch_metrics['per_param_mae_dict'].items():
                    print(f"    {name}: {mae_val:.4f}")

            # 打印滑动窗口平均（平滑后的指标）
            if metrics_tracker is not None:
                smoothed = metrics_tracker.get_smoothed_metrics()
                print(f"  Smoothed (last {metrics_tracker.window_size} batches):")
                print(f"    Loss: {smoothed['loss']:.6f}, MAE: {smoothed['mae']:.4f}, "
                      f"RMSE: {smoothed['rmse']:.4f}, R²: {smoothed['r2']:.4f}")
                print("    Per-param MAE (smoothed):")
                for name, mae_val in smoothed['per_param_mae'].items():
                    print(f"      {name}: {mae_val:.4f}")

        # 按步数保存 checkpoint
        if config.save_steps > 0 and global_step % config.save_steps == 0:
            step_metrics = {'loss': total_loss / (batch_idx + 1)}
            # 保存 step checkpoint
            save_checkpoint(model, optimizer, scheduler, epoch, step_metrics,
                          param_stats, config.output_dir,
                          global_step=global_step, save_type="step")
            # 保存 latest
            save_checkpoint(model, optimizer, scheduler, epoch, step_metrics,
                          param_stats, config.output_dir,
                          global_step=global_step, save_type="latest")
            # 清理旧的 checkpoint
            cleanup_old_checkpoints(config.output_dir, config.save_total_limit)

    # 计算整个 epoch 的指标
    all_predictions = torch.cat(all_predictions, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    metrics = compute_metrics(all_predictions, all_targets, param_names)
    metrics['loss'] = total_loss / len(dataloader)

    return metrics, global_step


@torch.no_grad()
def evaluate(model, dataloader, criterion, device, config, param_names=None):
    """验证

    Args:
        param_names: 参数名称列表
    """
    model.eval()
    total_loss = 0
    all_predictions = []
    all_targets = []

    pbar = tqdm(dataloader, desc="Evaluating")

    for batch in pbar:
        current_images = batch['current_images'].to(device)
        history_images = batch['history_images'].to(device)
        velocity_states = batch['velocity_states'].to(device)
        parameters = batch['parameters'].to(device)

        predictions = model(current_images, history_images, velocity_states)
        loss = criterion(predictions, parameters)

        total_loss += loss.item()
        all_predictions.append(predictions)
        all_targets.append(parameters)

    all_predictions = torch.cat(all_predictions, dim=0)
    all_targets = torch.cat(all_targets, dim=0)

    metrics = compute_metrics(all_predictions, all_targets, param_names)
    metrics['loss'] = total_loss / len(dataloader)

    return metrics


def save_checkpoint(model, optimizer, scheduler, epoch, metrics, param_stats, output_dir,
                    global_step=None, is_best=False, save_type="epoch"):
    """保存 checkpoint

    Args:
        save_type: "epoch", "step", "latest", "best"
    """
    checkpoint = {
        'epoch': epoch,
        'global_step': global_step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'metrics': metrics,
        'param_stats': param_stats
    }

    if save_type == "epoch":
        checkpoint_path = os.path.join(output_dir, f'checkpoint_epoch_{epoch}.pth')
    elif save_type == "step":
        checkpoint_path = os.path.join(output_dir, f'checkpoint_step_{global_step}.pth')
    elif save_type == "latest":
        checkpoint_path = os.path.join(output_dir, 'latest.pth')
    elif save_type == "best":
        checkpoint_path = os.path.join(output_dir, 'best_model.pth')
    else:
        checkpoint_path = os.path.join(output_dir, f'checkpoint_{save_type}.pth')

    torch.save(checkpoint, checkpoint_path)
    print(f"Saved checkpoint to {checkpoint_path}")

    return checkpoint_path


def cleanup_old_checkpoints(output_dir, save_total_limit, checkpoint_prefix="checkpoint_step_"):
    """清理旧的 step checkpoint，保留最新的 N 个"""
    if save_total_limit <= 0:
        return

    # 找到所有 step checkpoint
    checkpoints = []
    for f in os.listdir(output_dir):
        if f.startswith(checkpoint_prefix) and f.endswith('.pth'):
            # 提取步数
            try:
                step = int(f.replace(checkpoint_prefix, '').replace('.pth', ''))
                checkpoints.append((step, f))
            except ValueError:
                continue

    # 按步数排序
    checkpoints.sort(key=lambda x: x[0], reverse=True)

    # 删除多余的 checkpoint
    for step, filename in checkpoints[save_total_limit:]:
        filepath = os.path.join(output_dir, filename)
        os.remove(filepath)
        print(f"Removed old checkpoint: {filepath}")


def main():
    # 获取配置
    config = get_config()

    # 设置随机种子
    set_seed(config.seed)

    # 创建输出目录
    os.makedirs(config.output_dir, exist_ok=True)

    # 保存配置
    config_path = os.path.join(config.output_dir, 'config.json')
    with open(config_path, 'w') as f:
        json.dump(vars(config), f, indent=2)

    # 初始化 wandb (可选)
    # wandb.init(project="transformer-il", config=vars(config))

    # 设置设备
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 加载数据集
    print("Loading datasets...")
    train_dataset = NavigationDataset(
        json_path=config.train_json,
        image_folder=config.image_folder,
        num_history_frames=config.num_history_frames,
        image_size=config.image_size,
        normalize_params=config.normalize_params
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )

    # 验证集
    eval_loader = None
    if config.eval_json:
        eval_dataset_full = NavigationDataset(
            json_path=config.eval_json,
            image_folder=config.image_folder,
            num_history_frames=config.num_history_frames,
            image_size=config.image_size,
            normalize_params=config.normalize_params,
            param_stats=train_dataset.get_param_stats()  # 使用训练集的统计
        )

        # 限制评估样本数量
        if config.eval_samples > 0 and len(eval_dataset_full) > config.eval_samples:
            eval_dataset = Subset(eval_dataset_full, range(config.eval_samples))
            print(f"Using {config.eval_samples} samples for evaluation (out of {len(eval_dataset_full)})")
        else:
            eval_dataset = eval_dataset_full
            print(f"Using all {len(eval_dataset_full)} samples for evaluation")

        eval_loader = DataLoader(
            eval_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers,
            collate_fn=collate_fn,
            pin_memory=True
        )

    # 保存参数统计
    param_stats_path = os.path.join(config.output_dir, 'param_stats.json')
    param_stats = train_dataset.get_param_stats()
    np.savez(param_stats_path, mean=param_stats['mean'], std=param_stats['std'])

    # 创建模型
    print("Creating model...")
    model = NavigationTransformerIL(
        num_params=config.num_params,
        num_history_frames=config.num_history_frames,
        vision_model=config.vision_model,
        vision_pretrained=config.vision_pretrained,
        vision_freeze=config.vision_freeze,
        d_model=config.d_model,
        nhead=config.nhead,
        num_transformer_layers=config.num_transformer_layers,
        dropout=config.dropout,
        use_velocity=config.use_velocity
    )
    model = model.to(device)

    print(f"Model parameters: {model.get_num_params() / 1e6:.2f}M")
    print(f"Trainable parameters: {model.get_trainable_params() / 1e6:.2f}M")

    # 创建优化器和调度器
    optimizer = create_optimizer(model, config)

    num_training_steps_per_epoch = len(train_loader)
    total_training_steps = num_training_steps_per_epoch * config.num_epochs
    scheduler = create_scheduler(optimizer, config, total_training_steps)

    # 损失函数
    criterion = nn.MSELoss()

    # 混合精度训练
    scaler = GradScaler() if config.mixed_precision else None

    # 恢复训练
    start_epoch = 0
    global_step = 0
    best_eval_loss = float('inf')

    if config.resume:
        print(f"Resuming from {config.resume}")
        checkpoint = torch.load(config.resume)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if scheduler and checkpoint['scheduler_state_dict']:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        global_step = checkpoint.get('global_step', start_epoch * num_training_steps_per_epoch)
        best_eval_loss = checkpoint['metrics'].get('loss', float('inf'))
        print(f"Resumed from epoch {start_epoch}, global_step {global_step}")

    # 获取参数名称
    param_names = get_param_names(config.planner)
    print(f"Planner: {config.planner}, Parameters: {param_names}")

    # 创建指标跟踪器
    metrics_tracker = MetricsTracker(param_names, window_size=100)

    # 训练循环
    print("Starting training...")
    print(f"Save steps: {config.save_steps}, Save total limit: {config.save_total_limit}")
    print(f"Output directory: {config.output_dir}")

    for epoch in range(start_epoch, config.num_epochs):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch + 1}/{config.num_epochs} | Global Step: {global_step}")
        print(f"{'='*60}")

        # 重置指标跟踪器（每个 epoch 开始时）
        metrics_tracker.reset()

        # 训练
        train_metrics, global_step = train_epoch(
            model, train_loader, optimizer, scheduler, criterion,
            device, scaler, config, epoch + 1,
            global_step_start=global_step, param_stats=param_stats,
            param_names=param_names, metrics_tracker=metrics_tracker
        )

        # 打印 epoch 总结
        print(f"\n{'='*60}")
        print(f"Epoch {epoch + 1} Summary")
        print(f"{'='*60}")
        print(f"Overall: Loss={train_metrics['loss']:.6f}, MAE={train_metrics['mae']:.4f}, "
              f"RMSE={train_metrics['rmse']:.4f}, R²={train_metrics['r2']:.4f}")

        # 打印每个参数的详细指标
        print("\nPer-parameter MAE:")
        if 'per_param_mae_dict' in train_metrics:
            for name, mae_val in train_metrics['per_param_mae_dict'].items():
                print(f"  {name:25s}: {mae_val:.4f}")
        else:
            for i, mae_val in enumerate(train_metrics['per_param_mae']):
                print(f"  param_{i}: {mae_val:.4f}")

        # 每个 epoch 结束保存 latest
        save_checkpoint(model, optimizer, scheduler, epoch, train_metrics,
                       param_stats, config.output_dir,
                       global_step=global_step, save_type="latest")

        # 验证
        if eval_loader and (epoch + 1) % config.eval_interval == 0:
            eval_metrics = evaluate(model, eval_loader, criterion, device, config, param_names)

            print(f"\n--- Evaluation ---")
            print(f"Overall: Loss={eval_metrics['loss']:.6f}, MAE={eval_metrics['mae']:.4f}, "
                  f"RMSE={eval_metrics['rmse']:.4f}, R²={eval_metrics['r2']:.4f}")

            print("\nPer-parameter MAE:")
            if 'per_param_mae_dict' in eval_metrics:
                for name, mae_val in eval_metrics['per_param_mae_dict'].items():
                    print(f"  {name:25s}: {mae_val:.4f}")
            else:
                for i, mae_val in enumerate(eval_metrics['per_param_mae']):
                    print(f"  param_{i}: {mae_val:.4f}")

            # 保存最佳模型
            is_best = eval_metrics['loss'] < best_eval_loss
            if is_best:
                best_eval_loss = eval_metrics['loss']
                print(f"New best model! Eval loss: {best_eval_loss:.6f}")

            if config.save_best and is_best:
                save_checkpoint(model, optimizer, scheduler, epoch, eval_metrics,
                              param_stats, config.output_dir,
                              global_step=global_step, save_type="best")

        # 按 epoch 间隔保存
        if (epoch + 1) % config.save_interval == 0:
            save_checkpoint(model, optimizer, scheduler, epoch, train_metrics,
                          param_stats, config.output_dir,
                          global_step=global_step, save_type="epoch")

    print("\nTraining completed!")
    print(f"Total steps: {global_step}")
    print(f"Best eval loss: {best_eval_loss:.6f}")
    print(f"Checkpoints saved to: {config.output_dir}")


if __name__ == "__main__":
    main()
