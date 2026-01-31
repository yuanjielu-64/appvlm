#!/usr/bin/env python3
"""
训练 Transformer Imitation Learning 模型 - Python 调试版本
使用方法:
    python train_example.py --planner ddp
    python train_example.py --planner dwa --debug
"""

import argparse
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description='Train Transformer IL Model - Debug Version')

    # ============================================================
    # 主要配置
    # ============================================================
    parser.add_argument('--planner', type=str, default='ddp',
                        choices=['dwa', 'teb', 'mppi', 'ddp'],
                        help='Planner type')

    # ============================================================
    # 数据路径配置
    # ============================================================
    parser.add_argument('--data_root', type=str,
                        default='/home/yuanjielu/robot_navigation/noetic/app_data',
                        help='Data root directory')
    parser.add_argument('--train_json', type=str, default="/home/yuanjielu/robot_navigation/noetic/app_data/splits_200k/splits_200k",
                        help='Override default train JSON path')
    parser.add_argument('--eval_json', type=str, default="/home/yuanjielu/robot_navigation/noetic/app_data/splits_200k/splits_200k",
                        help='Override default eval JSON path')
    parser.add_argument('--image_folder', type=str, default="/home/yuanjielu/robot_navigation/noetic/app_data/",
                        help='Override default image folder')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Override default output directory')

    # ============================================================
    # 训练配置
    # ============================================================
    parser.add_argument('--num_history_frames', type=int, default=2,
                        help='Number of history frames')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size')
    parser.add_argument('--num_epochs', type=int, default=50,
                        help='Number of epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                        help='Learning rate')

    # ============================================================
    # 模型配置
    # ============================================================
    parser.add_argument('--vision_model', type=str, default='vit_base_patch16_224',
                        help='Vision Transformer model')
    parser.add_argument('--num_transformer_layers', type=int, default=4,
                        help='Number of transformer layers')

    # ============================================================
    # 调试选项
    # ============================================================
    parser.add_argument('--debug', action='store_true',
                        help='Print command without executing')
    parser.add_argument('--dry_run', action='store_true',
                        help='Dry run mode - only show paths and configuration')

    args = parser.parse_args()

    # ============================================================
    # 构建训练命令
    # ============================================================
    train_cmd = [
        'python', 'train.py',
        '--planner', args.planner,
        '--data_root', args.data_root,
        '--num_history_frames', str(args.num_history_frames),
        '--batch_size', str(args.batch_size),
        '--num_epochs', str(args.num_epochs),
        '--learning_rate', str(args.learning_rate),
        '--vision_model', args.vision_model,
        '--num_transformer_layers', str(args.num_transformer_layers),
        '--normalize_params',
        '--use_velocity',
        '--vision_pretrained',
        '--mixed_precision',
        '--num_workers', '4',
        '--log_interval', '50',
        '--eval_interval', '1',
        '--save_interval', '5',
        '--save_best',
        '--device', 'cuda'
    ]

    # 添加可选路径参数
    if args.train_json:
        train_cmd.extend(['--train_json', args.train_json])
    if args.eval_json:
        train_cmd.extend(['--eval_json', args.eval_json])
    if args.image_folder:
        train_cmd.extend(['--image_folder', args.image_folder])
    if args.output_dir:
        train_cmd.extend(['--output_dir', args.output_dir])

    # ============================================================
    # 显示配置信息
    # ============================================================
    print("=" * 60)
    print(f"Training Transformer IL for planner: {args.planner}")
    print(f"Data root: {args.data_root}")
    print("=" * 60)
    print()

    # 显示预期的路径（自动生成规则）
    if not args.train_json:
        default_train_json = f"{args.data_root}/{args.planner}_heurstic/splits_200k/chunk_000.json"
        print(f"Default train JSON: {default_train_json}")
        print(f"  Exists: {Path(default_train_json).exists()}")

    if not args.eval_json:
        default_eval_json = f"{args.data_root}/{args.planner}_heurstic/splits_200k/chunk_000.json"
        print(f"Default eval JSON: {default_eval_json}")
        print(f"  Exists: {Path(default_eval_json).exists()}")

    if not args.image_folder:
        default_image_folder = f"{args.data_root}/{args.planner}_heurstic/"
        print(f"Default image folder: {default_image_folder}")
        print(f"  Exists: {Path(default_image_folder).exists()}")

    if not args.output_dir:
        default_output_dir = f"./output/{args.planner}_transformer_il"
        print(f"Default output dir: {default_output_dir}")

    print()

    # 显示参数数量
    param_counts = {
        'dwa': 9,
        'teb': 9,
        'mppi': 10,
        'ddp': 8
    }
    print(f"Expected parameter count for {args.planner}: {param_counts[args.planner]}")
    print()

    # ============================================================
    # 显示完整命令
    # ============================================================
    print("Full command:")
    print("-" * 60)
    print(' '.join(train_cmd))
    print("-" * 60)
    print()

    # ============================================================
    # 执行或调试
    # ============================================================
    if args.dry_run:
        print("[DRY RUN] Command would be executed above")
        print("Use --debug to print command without validation")
        return 0

    if args.debug:
        print("[DEBUG MODE] Command printed above, not executing")
        print("Remove --debug flag to actually run training")
        return 0

    # 执行训练
    print("[EXECUTING] Starting training...")
    print()
    try:
        result = subprocess.run(train_cmd, check=True)
        return result.returncode
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Training failed with exit code {e.returncode}", file=sys.stderr)
        return e.returncode
    except KeyboardInterrupt:
        print("\n[INTERRUPTED] Training interrupted by user")
        return 130


if __name__ == '__main__':
    sys.exit(main())
