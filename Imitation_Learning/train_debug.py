#!/usr/bin/env python3
"""
训练脚本 - PyCharm 调试版本
可以直接在 PyCharm 里设置断点调试

使用方法:
1. 在 PyCharm 里打开这个文件
2. 修改下面的 DEBUG_CONFIG 字典来配置训练参数
3. 点击右键 -> Debug 'train_debug' 或者设置断点后 Debug
"""

import sys
from pathlib import Path

# 将当前目录添加到 Python 路径
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

from train import main


if __name__ == '__main__':
    # ============================================================
    # 调试配置 - 直接在这里修改参数！
    # ============================================================
    DEBUG_CONFIG = {
        # 主要配置
        'planner': 'ddp',  # 选择: dwa, teb, mppi, ddp

        # 数据路径 (设置为 None 会自动根据 planner 生成路径)
        'data_root': '/home/yuanjielu/robot_navigation/noetic/app_data',
        'train_json': None,  # 自动生成: {data_root}/{planner}_heurstic/splits_200k/chunk_000.json
        'eval_json': None,   # 自动生成: {data_root}/{planner}_heurstic/splits_200k/chunk_000.json
        'image_folder': None,  # 自动生成: {data_root}/{planner}_heurstic/

        # 训练配置
        'num_history_frames': 2,
        'batch_size': 16,
        'num_epochs': 50,
        'learning_rate': 1e-4,

        # 模型配置
        'vision_model': 'vit_base_patch16_224',
        'num_transformer_layers': 4,

        # 优化器配置
        'weight_decay': 0.01,
        'lr_scheduler': 'cosine',
        'warmup_epochs': 2,

        # 其他选项
        'normalize_params': True,
        'use_velocity': True,
        'vision_pretrained': True,
        'mixed_precision': True,
        'num_workers': 4,
        'log_interval': 50,
        'eval_interval': 1,
        'save_interval': 5,
        'save_best': True,
        'device': 'cuda',
        'seed': 42,
    }

    # ============================================================
    # 调试模式选项
    # ============================================================
    DEBUG_OPTIONS = {
        # 快速测试模式 - 只训练几个 batch
        'quick_test': False,
        'quick_test_batches': 5,

        # 数据加载测试 - 只加载数据不训练
        'test_dataloader_only': False,

        # 单步调试 - 在第一个 batch 后暂停
        'pause_after_first_batch': False,

        # 详细日志
        'verbose': True,
    }

    # ============================================================
    # 构建命令行参数
    # ============================================================
    args = []

    for key, value in DEBUG_CONFIG.items():
        if value is None:
            continue

        arg_name = f'--{key}'

        # 布尔值参数
        if isinstance(value, bool):
            if value:
                args.append(arg_name)
        else:
            args.extend([arg_name, str(value)])

    # 打印配置信息
    if DEBUG_OPTIONS.get('verbose', True):
        print("=" * 80)
        print("DEBUG CONFIGURATION")
        print("=" * 80)
        print(f"Planner: {DEBUG_CONFIG['planner']}")
        print(f"Data root: {DEBUG_CONFIG['data_root']}")
        print(f"Batch size: {DEBUG_CONFIG['batch_size']}")
        print(f"Num epochs: {DEBUG_CONFIG['num_epochs']}")
        print(f"Learning rate: {DEBUG_CONFIG['learning_rate']}")
        print(f"Device: {DEBUG_CONFIG['device']}")
        print("=" * 80)

        # 显示自动生成的路径
        planner = DEBUG_CONFIG['planner']
        data_root = DEBUG_CONFIG['data_root']

        train_json = DEBUG_CONFIG.get('train_json')
        eval_json = DEBUG_CONFIG.get('eval_json')
        image_folder = DEBUG_CONFIG.get('image_folder')

        if train_json is None:
            train_json = f"{data_root}/{planner}_heurstic/splits_200k/chunk_000.json"
            print(f"\nAuto-generated train_json: {train_json}")
        else:
            print(f"\nCustom train_json: {train_json}")

        if eval_json is None:
            eval_json = f"{data_root}/{planner}_heurstic/splits_200k/chunk_000.json"
            print(f"Auto-generated eval_json: {eval_json}")
        else:
            print(f"Custom eval_json: {eval_json}")

        if image_folder is None:
            image_folder = f"{data_root}/{planner}_heurstic/"
            print(f"Auto-generated image_folder: {image_folder}")
        else:
            print(f"Custom image_folder: {image_folder}")

        # 检查路径是否存在
        print("\nPATH VALIDATION:")
        print(f"Train JSON exists: {Path(train_json).exists()} ({train_json})")
        print(f"Eval JSON exists: {Path(eval_json).exists()} ({eval_json})")
        print(f"Image folder exists: {Path(image_folder).exists()} ({image_folder})")
        print("=" * 80)
        print()

    # ============================================================
    # 快速测试模式
    # ============================================================
    if DEBUG_OPTIONS.get('quick_test', False):
        print(f"\n[QUICK TEST MODE] Only running {DEBUG_OPTIONS['quick_test_batches']} batches")
        DEBUG_CONFIG['num_epochs'] = 1
        # 修改 sys.argv 以便 train.py 可以识别
        args.extend(['--quick_test_batches', str(DEBUG_OPTIONS['quick_test_batches'])])

    # ============================================================
    # 数据加载测试模式
    # ============================================================
    if DEBUG_OPTIONS.get('test_dataloader_only', False):
        print("\n[DATALOADER TEST MODE] Only testing data loading")
        # 这里可以直接测试数据加载
        from dataset import NavigationDataset, collate_fn
        from torch.utils.data import DataLoader
        import torch

        print("Loading dataset...")
        dataset = NavigationDataset(
            json_path=DEBUG_CONFIG['train_json'],
            image_folder=DEBUG_CONFIG['image_folder'],
            planner=DEBUG_CONFIG['planner'],
            num_history_frames=DEBUG_CONFIG['num_history_frames'],
            normalize_params=DEBUG_CONFIG['normalize_params'],
            use_velocity=DEBUG_CONFIG['use_velocity']
        )

        print(f"Dataset size: {len(dataset)}")
        print(f"Number of parameters: {dataset.num_params}")

        print("\nCreating dataloader...")
        dataloader = DataLoader(
            dataset,
            batch_size=DEBUG_CONFIG['batch_size'],
            shuffle=True,
            num_workers=DEBUG_CONFIG['num_workers'],
            collate_fn=collate_fn,
            pin_memory=True if DEBUG_CONFIG['device'] == 'cuda' else False
        )

        print(f"Number of batches: {len(dataloader)}")

        print("\nTesting first batch...")
        for batch_idx, batch in enumerate(dataloader):
            print(f"\nBatch {batch_idx}:")
            print(f"  Current images: {batch['current_image'].shape}")
            print(f"  History images: {batch['history_images'].shape}")
            print(f"  Velocity: {batch['velocity'].shape}")
            print(f"  Parameters: {batch['parameters'].shape}")

            if batch_idx >= 2:  # 只测试前3个batch
                break

        print("\n[DATALOADER TEST COMPLETE]")
        sys.exit(0)

    # ============================================================
    # 设置 sys.argv 并调用主训练函数
    # ============================================================
    sys.argv = ['train.py'] + args

    print("\n[STARTING TRAINING]")
    print("You can now set breakpoints in train.py and debug!")
    print("=" * 80)
    print()

    # 调用训练主函数
    try:
        main()
    except KeyboardInterrupt:
        print("\n[TRAINING INTERRUPTED]")
    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
