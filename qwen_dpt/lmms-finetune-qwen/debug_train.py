#!/usr/bin/env python
"""
简化的调试脚本 - 不使用 DeepSpeed 和 torchrun
直接在 PyCharm 中打断点调试
"""

import sys
import os

# 模拟命令行参数
sys.argv = [
    'debug_train.py',
    '--model_id', 'qwen2.5-vl-regression',
    '--planner', 'ddp',
    '--head_type', 'dpt',  # 选项: simple_mlp, transformer, dpt
    '--use_history', 'True',  # 启用历史帧
    '--num_history_frames', '4',  # 使用 2 个历史帧
    '--history_dim', '256',  # 历史特征维度
    '--history_image_size', '224',  # 历史帧图像大小
    '--label_noise_std', '0.02',  # Label 噪音标准差 (0.0=无噪音, 0.02=轻微, 0.05=中等)
    '--data_path', '/home/yuanjielu/robot_navigation/noetic/app_data/ddp_heurstic/splits_200k/chunk_000.json',
    '--image_folder', '/home/yuanjielu/robot_navigation/noetic/app_data/ddp_heurstic',
    '--video_folder', './example_data/videos',
    '--num_frames', '8',
    '--output_dir', '../../ros_jackal/model/ddp/debug_test',
    '--report_to', 'wandb',  # 改为 wandb
    '--run_name', 'debug_test',
    # 不使用 DeepSpeed，直接用 PyTorch
    # '--deepspeed', './ds_configs/zero2.json',
    '--bf16', 'True',  # 开启 bf16 省一半显存
    '--num_train_epochs', '1',
    '--per_device_train_batch_size', '1',  # batch_size=1 最小
    '--per_device_eval_batch_size', '1',  # 评估也用batch_size=1
    '--gradient_accumulation_steps', '1',  # 减少累积步数
    '--eval_strategy', 'steps',
    '--eval_steps', '50',  # 每50步评估一次
    '--max_eval_samples', '50',  # 只评估50个样本
    '--save_strategy', 'steps',  # 每N步保存
    '--save_steps', '10',  # 每10步保存一次
    '--save_total_limit', '5',  # 只保留最近5个checkpoint
    '--learning_rate', '1e-5',  # 降低 10 倍 (1e-4 → 1e-5)
    '--weight_decay', '0.01',  # 添加 weight decay 稳定训练
    '--warmup_ratio', '0.03',
    '--lr_scheduler_type', 'cosine',
    '--logging_steps', '1',
    '--tf32', 'True',
    '--model_max_length', '590',  # 减少序列长度 (从 600 → 512)
    '--gradient_checkpointing', 'True',  # 开启梯度检查点省显存
    '--max_grad_norm', '1.0',  # 梯度裁剪，防止梯度爆炸
    '--dataloader_num_workers', '0',  # 调试时使用 0
    '--train_vision_encoder', 'False',  # 冻结 vision encoder
    '--use_vision_lora', 'False',
    '--train_vision_projector', 'False',  # 冻结 vision projector
    '--use_lora', 'True',  # 必须开启 LoRA！
    '--q_lora', 'True',  # 必须开启 4-bit 量化！
    '--lora_r', '8',  # 进一步减少 rank (16 → 8)
    '--lora_alpha', '16',  # 调整 alpha (保持 2*r)
    '--max_steps', '100000',  # 只运行 100 步用于调试
]

# 设置环境变量
os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # 只使用第一张 GPU

# 导入并运行训练脚本
if __name__ == '__main__':
    # 在这里打断点，可以逐步调试
    from train_regression import train
    train()
