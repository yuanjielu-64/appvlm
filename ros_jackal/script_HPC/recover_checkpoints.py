#!/usr/bin/env python3
"""
批量恢复 DeepSpeed checkpoint 中的空 tensor
适用于被 DeepSpeed ZeRO-3 bug 影响的 checkpoints
"""

import os
import sys
import torch
import numpy as np
from pathlib import Path
from safetensors.torch import save_file, load_file

def check_checkpoint_has_empty_tensors(checkpoint_dir):
    """检查 checkpoint 是否有空 tensor"""
    adapter_path = os.path.join(checkpoint_dir, 'adapter_model.safetensors')

    if not os.path.exists(adapter_path):
        return None, "No adapter_model.safetensors found"

    try:
        state_dict = load_file(adapter_path)
        total = len(state_dict)
        empty = sum(1 for v in state_dict.values() if v.numel() == 0)
        non_empty = total - empty

        return {
            'total': total,
            'empty': empty,
            'non_empty': non_empty,
            'percentage': (non_empty / total * 100) if total > 0 else 0
        }, None
    except Exception as e:
        return None, str(e)

def has_deepspeed_checkpoint(checkpoint_dir):
    """检查是否有 DeepSpeed checkpoint 文件"""
    zero_script = os.path.join(checkpoint_dir, 'zero_to_fp32.py')
    global_step_dir = None

    # 查找 global_stepXXXX 目录
    for item in os.listdir(checkpoint_dir):
        if item.startswith('global_step'):
            potential_dir = os.path.join(checkpoint_dir, item)
            if os.path.isdir(potential_dir):
                global_step_dir = potential_dir
                break

    return os.path.exists(zero_script) and global_step_dir is not None, global_step_dir

def recover_single_checkpoint(checkpoint_dir, force=False):
    """恢复单个 checkpoint"""
    print(f"\n{'='*60}")
    print(f"Processing: {checkpoint_dir}")
    print('='*60)

    # 1. 检查是否有空 tensor
    status, error = check_checkpoint_has_empty_tensors(checkpoint_dir)

    if error:
        print(f"❌ Error checking checkpoint: {error}")
        return False

    if status:
        print(f"LoRA weights status:")
        print(f"  Total: {status['total']}")
        print(f"  Non-empty: {status['non_empty']} ({status['percentage']:.1f}%)")
        print(f"  Empty: {status['empty']}")

        if status['percentage'] > 90 and not force:
            print("✓ Checkpoint looks good (>90% weights present), skipping...")
            return True

    # 2. 检查是否有 DeepSpeed checkpoint
    has_ds, global_step_dir = has_deepspeed_checkpoint(checkpoint_dir)

    if not has_ds:
        print("❌ No DeepSpeed checkpoint found, cannot recover")
        print("   This checkpoint is corrupted and cannot be fixed.")
        return False

    print(f"✓ Found DeepSpeed checkpoint: {global_step_dir}")

    # 3. 运行 zero_to_fp32.py
    print("\nStep 1: Running zero_to_fp32.py to recover full weights...")

    zero_script = os.path.join(checkpoint_dir, 'zero_to_fp32.py')
    recovered_path = os.path.join(checkpoint_dir, 'recovered_checkpoint.pt')

    # 修复 zero_to_fp32.py 中的 torch.load 问题
    import subprocess
    fix_cmd = f"sed -i 's/torch.load(f, map_location=device)/torch.load(f, map_location=device, weights_only=False)/g' {zero_script}"
    subprocess.run(fix_cmd, shell=True, check=False)

    # 运行恢复脚本
    recover_cmd = f"cd {checkpoint_dir} && python zero_to_fp32.py . recovered_checkpoint.pt"
    result = subprocess.run(recover_cmd, shell=True, capture_output=True, text=True)

    if result.returncode != 0 or not os.path.exists(recovered_path):
        print(f"❌ Recovery failed: {result.stderr}")
        return False

    print("✓ Recovered full checkpoint")

    # 4. 提取并保存 LoRA 权重
    print("\nStep 2: Extracting LoRA weights...")

    state_dict = torch.load(recovered_path, map_location='cpu', weights_only=False)

    # 提取 LoRA
    lora_dict = {}
    for k, v in state_dict.items():
        if 'lora' in k:
            clean_key = k.replace('base_model.model.', '')
            if not clean_key.startswith('base_model.model.'):
                peft_key = f'base_model.model.{clean_key}'
            else:
                peft_key = clean_key
            lora_dict[peft_key] = v

    print(f"✓ Extracted {len(lora_dict)} LoRA weights")

    # 备份原文件
    adapter_path = os.path.join(checkpoint_dir, 'adapter_model.safetensors')
    backup_path = os.path.join(checkpoint_dir, 'adapter_model_corrupted.safetensors.bak')

    if os.path.exists(adapter_path):
        os.rename(adapter_path, backup_path)
        print(f"✓ Backed up corrupted file to {backup_path}")

    # 保存新的 LoRA 权重
    save_file(lora_dict, adapter_path)
    print(f"✓ Saved recovered LoRA weights to {adapter_path}")

    # 5. 提取并保存 regression_head
    print("\nStep 3: Extracting regression_head...")

    regression_dict = {}
    for k, v in state_dict.items():
        if 'regression_head' in k:
            clean_key = k.replace('regression_head.', '')
            regression_dict[clean_key] = v

    if regression_dict:
        print(f"✓ Extracted {len(regression_dict)} regression_head weights")

        reg_dir = os.path.join(checkpoint_dir, 'regression_head')
        reg_path = os.path.join(reg_dir, 'pytorch_model.bin')
        reg_backup = os.path.join(reg_dir, 'pytorch_model_corrupted.bin.bak')

        if os.path.exists(reg_path):
            os.rename(reg_path, reg_backup)
            print(f"✓ Backed up corrupted regression_head")

        torch.save(regression_dict, reg_path)
        print(f"✓ Saved recovered regression_head to {reg_path}")

    # 6. 验证恢复结果
    print("\nStep 4: Verifying recovery...")

    status, error = check_checkpoint_has_empty_tensors(checkpoint_dir)
    if status:
        print(f"✓ Recovery complete!")
        print(f"  Final status: {status['non_empty']}/{status['total']} weights ({status['percentage']:.1f}%)")

    return True

def scan_and_recover_all(base_dir, planners=['ddp', 'dwa', 'teb', 'mppi'], dry_run=False):
    """扫描并恢复所有 checkpoints"""
    print("="*60)
    print("Checkpoint Recovery Tool")
    print("="*60)

    if dry_run:
        print("DRY RUN MODE - will only scan, not modify files")

    # 扫描所有 checkpoint
    checkpoints_to_recover = []

    for planner in planners:
        planner_dir = os.path.join(base_dir, planner)
        if not os.path.exists(planner_dir):
            continue

        # 遍历 planner 目录下的所有子目录（如 qwen2.5-vl-regression_lora-True_ddp_regression）
        for subdir in os.listdir(planner_dir):
            subdir_path = os.path.join(planner_dir, subdir)
            if not os.path.isdir(subdir_path):
                continue

            # 在子目录中查找 checkpoint-XXXX
            for item in os.listdir(subdir_path):
                if item.startswith('checkpoint-'):
                    checkpoint_dir = os.path.join(subdir_path, item)
                    if os.path.isdir(checkpoint_dir):
                        checkpoints_to_recover.append({
                            'planner': planner,
                            'subdir': subdir,
                            'name': item,
                            'path': checkpoint_dir
                        })

    print(f"\nFound {len(checkpoints_to_recover)} checkpoints to check:")
    for ckpt in checkpoints_to_recover:
        print(f"  - {ckpt['planner']}/{ckpt['subdir']}/{ckpt['name']}")

    if dry_run:
        print("\n--- DRY RUN: Checking status only ---")
        for ckpt in checkpoints_to_recover:
            status, error = check_checkpoint_has_empty_tensors(ckpt['path'])
            if status:
                needs_recovery = status['percentage'] < 90
                status_symbol = "❌" if needs_recovery else "✓"
                print(f"{status_symbol} {ckpt['planner']}/{ckpt['subdir']}/{ckpt['name']}: {status['percentage']:.1f}% complete")
            else:
                print(f"⚠️  {ckpt['planner']}/{ckpt['subdir']}/{ckpt['name']}: {error}")
        return

    # 开始恢复
    print(f"\n{'='*60}")
    print("Starting recovery process...")
    print('='*60)

    success_count = 0
    skip_count = 0
    fail_count = 0

    for i, ckpt in enumerate(checkpoints_to_recover, 1):
        print(f"\n[{i}/{len(checkpoints_to_recover)}] {ckpt['planner']}/{ckpt['subdir']}/{ckpt['name']}")

        result = recover_single_checkpoint(ckpt['path'])

        if result:
            success_count += 1
        else:
            fail_count += 1

    # 总结
    print(f"\n{'='*60}")
    print("RECOVERY SUMMARY")
    print('='*60)
    print(f"Total checkpoints: {len(checkpoints_to_recover)}")
    print(f"✓ Successfully recovered: {success_count}")
    print(f"❌ Failed to recover: {fail_count}")
    print('='*60)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Recover corrupted DeepSpeed checkpoints")
    parser.add_argument("--base_dir", type=str,
                       default="/scratch/ylu22/appvlm_ws/src/ros_jackal/model",
                       help="Base model directory")
    parser.add_argument("--checkpoint", type=str,
                       help="Recover single checkpoint (full path)")
    parser.add_argument("--dry_run", action="store_true",
                       help="Only scan, don't modify files")
    parser.add_argument("--planners", nargs='+',
                       default=['ddp', 'dwa', 'teb', 'mppi'],
                       help="Planners to scan")

    args = parser.parse_args()

    if args.checkpoint:
        # 恢复单个 checkpoint
        recover_single_checkpoint(args.checkpoint)
    else:
        # 批量恢复
        scan_and_recover_all(args.base_dir, args.planners, args.dry_run)
