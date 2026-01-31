#!/usr/bin/env python3
"""
从包含LoRA参数的regression_head中提取LoRA adapter

背景: DeepSpeed ZeRO训练时，regression_head的storage包含了完整的LoRA参数
本脚本从storage中提取LoRA并保存为独立的adapter_model.safetensors

Usage:
    python extract_lora_from_checkpoint.py <checkpoint_dir>

Example:
    python extract_lora_from_checkpoint.py /path/to/checkpoint-5000
"""

import torch
from safetensors.torch import save_file
from safetensors import safe_open
import os
import sys
import argparse


def load_lora_template(checkpoint_dir):
    """
    从现有的adapter_model.safetensors加载LoRA结构作为模板
    返回: (ordered_keys, shapes_dict)
    """
    adapter_path = os.path.join(checkpoint_dir, "adapter_model.safetensors")

    if not os.path.exists(adapter_path):
        raise FileNotFoundError(f"No adapter_model.safetensors found in {checkpoint_dir}")

    lora_keys = []
    lora_shapes = {}

    with safe_open(adapter_path, framework="pt", device="cpu") as f:
        for key in f.keys():
            lora_keys.append(key)
            tensor = f.get_tensor(key)
            lora_shapes[key] = tensor.shape

    return lora_keys, lora_shapes


def extract_lora_from_regression_head(checkpoint_dir, template_keys, template_shapes):
    """
    从regression_head的storage中提取LoRA参数

    Args:
        checkpoint_dir: checkpoint目录路径
        template_keys: LoRA参数的key列表（有序）
        template_shapes: LoRA参数的shape字典

    Returns:
        lora_state_dict: 提取的LoRA参数字典
    """
    reg_head_path = os.path.join(checkpoint_dir, "regression_head/pytorch_model.bin")

    if not os.path.exists(reg_head_path):
        raise FileNotFoundError(f"No regression_head found in {checkpoint_dir}")

    print(f"Loading regression_head from {reg_head_path}...")
    reg_state = torch.load(reg_head_path, map_location='cpu')

    # 获取完整的storage (包含LoRA)
    first_tensor = list(reg_state.values())[0]
    storage_size = first_tensor.storage().size()

    print(f"  Storage size: {storage_size:,} elements ({first_tensor.dtype})")

    # 创建view到整个storage
    full_storage_view = torch.as_strided(
        first_tensor,
        size=(storage_size,),
        stride=(1,),
        storage_offset=0
    )

    # 计算DPT参数占用
    dpt_params = sum(v.numel() for v in reg_state.values())
    print(f"  DPT params: {dpt_params:,}")
    print(f"  Remaining (LoRA): {storage_size - dpt_params:,}")

    # 提取LoRA (假设在storage开头)
    lora_extracted = {}
    offset = 0

    for key in template_keys:
        shape = template_shapes[key]
        numel = torch.Size(shape).numel()

        # 从storage中提取并克隆
        extracted = full_storage_view[offset:offset+numel].clone().reshape(shape)
        lora_extracted[key] = extracted

        offset += numel

    print(f"  Extracted {len(lora_extracted)} LoRA tensors, {offset:,} total elements")

    return lora_extracted


def verify_lora(lora_state_dict):
    """验证提取的LoRA是否有效"""
    print("\nVerifying extracted LoRA...")

    all_stats = []
    zero_count = 0

    for key, tensor in lora_state_dict.items():
        mean = tensor.float().mean().item()
        std = tensor.float().std().item()
        non_zero = (tensor != 0).float().mean().item()

        all_stats.append({
            'mean': mean,
            'std': std,
            'non_zero': non_zero
        })

        if non_zero < 0.01:
            zero_count += 1

    avg_std = sum(s['std'] for s in all_stats) / len(all_stats)
    avg_nonzero = sum(s['non_zero'] for s in all_stats) / len(all_stats)

    print(f"  Total tensors: {len(lora_state_dict)}")
    print(f"  Tensors with >1% non-zero: {len(lora_state_dict) - zero_count}/{len(lora_state_dict)}")
    print(f"  Average std: {avg_std:.6f}")
    print(f"  Average non-zero ratio: {avg_nonzero:.4f}")

    # 显示前3个样本
    print(f"\n  Sample tensors:")
    for i, (key, tensor) in enumerate(list(lora_state_dict.items())[:3]):
        stat = all_stats[i]
        print(f"    {key[:70]}...")
        print(f"      mean={stat['mean']:.6f}, std={stat['std']:.6f}, non_zero={stat['non_zero']:.4f}")

    # 判断
    if avg_nonzero > 0.1 and avg_std > 0.001:
        print(f"\n  ✅ LoRA appears valid (has trained weights)")
        return True
    else:
        print(f"\n  ⚠️  LoRA might be empty or not trained")
        return False


def main():
    parser = argparse.ArgumentParser(description="Extract LoRA from checkpoint regression_head")
    parser.add_argument("checkpoint_dir", help="Path to checkpoint directory")
    parser.add_argument("--output", help="Output path for extracted adapter (default: <checkpoint>/adapter_model_extracted.safetensors)")
    parser.add_argument("--backup", action="store_true", help="Backup original adapter_model.safetensors before replacing")
    parser.add_argument("--replace", action="store_true", help="Replace original adapter_model.safetensors with extracted one")

    args = parser.parse_args()

    checkpoint_dir = os.path.abspath(args.checkpoint_dir)

    if not os.path.exists(checkpoint_dir):
        print(f"❌ Checkpoint directory not found: {checkpoint_dir}")
        sys.exit(1)

    print("=" * 70)
    print("LoRA Extraction from Regression Head Storage")
    print("=" * 70)
    print(f"Checkpoint: {checkpoint_dir}")
    print()

    # Step 1: 加载LoRA模板
    print("Step 1: Loading LoRA template structure...")
    try:
        lora_keys, lora_shapes = load_lora_template(checkpoint_dir)
        total_params = sum(torch.Size(s).numel() for s in lora_shapes.values())
        print(f"  Template: {len(lora_keys)} tensors, {total_params:,} parameters")
    except FileNotFoundError as e:
        print(f"❌ {e}")
        sys.exit(1)

    # Step 2: 从regression_head提取LoRA
    print("\nStep 2: Extracting LoRA from regression_head...")
    try:
        lora_extracted = extract_lora_from_regression_head(checkpoint_dir, lora_keys, lora_shapes)
    except FileNotFoundError as e:
        print(f"❌ {e}")
        sys.exit(1)

    # Step 3: 验证
    print("\nStep 3: Validation")
    is_valid = verify_lora(lora_extracted)

    if not is_valid:
        print("\n⚠️  Warning: Extracted LoRA may not be valid. Continue anyway? (y/n)")
        response = input().strip().lower()
        if response != 'y':
            print("Aborted.")
            sys.exit(0)

    # Step 4: 保存
    print("\nStep 4: Saving extracted LoRA...")

    if args.output:
        output_path = args.output
    else:
        output_path = os.path.join(checkpoint_dir, "adapter_model_extracted.safetensors")

    save_file(lora_extracted, output_path)
    file_size = os.path.getsize(output_path) / (1024**2)
    print(f"  ✅ Saved to: {output_path}")
    print(f"  File size: {file_size:.2f} MB")

    # Step 5: 可选替换
    if args.replace:
        original_path = os.path.join(checkpoint_dir, "adapter_model.safetensors")

        if args.backup:
            backup_path = original_path + ".backup"
            print(f"\n  Backing up original to: {backup_path}")
            os.rename(original_path, backup_path)
        else:
            print(f"\n  Removing original: {original_path}")
            os.remove(original_path)

        print(f"  Renaming extracted to: {original_path}")
        os.rename(output_path, original_path)

        print(f"\n  ✅ Replacement complete!")

    print("\n" + "=" * 70)
    print("Done!")
    print("=" * 70)

    if not args.replace:
        print("\nNext steps:")
        print(f"  1. Verify: python -c \"from safetensors import safe_open; f = safe_open('{output_path}', framework='pt'); print(len(list(f.keys())))\"")
        print(f"  2. Replace original (with backup):")
        print(f"     mv {os.path.join(checkpoint_dir, 'adapter_model.safetensors')} {os.path.join(checkpoint_dir, 'adapter_model.safetensors.backup')}")
        print(f"     mv {output_path} {os.path.join(checkpoint_dir, 'adapter_model.safetensors')}")
        print(f"  3. Or run with --replace --backup flags")


if __name__ == "__main__":
    main()
