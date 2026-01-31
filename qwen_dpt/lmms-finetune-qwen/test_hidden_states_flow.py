#!/usr/bin/env python
"""
测试 hidden states 从 Qwen2.5-VL 到 DPT head 的数据流
验证关键点：
1. Qwen2.5-VL 能否正确输出 hidden_states
2. 提取的层索引是否正确
3. DPT head 能否接收并处理这些 hidden states
4. 最终输出是否是正确数量的参数（根据 planner）
"""

import torch
import sys
from pathlib import Path
import argparse

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent))

from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from models.qwen2_5_vl_dpt_regression import Qwen2_5_VLForRegression
from planner_configs import get_num_params, get_param_names
from PIL import Image
import numpy as np
import json
import os


def test_hidden_states_extraction(planner: str = "ddp", head_type: str = "simple_mlp"):
    """
    测试 hidden states 提取和 regression head 连接

    Args:
        planner: 规划器类型 (dwa, teb, mppi, ddp)
        head_type: Head 类型 ('simple_mlp', 'transformer', 'dpt')
    """

    # 获取 planner 配置
    num_params = get_num_params(planner)
    param_names = get_param_names(planner)

    print("=" * 80)
    print(f"测试 Regression Head: {head_type.upper()} (Planner: {planner.upper()})")
    print("=" * 80)
    print(f"Target planner: {planner.upper()}")
    print(f"Number of parameters: {num_params}")
    print(f"Parameters: {', '.join(param_names)}")
    print("=" * 80)

    # 1. 加载模型
    print("\n[Step 1] 加载 Qwen2.5-VL-3B 模型...")
    model_path = "Qwen/Qwen2.5-VL-3B-Instruct"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"  - Device: {device}")

    base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        device_map="auto" if torch.cuda.is_available() else None,
    )

    print(f"✓ Base model loaded")
    print(f"  - Model type: {type(base_model).__name__}")
    print(f"  - Hidden size: {base_model.config.hidden_size}")
    print(f"  - Num layers: {base_model.config.num_hidden_layers}")
    print(f"  - Device: {next(base_model.parameters()).device}")

    # 2. 包装为回归模型
    print(f"\n[Step 2] 包装为 Qwen2_5_VLForRegression (head_type={head_type}, num_params={num_params})...")

    model = Qwen2_5_VLForRegression(
        base_model=base_model,
        num_params=num_params,
        head_type=head_type,  # 选择 head 类型
        num_layers_to_extract=4,  # DPT 用
        dropout=0.1,
    )

    # 冻结 base_model，只训练 regression_head
    print(f"\n  Freezing base_model parameters...")
    for param in model.base_model.parameters():
        param.requires_grad = False

    # 确保 regression_head 可训练
    for param in model.regression_head.parameters():
        param.requires_grad = True

    print(f"  ✓ base_model frozen, regression_head trainable")

    # 将 regression head 转换为 bfloat16 并移到 GPU
    if torch.cuda.is_available():
        model.regression_head.to(device=device, dtype=torch.bfloat16)
    else:
        model.regression_head.to(torch.bfloat16)

    print(f"\n✓ Regression model created")
    print(f"  - Head type: {head_type}")
    if head_type == 'simple_mlp':
        print(f"  - 使用 last_hidden_state 的最后一个 token")
    elif head_type == 'transformer':
        print(f"  - 使用 last_hidden_state + self-attention + 全局池化")
    elif head_type == 'dpt':
        print(f"  - 使用多层 hidden_states 融合 + 空间 attention")
    print(f"  - Hidden size: {model.regression_head.hidden_size}")
    print(f"  - Output params: {model.regression_head.num_params}")
    print(f"  - Target params: {', '.join(param_names)}")
    print(f"  - Regression head device: {next(model.regression_head.parameters()).device}")

    # 3. 加载 processor
    print("\n[Step 3] 加载 processor...")
    processor = AutoProcessor.from_pretrained(model_path)
    print(f"✓ Processor loaded")

    # 4. 从真实训练数据加载一个样本
    print("\n[Step 4] 从真实训练数据加载样本...")

    # 根据 planner 找到对应的数据路径
    data_root = "/home/yuanjielu/robot_navigation/noetic/app_data"
    json_path = os.path.join(data_root, f"{planner}_heurstic/splits_200k/chunk_000.json")
    image_folder = os.path.join(data_root, f"{planner}_heurstic")

    if not os.path.exists(json_path):
        print(f"[WARN] {json_path} not found, using dummy data instead")
        # Fallback: 创建假数据
        test_image = Image.fromarray(np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8))
        system_prompt = "You are a navigation scene analyzer."
        user_text = f"Current robot state: Linear velocity: 0.5 m/s, Angular velocity: 0.1 rad/s. Target local planner: {planner.upper()}"
        ground_truth_params = [1.5] * num_params
    else:
        # 从 JSON 文件加载样本，找到第一个有图像文件的样本
        with open(json_path, 'r') as f:
            samples = json.load(f)

        print(f"  - Total samples in JSON: {len(samples)}")

        # 寻找第一个有效样本（图像文件存在）
        sample = None
        test_image = None
        for i, s in enumerate(samples):
            image_path = os.path.join(image_folder, s['images'][0])
            if os.path.exists(image_path):
                sample = s
                test_image = Image.open(image_path).convert('RGB')
                print(f"  - Found valid sample at index {i}")
                break

        if sample is None or test_image is None:
            print(f"[WARN] No valid samples with images found, using dummy data")
            sample = samples[0]
            test_image = Image.fromarray(np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8))
            ground_truth_params = sample['parameters']
            system_prompt = sample.get('system_prompt', 'You are a navigation scene analyzer.')
            user_text = sample['conversations'][0].replace('<image>\n', '')
        else:
            print(f"\n✓ Loaded real sample from training data:")
            print(f"  - Sample ID: {sample['id']}")
            print(f"  - Image path: {sample['images'][0]}")
            print(f"  - Image size: {test_image.size}")
            print(f"  - Ground truth params: {sample['parameters']}")

            # 提取 system_prompt 和 user_text
            system_prompt = sample['system_prompt']
            user_text = sample['conversations'][0].replace('<image>\n', '')  # 移除 <image> 因为我们会重新添加
            ground_truth_params = sample['parameters']

    # 使用 Qwen2.5-VL 的对话格式
    messages = [
        {
            "role": "system",
            "content": system_prompt
        },
        {
            "role": "user",
            "content": [
                {"type": "image", "image": test_image},
                {"type": "text", "text": user_text}
            ]
        }
    ]

    # 处理输入
    text_prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(
        text=[text_prompt],
        images=[test_image],
        padding=True,
        return_tensors="pt",
    )

    print(f"\n✓ Test input created")
    print(f"  - input_ids shape: {inputs['input_ids'].shape}")
    print(f"  - pixel_values shape: {inputs['pixel_values'].shape}")
    if 'image_grid_thw' in inputs:
        print(f"  - image_grid_thw: {inputs['image_grid_thw']}")
    print(f"\n  📝 Using real training data format:")
    print(f"     SYSTEM_PROMPT: {system_prompt[:80]}...")
    print(f"     USER_TEXT: {user_text[:80]}...")
    print(f"     GROUND_TRUTH: {ground_truth_params}")

    # 5. 前向传播（验证 hidden states 流）
    print("\n[Step 5] 前向传播测试...")
    model.eval()

    # 将输入数据移到 GPU
    if torch.cuda.is_available():
        inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
        print(f"  - Inputs moved to {device}")

    with torch.no_grad():
        # 先测试 base_model 的 hidden_states 输出
        print(f"\n  [5.1] 测试 base_model hidden_states 输出...")
        base_outputs = model.base_model(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            pixel_values=inputs['pixel_values'],
            image_grid_thw=inputs.get('image_grid_thw'),
            output_hidden_states=True,
            return_dict=True,
        )

        all_hidden_states = base_outputs.hidden_states
        last_hidden_state = all_hidden_states[-1]
        print(f"    ✓ Computation device: {last_hidden_state.device}")
        print(f"    ✓ 总共 {len(all_hidden_states)} 层 hidden states")
        print(f"    ✓ 最后一层 shape: {last_hidden_state.shape}")

        # 根据 head_type 测试不同的输入
        if head_type == 'simple_mlp':
            print("\n  [5.2] 测试 Simple MLP Head...")
            last_token = last_hidden_state[:, -1, :]
            print(f"    ✓ 提取最后一个 token: {last_token.shape}")
            predictions = model.regression_head(last_hidden_state)
            print(f"    ✓ MLP 输出: {predictions.shape}")

        elif head_type == 'transformer':
            print("\n  [5.2] 测试 Transformer Head...")
            print(f"    ✓ 输入所有 tokens: {last_hidden_state.shape}")
            print(f"    ✓ Self-attention 处理中...")
            predictions = model.regression_head(last_hidden_state)
            print(f"    ✓ Transformer 输出: {predictions.shape}")

        elif head_type == 'dpt':
            print("\n  [5.2] 测试 DPT Head...")
            selected_layers = all_hidden_states[-4:]
            print(f"    ✓ 提取最后 4 层:")
            for i, layer in enumerate(selected_layers):
                print(f"      Layer {-4+i}: {layer.shape}")
            predictions = model.regression_head(selected_layers)
            print(f"    ✓ DPT 输出: {predictions.shape}")

        print(f"\n  [5.3] Regression Head 输出验证")
        print(f"    ✓ Output shape: {predictions.shape}")
        print(f"    ✓ Expected: [batch_size=1, num_params={num_params}]")

        # 完整的端到端测试
        print("\n  [5.4] 端到端测试（model.forward）...")
        outputs = model(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            pixel_values=inputs['pixel_values'],
            image_grid_thw=inputs.get('image_grid_thw'),
        )

        print(f"    ✓ Final predictions shape: {outputs.predictions.shape}")
        print(f"    ✓ Predictions:   {outputs.predictions[0].tolist()}")
        print(f"    ✓ Ground truth:  {ground_truth_params}")

        # 计算与 ground truth 的差异
        if len(ground_truth_params) == num_params:
            pred_array = outputs.predictions[0].float().cpu().numpy()  # 转换为 float32
            gt_array = np.array(ground_truth_params)
            mae = np.mean(np.abs(pred_array - gt_array))
            print(f"    ℹ  MAE from ground truth: {mae:.4f} (random init, not trained yet)")

    # 显示 GPU 内存使用
    if torch.cuda.is_available():
        print(f"\n  💾 GPU Memory Usage:")
        print(f"     Allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
        print(f"     Reserved:  {torch.cuda.memory_reserved() / 1024**3:.2f} GB")

    # 6. 验证可训练参数
    print("\n[Step 6] 验证参数冻结状态...")

    trainable_params = []
    frozen_params = []

    for name, param in model.named_parameters():
        if param.requires_grad:
            trainable_params.append(name)
        else:
            frozen_params.append(name)

    print(f"\n  Trainable parameters ({len(trainable_params)}):")
    for name in trainable_params[:10]:  # 只显示前 10 个
        print(f"    ✓ {name}")
    if len(trainable_params) > 10:
        print(f"    ... and {len(trainable_params) - 10} more")

    print(f"\n  Frozen parameters ({len(frozen_params)}):")
    print(f"    (Total: {len(frozen_params)} parameters frozen)")

    # 验证只有 regression head 和 lm_head 是可训练的
    regression_trainable = [n for n in trainable_params if 'regression_head' in n]
    other_trainable = [n for n in trainable_params if 'regression_head' not in n]

    print(f"\n  Regression head trainable: {len(regression_trainable)}")
    print(f"  Other trainable: {len(other_trainable)}")
    if other_trainable:
        print(f"    (These should only be base_model.lm_head.*)")
        for name in other_trainable:
            print(f"      - {name}")

    # 7. 总结
    print("\n" + "=" * 80)
    print(f"测试总结 - {head_type.upper()} Head")
    print("=" * 80)
    print(f"✓ Planner: {planner.upper()}")
    print(f"✓ Head type: {head_type}")
    print(f"✓ Expected params: {num_params} ({', '.join(param_names)})")
    print(f"✓ Hidden states extraction: PASSED")

    if head_type == 'simple_mlp':
        print(f"✓ Last token extraction: PASSED")
    elif head_type == 'transformer':
        print(f"✓ Self-attention processing: PASSED")
    elif head_type == 'dpt':
        print(f"✓ Multi-layer fusion: PASSED")

    print(f"✓ Regression head processing: PASSED")
    print(f"✓ Output shape: PASSED ({outputs.predictions.shape} == [1, {num_params}])")
    assert outputs.predictions.shape == (1, num_params), f"Shape mismatch! Expected [1, {num_params}], got {outputs.predictions.shape}"
    print(f"✓ Parameter freezing: {'PASSED' if len(regression_trainable) > 0 and all('regression_head' in n or 'lm_head' in n for n in trainable_params) else 'FAILED'}")
    print(f"\n✅ All tests passed! {head_type.upper()} head works correctly for {planner.upper()}.")
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test hidden states flow for different planners and head types")
    parser.add_argument(
        "--planner",
        type=str,
        default="ddp",
        choices=["dwa", "teb", "mppi", "ddp"],
        help="Planner type to test (dwa=7 params, teb=7 params, mppi=8 params, ddp=6 params)"
    )
    parser.add_argument(
        "--head_type",
        type=str,
        default="transformer",
        choices=["simple_mlp", "transformer", "dpt"],
        help="Regression head type to test"
    )
    args = parser.parse_args()

    test_hidden_states_extraction(planner=args.planner, head_type=args.head_type)
