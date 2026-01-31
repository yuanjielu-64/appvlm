"""
快速测试脚本：验证模型是否正确构建
"""

import torch
from model import NavigationTransformerIL, SimpleTransformerIL


def test_navigation_transformer():
    """测试完整的 NavigationTransformerIL 模型"""
    print("="*60)
    print("Testing NavigationTransformerIL")
    print("="*60)

    # 创建模型
    model = NavigationTransformerIL(
        num_params=8,
        num_history_frames=2,
        vision_model='vit_base_patch16_224',
        vision_pretrained=False,  # 测试时不下载预训练权重
        d_model=768,
        nhead=8,
        num_transformer_layers=4,
        use_velocity=True
    )

    print(f"Total parameters: {model.get_num_params() / 1e6:.2f}M")
    print(f"Trainable parameters: {model.get_trainable_params() / 1e6:.2f}M")

    # 创建测试输入
    batch_size = 4
    current_image = torch.randn(batch_size, 3, 224, 224)
    history_images = torch.randn(batch_size, 2, 3, 224, 224)
    velocity_state = torch.randn(batch_size, 2)

    print(f"\nInput shapes:")
    print(f"  Current image: {current_image.shape}")
    print(f"  History images: {history_images.shape}")
    print(f"  Velocity state: {velocity_state.shape}")

    # 前向传播
    with torch.no_grad():
        output = model(current_image, history_images, velocity_state)

    print(f"\nOutput shape: {output.shape}")
    print(f"Output sample:\n{output[0]}")

    assert output.shape == (batch_size, 8), f"Expected shape (4, 8), got {output.shape}"

    print("\n✓ NavigationTransformerIL test passed!")


def test_simple_transformer():
    """测试简化版本的模型"""
    print("\n" + "="*60)
    print("Testing SimpleTransformerIL")
    print("="*60)

    model = SimpleTransformerIL(
        num_params=8,
        vision_model='vit_base_patch16_224',
        vision_pretrained=False
    )

    print(f"Total parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

    # 测试输入
    batch_size = 4
    image = torch.randn(batch_size, 3, 224, 224)
    velocity = torch.randn(batch_size, 2)

    with torch.no_grad():
        output = model(image, velocity)

    print(f"Output shape: {output.shape}")
    assert output.shape == (batch_size, 8)

    print("\n✓ SimpleTransformerIL test passed!")


def test_without_history():
    """测试不使用历史帧"""
    print("\n" + "="*60)
    print("Testing without history frames")
    print("="*60)

    model = NavigationTransformerIL(
        num_params=8,
        num_history_frames=0,
        vision_pretrained=False,
        use_velocity=True
    )

    batch_size = 4
    current_image = torch.randn(batch_size, 3, 224, 224)
    velocity_state = torch.randn(batch_size, 2)

    with torch.no_grad():
        # 不传递历史帧
        output = model(current_image, history_images=None, velocity_state=velocity_state)

    print(f"Output shape: {output.shape}")
    assert output.shape == (batch_size, 8)

    print("\n✓ No-history test passed!")


def test_without_velocity():
    """测试不使用速度信息"""
    print("\n" + "="*60)
    print("Testing without velocity state")
    print("="*60)

    model = NavigationTransformerIL(
        num_params=8,
        num_history_frames=2,
        vision_pretrained=False,
        use_velocity=False
    )

    batch_size = 4
    current_image = torch.randn(batch_size, 3, 224, 224)
    history_images = torch.randn(batch_size, 2, 3, 224, 224)

    with torch.no_grad():
        # 不传递速度
        output = model(current_image, history_images, velocity_state=None)

    print(f"Output shape: {output.shape}")
    assert output.shape == (batch_size, 8)

    print("\n✓ No-velocity test passed!")


def test_gradient_flow():
    """测试梯度流"""
    print("\n" + "="*60)
    print("Testing gradient flow")
    print("="*60)

    model = NavigationTransformerIL(
        num_params=8,
        num_history_frames=2,
        vision_pretrained=False,
        use_velocity=True
    )

    # 创建输入
    current_image = torch.randn(2, 3, 224, 224, requires_grad=True)
    history_images = torch.randn(2, 2, 3, 224, 224, requires_grad=True)
    velocity_state = torch.randn(2, 2, requires_grad=True)
    target = torch.randn(2, 8)

    # 前向传播
    output = model(current_image, history_images, velocity_state)

    # 计算损失
    loss = torch.nn.functional.mse_loss(output, target)

    # 反向传播
    loss.backward()

    # 检查梯度
    has_grad = 0
    total_params = 0
    for name, param in model.named_parameters():
        if param.requires_grad:
            total_params += 1
            if param.grad is not None:
                has_grad += 1

    print(f"Parameters with gradient: {has_grad}/{total_params}")
    print(f"Loss: {loss.item():.6f}")

    assert has_grad == total_params, "Some parameters don't have gradients!"

    print("\n✓ Gradient flow test passed!")


if __name__ == "__main__":
    # 运行所有测试
    test_navigation_transformer()
    test_simple_transformer()
    test_without_history()
    test_without_velocity()
    test_gradient_flow()

    print("\n" + "="*60)
    print("All tests passed! ✓")
    print("="*60)
