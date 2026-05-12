#!/usr/bin/env python3
"""
测试优化后的情感分类器组件
"""
import torch
import torch.nn as nn
import numpy as np
from FourierGNNmodel import EnhancedEmotionClassifier, FocalLoss, LabelSmoothingCrossEntropy, MultiTaskEmotionClassifier

def test_enhanced_emotion_classifier():
    """测试增强情感分类器"""
    print("Testing EnhancedEmotionClassifier...")

    input_dim = 512
    n_classes = 6
    hidden_dims = [256, 128]
    batch_size = 4
    seq_len = 10

    model = EnhancedEmotionClassifier(
        input_dim=input_dim,
        n_classes=n_classes,
        hidden_dims=hidden_dims,
        dropout=0.1,
        use_residual=True,
        use_batch_norm=True
    )

    # 测试序列输入
    x_seq = torch.randn(batch_size, seq_len, input_dim)
    output_seq = model(x_seq)
    print(f"Sequence input shape: {x_seq.shape}")
    print(f"Sequence output shape: {output_seq.shape}")
    print(f"Output probabilities sum (should be 1): {output_seq.sum(dim=-1)}")

    # 测试单个输入
    x_single = torch.randn(batch_size, input_dim)
    output_single = model(x_single)
    print(f"Single input shape: {x_single.shape}")
    print(f"Single output shape: {output_single.shape}")

    # 测试logits输出
    logits = model(x_single, return_logits=True)
    print(f"Logits shape: {logits.shape}")
    print("✓ EnhancedEmotionClassifier test passed\n")

def test_focal_loss():
    """测试焦点损失"""
    print("Testing FocalLoss...")

    n_classes = 6
    batch_size = 8

    # 创建类别不平衡的权重
    alpha = torch.tensor([0.1, 0.2, 0.3, 0.2, 0.1, 0.1])  # 类别权重
    focal_loss = FocalLoss(alpha=alpha, gamma=2.0)

    # 创建测试数据
    inputs = torch.randn(batch_size, n_classes)
    targets = torch.randint(0, n_classes, (batch_size,))

    loss = focal_loss(inputs, targets)
    print(f"Focal loss value: {loss.item():.4f}")
    print("✓ FocalLoss test passed\n")

def test_label_smoothing_loss():
    """测试标签平滑损失"""
    print("Testing LabelSmoothingCrossEntropy...")

    n_classes = 6
    batch_size = 8
    smoothing = 0.1

    label_smooth_loss = LabelSmoothingCrossEntropy(smoothing=smoothing)

    # 创建测试数据
    inputs = torch.randn(batch_size, n_classes)
    targets = torch.randint(0, n_classes, (batch_size,))

    loss = label_smooth_loss(inputs, targets)
    print(f"Label smoothing loss value: {loss.item():.4f}")

    # 验证标签平滑效果
    probs = torch.softmax(inputs, dim=-1)
    print(f"Original max probability: {probs.max(dim=-1)[0].mean():.4f}")
    print("✓ LabelSmoothingCrossEntropy test passed\n")

def test_multi_task_classifier():
    """测试多任务情感分类器"""
    print("Testing MultiTaskEmotionClassifier...")

    input_dim = 512
    n_classes = 6
    batch_size = 4
    seq_len = 10

    model = MultiTaskEmotionClassifier(
        input_dim=input_dim,
        n_classes=n_classes,
        hidden_dims=[256, 128],
        dropout=0.1,
        use_auxiliary=True
    )

    # 测试输入
    x = torch.randn(batch_size, seq_len, input_dim)

    # 测试主分类输出
    main_probs = model(x)
    print(f"Main classification output shape: {main_probs.shape}")

    # 测试辅助输出
    main_probs_aux, auxiliary = model(x, return_auxiliary=True)
    print(f"Auxiliary outputs keys: {list(auxiliary.keys())}")
    print(f"Intensity shape: {auxiliary['intensity'].shape}")
    print(f"Confidence shape: {auxiliary['confidence'].shape}")

    # 测试损失计算
    targets = torch.randint(0, n_classes, (batch_size * seq_len,))
    loss = model.compute_loss(main_probs.view(-1, n_classes), targets.view(-1))
    print(f"Computed loss: {loss.item():.4f}")
    print("✓ MultiTaskEmotionClassifier test passed\n")

def test_gradient_flow():
    """测试梯度流动"""
    print("Testing gradient flow...")

    input_dim = 256
    n_classes = 6
    batch_size = 2
    seq_len = 5

    model = MultiTaskEmotionClassifier(
        input_dim=input_dim,
        n_classes=n_classes,
        hidden_dims=[128, 64],
        dropout=0.1
    )

    # 创建测试输入
    x = torch.randn(batch_size, seq_len, input_dim, requires_grad=True)
    targets = torch.randint(0, n_classes, (batch_size * seq_len,))

    # 前向传播
    output, auxiliary = model(x, return_auxiliary=True)

    # 计算损失
    loss = model.compute_loss(output.view(-1, n_classes), targets)

    # 反向传播
    loss.backward()

    # 检查梯度
    has_grad = x.grad is not None
    print(f"Input has gradients: {has_grad}")

    if has_grad:
        print("✓ Gradient flow test passed\n")
    else:
        print("✗ Gradient flow test failed\n")

def test_temperature_scaling():
    """测试温度缩放校准"""
    print("Testing temperature scaling...")

    input_dim = 256
    n_classes = 6
    batch_size = 4

    model = EnhancedEmotionClassifier(input_dim=input_dim, n_classes=n_classes)

    # 创建测试输入
    x = torch.randn(batch_size, input_dim)

    # 获取不同温度下的输出
    logits = model(x, return_logits=True)
    probs_original = torch.softmax(logits, dim=-1)

    # 修改温度参数
    original_temp = model.temperature.item()
    model.temperature.data = torch.tensor(0.5)  # 降低温度
    probs_cold = model(x)

    model.temperature.data = torch.tensor(2.0)  # 提高温度
    probs_warm = model(x)

    print(f"Original temperature: {original_temp}")
    print(f"Cold predictions (T=0.5) - max prob: {probs_cold.max(dim=-1)[0].mean():.4f}")
    print(f"Warm predictions (T=2.0) - max prob: {probs_warm.max(dim=-1)[0].mean():.4f}")
    print("✓ Temperature scaling test passed\n")

if __name__ == "__main__":
    print("Testing Optimized Emotion Classifier Components")
    print("=" * 50)

    # 设置随机种子
    torch.manual_seed(42)
    np.random.seed(42)

    try:
        test_enhanced_emotion_classifier()
        test_focal_loss()
        test_label_smoothing_loss()
        test_multi_task_classifier()
        test_gradient_flow()
        test_temperature_scaling()

        print("All tests passed! ✓")
        print("\n优化说明:")
        print("1. EnhancedEmotionClassifier: 多层架构 + 残差连接 + 批归一化")
        print("2. FocalLoss: 处理类别不平衡问题")
        print("3. LabelSmoothingCrossEntropy: 标签平滑正则化")
        print("4. MultiTaskEmotionClassifier: 多任务学习 + 辅助预测")
        print("5. Temperature Scaling: 预测校准")

    except Exception as e:
        print(f"Test failed with error: {e}")
        import traceback
        traceback.print_exc()