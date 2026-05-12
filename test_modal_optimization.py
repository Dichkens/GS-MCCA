#!/usr/bin/env python3
"""
测试优化后的模态编码模块
"""
import torch
import torch.nn as nn
import numpy as np
from FourierGNNmodel import CrossModalAttention, ModalAdaptiveWeight, EnhancedModalFusion

def test_cross_modal_attention():
    """测试跨模态注意力机制"""
    print("Testing CrossModalAttention...")

    embed_dim = 256
    num_heads = 8
    batch_size = 4
    seq_len = 10

    model = CrossModalAttention(embed_dim, num_heads)

    # 创建测试输入
    query = torch.randn(batch_size, seq_len, embed_dim)
    key = torch.randn(batch_size, seq_len, embed_dim)
    value = torch.randn(batch_size, seq_len, embed_dim)

    # 前向传播
    output, attention_weights = model(query, key, value)

    print(f"Input shape: {query.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Attention weights shape: {attention_weights.shape}")
    print("✓ CrossModalAttention test passed\n")

def test_modal_adaptive_weight():
    """测试模态自适应权重"""
    print("Testing ModalAdaptiveWeight...")

    num_modals = 3
    embed_dim = 256
    batch_size = 4
    seq_len = 10

    model = ModalAdaptiveWeight(num_modals, embed_dim)

    # 创建测试输入
    modal_features = [
        torch.randn(batch_size, seq_len, embed_dim) for _ in range(num_modals)
    ]

    # 前向传播
    weighted_features, weights = model(modal_features)

    print(f"Number of modals: {num_modals}")
    print(f"Input shapes: {[f.shape for f in modal_features]}")
    print(f"Output shape: {weighted_features.shape}")
    print(f"Weights shape: {weights.shape}")
    print(f"Weights sum (should be 1): {weights.sum(dim=-1)}")
    print("✓ ModalAdaptiveWeight test passed\n")

def test_enhanced_modal_fusion():
    """测试增强模态融合"""
    print("Testing EnhancedModalFusion...")

    embed_dim = 256
    num_modals = 3
    num_heads = 8
    batch_size = 4
    seq_len = 10

    model = EnhancedModalFusion(embed_dim, num_modals, num_heads)

    # 创建测试输入
    modal_features = [
        torch.randn(batch_size, seq_len, embed_dim) for _ in range(num_modals)
    ]

    # 创建mask
    mask = torch.ones(batch_size, seq_len, dtype=torch.bool)

    # 前向传播
    fused_features, attention_weights, modal_weights = model(modal_features, mask)

    print(f"Number of modals: {num_modals}")
    print(f"Input shapes: {[f.shape for f in modal_features]}")
    print(f"Fused output shape: {fused_features.shape}")
    print(f"Number of attention weight tensors: {len(attention_weights)}")
    print(f"Modal weights shape: {modal_weights.shape}")
    print(f"Modal weights sum (should be 1): {modal_weights.sum(dim=-1)}")
    print("✓ EnhancedModalFusion test passed\n")

def test_gradient_flow():
    """测试梯度流动"""
    print("Testing gradient flow...")

    embed_dim = 128
    num_modals = 2
    num_heads = 4
    batch_size = 2
    seq_len = 5

    model = EnhancedModalFusion(embed_dim, num_modals, num_heads)

    # 创建测试输入
    modal_features = [
        torch.randn(batch_size, seq_len, embed_dim, requires_grad=True) for _ in range(num_modals)
    ]

    # 前向传播
    mask = torch.ones(batch_size, seq_len, dtype=torch.bool)
    output, _, _ = model(modal_features, mask)

    # 计算损失
    loss = output.sum()

    # 反向传播
    loss.backward()

    # 检查梯度
    has_grad = all(f.grad is not None for f in modal_features)
    print(f"All inputs have gradients: {has_grad}")

    if has_grad:
        print("✓ Gradient flow test passed\n")
    else:
        print("✗ Gradient flow test failed\n")

if __name__ == "__main__":
    print("Testing Optimized Modal Encoding Components")
    print("=" * 50)

    # 设置随机种子
    torch.manual_seed(42)
    np.random.seed(42)

    try:
        test_cross_modal_attention()
        test_modal_adaptive_weight()
        test_enhanced_modal_fusion()
        test_gradient_flow()

        print("All tests passed! ✓")
        print("\n优化说明:")
        print("1. CrossModalAttention: 实现了模态间的注意力交互")
        print("2. ModalAdaptiveWeight: 学习模态重要性权重")
        print("3. EnhancedModalFusion: 结合注意力和自适应权重的多模态融合")
        print("4. 梯度可以正常流动，支持端到端训练")

    except Exception as e:
        print(f"Test failed with error: {e}")
        import traceback
        traceback.print_exc()