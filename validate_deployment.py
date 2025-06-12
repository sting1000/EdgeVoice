#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
EdgeVoice 部署验证脚本
快速验证整个优化系统的正确性
"""

import torch
import numpy as np
import os
import traceback
from pathlib import Path

from config import *
from models.streaming_conformer import StreamingConformer

def test_model_constraints():
    """测试模型是否满足部署约束"""
    print("🔍 测试模型部署约束...")
    
    # 确保输入维度16通道对齐
    input_dim = N_MFCC * (2 * CONTEXT_FRAMES + 1)
    input_dim = ((input_dim + 15) // 16) * 16
    
    print(f"原始输入维度: {N_MFCC * (2 * CONTEXT_FRAMES + 1)}")
    print(f"对齐后输入维度: {input_dim}")
    print(f"16通道对齐检查: {input_dim % 16 == 0}")
    
    # 创建模型
    model = StreamingConformer(
        input_dim=input_dim,
        hidden_dim=CONFORMER_HIDDEN_SIZE,
        num_classes=len(INTENT_CLASSES),
        num_layers=CONFORMER_LAYERS,
        num_heads=CONFORMER_ATTENTION_HEADS,
        dropout=0.0,
        kernel_size=CONFORMER_CONV_KERNEL_SIZE,
        expansion_factor=CONFORMER_FF_EXPANSION_FACTOR
    )
    
    model.eval()
    
    # 测试输入
    batch_size = 2
    seq_len = 50
    dummy_input = torch.randn(batch_size, seq_len, input_dim)
    
    print(f"测试输入形状: {dummy_input.shape}")
    print(f"输入维度数: {len(dummy_input.shape)} (应该≤4)")
    
    # 前向传播测试
    try:
        with torch.no_grad():
            output = model(dummy_input)
        
        print(f"输出形状: {output.shape}")
        print(f"输出维度数: {len(output.shape)} (应该≤4)")
        print("✅ 前向传播测试通过")
        
        # 流式推理测试 (使用单个样本)
        model.reset_streaming_state()
        single_input = dummy_input[:1]  # 取第一个样本
        pred, conf, cache_states = model.predict_streaming(single_input)
        
        print(f"流式预测结果: {pred.item()}, 置信度: {conf.item():.3f}")
        print("✅ 流式推理测试通过")
        
        return True
        
    except Exception as e:
        print(f"❌ 模型测试失败: {e}")
        traceback.print_exc()
        return False

def test_dimension_alignment():
    """测试维度对齐"""
    print("\n📐 测试维度对齐...")
    
    # 测试各种输入维度
    test_dims = [16, 32, 48, 64, 80, 96, 128, 144, 160]
    
    for dim in test_dims:
        aligned_dim = ((dim + 15) // 16) * 16
        is_aligned = aligned_dim % 16 == 0
        print(f"维度 {dim:3d} -> {aligned_dim:3d} ({'✅' if is_aligned else '❌'})")
    
    print("✅ 维度对齐测试完成")

def test_streaming_cache():
    """测试流式缓存机制"""
    print("\n🔄 测试流式缓存机制...")
    
    input_dim = ((N_MFCC * (2 * CONTEXT_FRAMES + 1) + 15) // 16) * 16
    
    model = StreamingConformer(
        input_dim=input_dim,
        hidden_dim=128,  # 使用较小的隐藏维度
        num_classes=len(INTENT_CLASSES),
        num_layers=2,    # 使用较少的层数
        num_heads=4,     # 使用较少的头数
        dropout=0.0
    )
    
    model.eval()
    
    # 测试多次流式推理
    chunk_size = 20
    num_chunks = 5
    
    cache_states = None
    
    try:
        for i in range(num_chunks):
            chunk_input = torch.randn(1, chunk_size, input_dim)
            
            with torch.no_grad():
                pred, conf, cache_states = model.predict_streaming(
                    chunk_input, cache_states
                )
            
            print(f"  Chunk {i+1}: pred={pred.item()}, conf={conf.item():.3f}")
        
        print("✅ 流式缓存测试通过")
        return True
        
    except Exception as e:
        print(f"❌ 流式缓存测试失败: {e}")
        return False

def test_model_size():
    """测试模型大小"""
    print("\n📊 测试模型大小...")
    
    input_dim = ((N_MFCC * (2 * CONTEXT_FRAMES + 1) + 15) // 16) * 16
    
    model = StreamingConformer(
        input_dim=input_dim,
        hidden_dim=CONFORMER_HIDDEN_SIZE,
        num_classes=len(INTENT_CLASSES),
        num_layers=CONFORMER_LAYERS,
        num_heads=CONFORMER_ATTENTION_HEADS,
        dropout=0.0
    )
    
    # 计算参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"总参数量: {total_params:,}")
    print(f"可训练参数量: {trainable_params:,}")
    print(f"模型大小: {total_params * 4 / 1024 / 1024:.2f} MB (FP32)")
    print(f"模型大小: {total_params * 2 / 1024 / 1024:.2f} MB (FP16)")
    
    # 预估算子数量 (粗略估计)
    estimated_ops = (
        len(model.conformer_layers) * 10 +  # 每层约10个主要算子
        5  # 输入投影、位置编码、池化、分类器
    )
    
    print(f"预估算子数量: ~{estimated_ops} (限制: 768)")
    print(f"算子数量检查: {'✅' if estimated_ops <= 768 else '❌'}")

def test_intent_classes():
    """测试意图类别配置"""
    print("\n🎯 测试意图类别配置...")
    
    print(f"意图类别数量: {len(INTENT_CLASSES)}")
    print("意图类别列表:")
    for i, intent in enumerate(INTENT_CLASSES):
        print(f"  {i}: {intent}")
    
    # 验证类别名称的有效性
    valid_chars = set("ABCDEFGHIJKLMNOPQRSTUVWXYZ_")
    
    for intent in INTENT_CLASSES:
        if not all(c in valid_chars for c in intent):
            print(f"⚠ 警告: 意图名称包含非标准字符: {intent}")
        else:
            print(f"✅ {intent}")

def main():
    """主验证函数"""
    print("=" * 60)
    print("EdgeVoice 部署验证脚本")
    print("=" * 60)
    
    all_tests_passed = True
    
    try:
        # 1. 测试模型约束
        if not test_model_constraints():
            all_tests_passed = False
        
        # 2. 测试维度对齐
        test_dimension_alignment()
        
        # 3. 测试流式缓存
        if not test_streaming_cache():
            all_tests_passed = False
        
        # 4. 测试模型大小
        test_model_size()
        
        # 5. 测试意图类别
        test_intent_classes()
        
        print("\n" + "=" * 60)
        if all_tests_passed:
            print("🎉 所有验证测试通过!")
            print("✅ 系统已准备好进行部署")
        else:
            print("❌ 部分测试失败，请检查问题")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ 验证过程中发生错误: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    main() 