#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import torch
import numpy as np
from tqdm import tqdm

from streaming_dataset import prepare_streaming_dataloader
from models.fast_classifier import FastIntentClassifier
from config import *

def test_streaming_dataloader():
    """测试流式数据加载器"""
    print("测试流式数据加载器...")
    
    # 测试完整音频模式
    print("1. 完整音频模式:")
    dataloader, labels = prepare_streaming_dataloader(
        annotation_file="data/train_annotations.csv",
        streaming_mode=False,
        cache_dir="tmp/feature_cache",
        batch_size=2
    )
    
    print(f"  加载了 {len(dataloader)} 批次的数据")
    print(f"  类别标签: {labels}")
    
    # 查看第一个批次
    for features, batch_labels in dataloader:
        print(f"  特征形状: {features.shape}")
        print(f"  标签形状: {batch_labels.shape}")
        break
    
    # 测试流式模式
    print("\n2. 流式模式:")
    streaming_dataloader, _ = prepare_streaming_dataloader(
        annotation_file="data/train_annotations.csv",
        streaming_mode=True,
        cache_dir="tmp/feature_cache",
        batch_size=2
    )
    
    print(f"  加载了 {len(streaming_dataloader)} 批次的数据")
    
    # 查看第一个批次
    for features, batch_labels in streaming_dataloader:
        print(f"  特征形状: {features.shape}")
        print(f"  标签形状: {batch_labels.shape}")
        break

def test_streaming_model():
    """测试流式模型前向传播"""
    print("\n测试流式模型前向传播...")
    
    # 创建模型
    input_size = N_MFCC * 3
    model = FastIntentClassifier(input_size=input_size, num_classes=len(INTENT_CLASSES))
    model.eval()
    
    # 模拟输入
    batch_size = 2
    seq_len = 10
    x = torch.randn(batch_size, seq_len, input_size)
    
    # 测试普通前向传播
    outputs = model(x)
    print(f"普通前向传播输出形状: {outputs.shape}")
    
    # 测试流式前向传播
    cached_states = None
    for i in range(3):  # 模拟三个时间步
        # 确保输入是3D: [batch_size, seq_len, input_size]
        chunk = torch.randn(batch_size, 5, input_size)  # 每个chunk 5帧
        outputs, cached_states = model.forward_streaming(chunk, cached_states)
        print(f"流式前向传播步骤 {i+1} 输出形状: {outputs.shape}")
        print(f"缓存状态数量: {len(cached_states)}")

def main():
    """主函数"""
    # 设置随机种子
    torch.manual_seed(42)
    np.random.seed(42)
    
    # 测试数据加载器
    test_streaming_dataloader()
    
    # 测试模型
    test_streaming_model()
    
    print("\n所有测试完成!")

if __name__ == "__main__":
    main() 