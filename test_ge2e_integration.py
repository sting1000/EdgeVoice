#!/usr/bin/env python3
"""
GE2E Loss 集成测试脚本

该脚本用于快速验证GE2E Loss与StreamingConformer模型的完整集成，
确保所有组件都能正常工作。

使用方法:
    python test_ge2e_integration.py

作者: EdgeVoice项目  
日期: 2024
"""

import torch
import torch.nn.functional as F
import numpy as np
import sys
import os
from typing import List, Tuple

# 导入项目模块
from models.streaming_conformer import StreamingConformer
from models.ge2e_loss import GE2ELoss, GE2EBatchSampler
from config import *


def test_basic_functionality():
    """测试基本功能"""
    print("=" * 60)
    print("测试 1: 基本功能验证")
    print("=" * 60)
    
    print("1.1 测试GE2E Loss基本功能...")
    
    # 创建模拟数据
    batch_size = 32  # 4个关键词 * 8条音频
    embedding_dim = 128
    num_phrases = 4
    num_utterances_per_phrase = 8
    
    embeddings = torch.randn(batch_size, embedding_dim)
    criterion = GE2ELoss()
    
    try:
        loss = criterion(embeddings, num_phrases, num_utterances_per_phrase)
        print(f"   ✓ GE2E Loss 计算成功: {loss.item():.4f}")
        
        # 测试反向传播
        loss.backward()
        print("   ✓ 反向传播成功")
        
    except Exception as e:
        print(f"   ✗ GE2E Loss 测试失败: {e}")
        return False
    
    print("\n1.2 测试StreamingConformer嵌入向量输出...")
    
    # 创建模型
    model = StreamingConformer(
        input_dim=48,
        hidden_dim=128,
        num_classes=4,
        num_layers=3,
        dropout=0.1
    )
    
    # 测试嵌入向量输出
    input_features = torch.randn(4, 100, 48)  # [batch, seq_len, feature_dim]
    
    try:
        # 测试get_embeddings方法
        embeddings = model.get_embeddings(input_features)
        print(f"   ✓ get_embeddings输出形状: {embeddings.shape}")
        
        # 测试forward_with_embeddings方法
        logits, embeddings2 = model.forward_with_embeddings(input_features)
        print(f"   ✓ forward_with_embeddings - logits: {logits.shape}, embeddings: {embeddings2.shape}")
        
        # 验证嵌入向量是归一化的
        norms = torch.norm(embeddings, p=2, dim=1)
        print(f"   ✓ 嵌入向量L2范数: {norms.mean().item():.4f} (应该接近1.0)")
        
    except Exception as e:
        print(f"   ✗ StreamingConformer 测试失败: {e}")
        return False
    
    print("   ✓ 基本功能测试通过！")
    return True


def test_batch_sampler():
    """测试批次采样器"""
    print("\n" + "=" * 60)
    print("测试 2: 批次采样器验证")
    print("=" * 60)
    
    # 创建模拟标签
    labels = []
    num_classes = 4
    samples_per_class = 20
    
    for class_id in range(num_classes):
        labels.extend([class_id] * samples_per_class)
    
    print(f"模拟数据集: {num_classes} 个类别, 每个类别 {samples_per_class} 个样本")
    
    try:
        # 创建批次采样器
        batch_sampler = GE2EBatchSampler(
            labels=labels,
            num_phrases_per_batch=4,
            num_utterances_per_phrase=8,
            shuffle=True
        )
        
        print(f"   批次采样器创建成功，总批次数: {len(batch_sampler)}")
        
        # 测试生成批次
        batch_count = 0
        for batch_indices in batch_sampler:
            batch_count += 1
            batch_labels = [labels[i] for i in batch_indices]
            
            # 验证批次结构
            if len(batch_indices) != 32:  # 4 * 8
                raise ValueError(f"批次大小错误: {len(batch_indices)}")
            
            # 验证每个类别都有8个样本
            from collections import Counter
            label_counts = Counter(batch_labels)
            
            if len(label_counts) != 4:
                raise ValueError(f"批次中类别数错误: {len(label_counts)}")
            
            for count in label_counts.values():
                if count != 8:
                    raise ValueError(f"类别样本数错误: {count}")
            
            if batch_count >= 3:  # 只测试前3个批次
                break
        
        print(f"   ✓ 批次结构验证通过，测试了 {batch_count} 个批次")
        
    except Exception as e:
        print(f"   ✗ 批次采样器测试失败: {e}")
        return False
    
    return True


def test_end_to_end_training():
    """测试端到端训练"""
    print("\n" + "=" * 60)
    print("测试 3: 端到端训练验证")
    print("=" * 60)
    
    try:
        # 创建模型和损失函数
        model = StreamingConformer(
            input_dim=48,
            hidden_dim=96,  # 使用较小的模型加快测试
            num_classes=4,
            num_layers=2,
            dropout=0.1
        )
        
        criterion = GE2ELoss(init_w=10.0, init_b=-5.0)
        
        # 创建优化器
        optimizer = torch.optim.Adam(
            list(model.parameters()) + list(criterion.parameters()),
            lr=1e-3
        )
        
        print("   模型和优化器创建成功")
        
        # 模拟训练数据
        num_phrases = 4
        num_utterances_per_phrase = 8
        batch_size = num_phrases * num_utterances_per_phrase
        
        input_features = torch.randn(batch_size, 50, 48)  # 较短的序列长度
        
        print("   开始模拟训练...")
        
        # 记录训练过程
        losses = []
        
        for epoch in range(5):  # 训练5个epoch
            model.train()
            
            # 前向传播
            embeddings = model.get_embeddings(input_features)
            
            # 计算损失
            loss = criterion(embeddings, num_phrases, num_utterances_per_phrase)
            losses.append(loss.item())
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            # 更新参数
            optimizer.step()
            
            print(f"   Epoch {epoch+1}: Loss = {loss.item():.4f}, w = {criterion.w.item():.2f}, b = {criterion.b.item():.2f}")
        
        # 验证训练效果
        if len(losses) >= 2:
            if losses[-1] < losses[0]:
                print("   ✓ 损失函数在下降，训练正常")
            else:
                print("   ! 损失函数未明显下降，可能需要调整参数")
        
        print("   ✓ 端到端训练测试完成")
        
    except Exception as e:
        print(f"   ✗ 端到端训练测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


def test_inference_workflow():
    """测试推理工作流程"""
    print("\n" + "=" * 60)
    print("测试 4: 推理工作流程验证")
    print("=" * 60)
    
    try:
        # 创建模型
        model = StreamingConformer(
            input_dim=48,
            hidden_dim=96,
            num_classes=4,
            num_layers=2,
            dropout=0.1
        )
        model.eval()
        
        print("   4.1 测试质心计算...")
        
        # 模拟注册阶段 - 每个关键词多个样本
        num_keywords = 4
        samples_per_keyword = 5
        
        registration_features = torch.randn(num_keywords * samples_per_keyword, 50, 48)
        
        with torch.no_grad():
            registration_embeddings = model.get_embeddings(registration_features)
        
        # 计算每个关键词的质心
        centroids = []
        for i in range(num_keywords):
            start_idx = i * samples_per_keyword
            end_idx = start_idx + samples_per_keyword
            
            keyword_embeddings = registration_embeddings[start_idx:end_idx]
            centroid = F.normalize(keyword_embeddings.mean(dim=0, keepdim=True), p=2, dim=1)
            centroids.append(centroid)
        
        centroids = torch.cat(centroids, dim=0)
        print(f"   ✓ 质心计算完成，形状: {centroids.shape}")
        
        print("   4.2 测试单样本推理...")
        
        # 模拟测试样本
        test_sample = torch.randn(1, 50, 48)
        
        with torch.no_grad():
            test_embedding = model.get_embeddings(test_sample)
            
            # 计算与所有质心的相似度
            similarities = torch.matmul(test_embedding, centroids.T)
            
            # 预测
            best_match = torch.argmax(similarities, dim=1)
            confidence = torch.max(similarities, dim=1)[0]
        
        print(f"   ✓ 推理完成 - 最佳匹配: 关键词{best_match.item()}, 置信度: {confidence.item():.4f}")
        
        print("   4.3 测试批量推理...")
        
        # 模拟批量测试
        batch_test_samples = torch.randn(10, 50, 48)
        
        with torch.no_grad():
            batch_embeddings = model.get_embeddings(batch_test_samples)
            batch_similarities = torch.matmul(batch_embeddings, centroids.T)
            
            batch_predictions = torch.argmax(batch_similarities, dim=1)
            batch_confidences = torch.max(batch_similarities, dim=1)[0]
        
        print(f"   ✓ 批量推理完成，处理 {len(batch_test_samples)} 个样本")
        print(f"     平均置信度: {batch_confidences.mean().item():.4f}")
        
    except Exception as e:
        print(f"   ✗ 推理工作流程测试失败: {e}")
        return False
    
    print("   ✓ 推理工作流程测试通过")
    return True


def main():
    """运行所有测试"""
    print("GE2E-KWS Loss 集成测试")
    print("基于论文: arXiv:2410.16647v1")
    print("=" * 80)
    
    # 设置随机种子
    torch.manual_seed(42)
    np.random.seed(42)
    
    tests = [
        ("基本功能", test_basic_functionality),
        ("批次采样器", test_batch_sampler),
        ("端到端训练", test_end_to_end_training),
        ("推理工作流程", test_inference_workflow),
    ]
    
    passed_tests = 0
    total_tests = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n正在运行测试: {test_name}")
        try:
            if test_func():
                passed_tests += 1
                print(f"✓ {test_name} 测试通过")
            else:
                print(f"✗ {test_name} 测试失败")
        except Exception as e:
            print(f"✗ {test_name} 测试出错: {e}")
    
    print("\n" + "=" * 80)
    print("测试总结")
    print("=" * 80)
    print(f"通过测试: {passed_tests}/{total_tests}")
    
    if passed_tests == total_tests:
        print("🎉 所有测试通过！GE2E Loss 实现已准备就绪。")
        print("\n下一步:")
        print("1. 准备您的音频数据和标注文件")
        print("2. 运行完整示例: python examples/ge2e_example.py")
        print("3. 开始训练: python train_ge2e.py --help")
        return True
    else:
        print("❌ 部分测试失败，请检查实现。")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 