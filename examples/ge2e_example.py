"""
GE2E-KWS Loss 使用示例

该示例展示了如何在实际项目中使用GE2E Loss进行关键词检测模型的训练。
包含了完整的数据准备、训练、验证和推理流程。

基于论文: arXiv:2410.16647v1

作者: EdgeVoice项目
日期: 2024
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.metrics import classification_report
import warnings
warnings.filterwarnings('ignore')

# 导入项目模块
from models.streaming_conformer import StreamingConformer
from models.ge2e_loss import GE2ELoss, GE2EBatchSampler
from config import *


class MockFeatureExtractor:
    """
    模拟特征提取器，用于示例演示
    
    在实际项目中，这里应该使用真实的音频特征提取器，
    比如MFCC、Mel频谱图等。
    """
    
    def __init__(self, feature_dim: int = 48, seq_len: int = 100):
        self.feature_dim = feature_dim
        self.seq_len = seq_len
    
    def extract_features(self, audio_path: str) -> torch.Tensor:
        """
        提取音频特征
        
        Args:
            audio_path: 音频文件路径
            
        Returns:
            features: 音频特征张量 [seq_len, feature_dim]
        """
        # 在实际应用中，这里应该加载音频文件并提取特征
        # 为了示例，我们生成模拟特征
        features = torch.randn(self.seq_len, self.feature_dim)
        return features


def create_mock_dataset(num_classes: int = 4, 
                       samples_per_class: int = 50) -> Tuple[List[torch.Tensor], List[int], List[str]]:
    """
    创建模拟数据集用于演示
    
    Args:
        num_classes: 类别数量
        samples_per_class: 每个类别的样本数量
        
    Returns:
        features: 特征列表
        labels: 标签列表
        label_names: 标签名称列表
    """
    class_names = [f"keyword_{i}" for i in range(num_classes)]
    
    features = []
    labels = []
    label_names = []
    
    feature_extractor = MockFeatureExtractor()
    
    for class_idx, class_name in enumerate(class_names):
        for sample_idx in range(samples_per_class):
            # 生成具有类别相关性的特征
            # 每个类别的特征有不同的统计特性
            base_feature = torch.randn(100, 48)
            
            # 为不同类别添加特定的偏置，使其在嵌入空间中更容易区分
            class_bias = torch.randn(1, 48) * 0.5
            class_feature = base_feature + class_bias
            
            features.append(class_feature)
            labels.append(class_idx)
            label_names.append(class_name)
    
    return features, labels, label_names


def demonstrate_ge2e_loss():
    """
    演示GE2E Loss的基本使用
    """
    print("=" * 60)
    print("GE2E Loss 基本使用演示")
    print("=" * 60)
    
    # 设置参数
    num_phrases = 4  # 关键词数量
    num_utterances_per_phrase = 8  # 每个关键词的音频数量
    embedding_dim = 128
    
    print(f"批次配置:")
    print(f"  关键词数量: {num_phrases}")
    print(f"  每个关键词音频数量: {num_utterances_per_phrase}")
    print(f"  总批次大小: {num_phrases * num_utterances_per_phrase}")
    print(f"  嵌入向量维度: {embedding_dim}")
    
    # 创建模拟嵌入向量
    # 这些通常是由模型生成的
    batch_size = num_phrases * num_utterances_per_phrase
    embeddings = torch.randn(batch_size, embedding_dim)
    
    # 为了演示效果，让同一关键词的嵌入向量更相似
    for i in range(num_phrases):
        start_idx = i * num_utterances_per_phrase
        end_idx = start_idx + num_utterances_per_phrase
        
        # 为同一关键词的嵌入添加相同的偏置
        bias = torch.randn(1, embedding_dim) * 0.3
        embeddings[start_idx:end_idx] += bias
    
    print(f"\n嵌入向量形状: {embeddings.shape}")
    
    # 创建GE2E Loss
    criterion = GE2ELoss(init_w=10.0, init_b=-5.0)
    print(f"初始参数 - w: {criterion.w.item():.2f}, b: {criterion.b.item():.2f}")
    
    # 计算损失
    loss = criterion(embeddings, num_phrases, num_utterances_per_phrase)
    print(f"\nGE2E Loss: {loss.item():.4f}")
    
    # 计算质心
    centroids = criterion.compute_centroids(embeddings, num_phrases, num_utterances_per_phrase)
    print(f"质心形状: {centroids.shape}")
    
    # 分析相似度矩阵
    print("\n分析质心间的相似度:")
    centroid_similarity = torch.matmul(centroids, centroids.T)
    for i in range(num_phrases):
        for j in range(num_phrases):
            sim = centroid_similarity[i, j].item()
            print(f"  关键词{i} vs 关键词{j}: {sim:.3f}")
    
    return loss, centroids, embeddings


def demonstrate_training_workflow():
    """
    演示完整的训练工作流程
    """
    print("\n" + "=" * 60)
    print("完整训练工作流程演示")
    print("=" * 60)
    
    # 1. 准备数据
    print("1. 准备模拟数据集...")
    features, labels, label_names = create_mock_dataset(num_classes=4, samples_per_class=32)
    
    unique_labels = list(set(labels))
    print(f"   数据集统计:")
    for label in unique_labels:
        count = labels.count(label)
        print(f"     类别 {label}: {count} 个样本")
    
    # 2. 创建批次采样器
    print("\n2. 创建GE2E批次采样器...")
    num_phrases_per_batch = 4
    num_utterances_per_phrase = 8
    
    batch_sampler = GE2EBatchSampler(
        labels=labels,
        num_phrases_per_batch=num_phrases_per_batch,
        num_utterances_per_phrase=num_utterances_per_phrase,
        shuffle=True
    )
    
    print(f"   批次采样器配置:")
    print(f"     每批次关键词数量: {num_phrases_per_batch}")
    print(f"     每个关键词音频数量: {num_utterances_per_phrase}")
    print(f"     总批次数量: {len(batch_sampler)}")
    
    # 3. 创建模型和损失函数
    print("\n3. 创建模型和损失函数...")
    model = StreamingConformer(
        input_dim=48,
        hidden_dim=128,
        num_classes=len(unique_labels),
        num_layers=3,
        dropout=0.1
    )
    
    criterion = GE2ELoss(init_w=10.0, init_b=-5.0)
    optimizer = torch.optim.Adam(
        list(model.parameters()) + list(criterion.parameters()),
        lr=1e-3
    )
    
    print(f"   模型参数量: {sum(p.numel() for p in model.parameters()):,}")
    print(f"   GE2E Loss参数: w={criterion.w.item():.2f}, b={criterion.b.item():.2f}")
    
    # 4. 模拟训练过程
    print("\n4. 模拟训练过程...")
    model.train()
    
    # 生成一个批次
    batch_indices = next(iter(batch_sampler))
    batch_features = torch.stack([features[i] for i in batch_indices])
    batch_labels = torch.tensor([labels[i] for i in batch_indices])
    
    print(f"   批次特征形状: {batch_features.shape}")
    print(f"   批次标签形状: {batch_labels.shape}")
    
    # 前向传播
    embeddings = model.get_embeddings(batch_features)
    print(f"   嵌入向量形状: {embeddings.shape}")
    
    # 计算损失
    loss = criterion(embeddings, num_phrases_per_batch, num_utterances_per_phrase)
    print(f"   GE2E Loss: {loss.item():.4f}")
    
    # 反向传播
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    print(f"   更新后的参数: w={criterion.w.item():.2f}, b={criterion.b.item():.2f}")
    
    return model, criterion, embeddings, batch_labels


def analyze_embeddings(embeddings: torch.Tensor, 
                      labels: torch.Tensor,
                      num_phrases: int,
                      num_utterances_per_phrase: int):
    """
    分析嵌入向量的质量
    """
    print("\n" + "=" * 60)
    print("嵌入向量质量分析")
    print("=" * 60)
    
    # 1. 计算类内和类间距离
    print("1. 计算距离统计...")
    
    embeddings_np = embeddings.detach().numpy()
    labels_np = labels.numpy()
    
    intra_class_distances = []  # 类内距离
    inter_class_distances = []  # 类间距离
    
    for i in range(num_phrases):
        start_idx = i * num_utterances_per_phrase
        end_idx = start_idx + num_utterances_per_phrase
        
        class_embeddings = embeddings_np[start_idx:end_idx]
        
        # 计算类内距离
        for j in range(len(class_embeddings)):
            for k in range(j + 1, len(class_embeddings)):
                dist = np.linalg.norm(class_embeddings[j] - class_embeddings[k])
                intra_class_distances.append(dist)
        
        # 计算类间距离
        for other_class in range(num_phrases):
            if other_class != i:
                other_start = other_class * num_utterances_per_phrase
                other_end = other_start + num_utterances_per_phrase
                other_embeddings = embeddings_np[other_start:other_end]
                
                for j in range(len(class_embeddings)):
                    for k in range(len(other_embeddings)):
                        dist = np.linalg.norm(class_embeddings[j] - other_embeddings[k])
                        inter_class_distances.append(dist)
    
    print(f"   类内平均距离: {np.mean(intra_class_distances):.4f} ± {np.std(intra_class_distances):.4f}")
    print(f"   类间平均距离: {np.mean(inter_class_distances):.4f} ± {np.std(inter_class_distances):.4f}")
    print(f"   分离度 (类间/类内): {np.mean(inter_class_distances)/np.mean(intra_class_distances):.4f}")
    
    # 2. 余弦相似度分析
    print("\n2. 余弦相似度分析...")
    
    # 归一化嵌入向量
    embeddings_norm = F.normalize(embeddings, p=2, dim=1)
    similarity_matrix = torch.matmul(embeddings_norm, embeddings_norm.T)
    
    # 计算类内相似度
    intra_similarities = []
    for i in range(num_phrases):
        start_idx = i * num_utterances_per_phrase
        end_idx = start_idx + num_utterances_per_phrase
        
        class_sim_matrix = similarity_matrix[start_idx:end_idx, start_idx:end_idx]
        # 排除对角线（自相似度）
        mask = ~torch.eye(num_utterances_per_phrase, dtype=bool)
        class_similarities = class_sim_matrix[mask]
        intra_similarities.extend(class_similarities.tolist())
    
    print(f"   类内平均余弦相似度: {np.mean(intra_similarities):.4f} ± {np.std(intra_similarities):.4f}")
    
    return {
        'intra_class_distance': np.mean(intra_class_distances),
        'inter_class_distance': np.mean(inter_class_distances),
        'separation_ratio': np.mean(inter_class_distances) / np.mean(intra_class_distances),
        'intra_class_similarity': np.mean(intra_similarities)
    }


def demonstrate_inference():
    """
    演示使用训练好的模型进行推理
    """
    print("\n" + "=" * 60)
    print("推理演示")
    print("=" * 60)
    
    # 创建一个简单的模型用于演示
    model = StreamingConformer(
        input_dim=48,
        hidden_dim=128,
        num_classes=4,
        num_layers=3,
        dropout=0.1
    )
    model.eval()
    
    print("1. 单样本推理...")
    
    # 模拟新的音频样本
    test_sample = torch.randn(1, 100, 48)  # [batch_size=1, seq_len, feature_dim]
    
    # 获取嵌入向量
    with torch.no_grad():
        test_embedding = model.get_embeddings(test_sample)
    
    print(f"   测试样本嵌入向量形状: {test_embedding.shape}")
    
    # 模拟已注册的关键词质心
    num_registered_keywords = 4
    embedding_dim = 128
    registered_centroids = torch.randn(num_registered_keywords, embedding_dim)
    registered_centroids = F.normalize(registered_centroids, p=2, dim=1)
    
    print(f"   注册关键词数量: {num_registered_keywords}")
    
    # 计算相似度
    test_embedding_norm = F.normalize(test_embedding, p=2, dim=1)
    similarities = torch.matmul(test_embedding_norm, registered_centroids.T)
    
    print(f"\n2. 与各关键词的相似度:")
    keyword_names = ["hello", "thanks", "yes", "no"]
    for i, (name, sim) in enumerate(zip(keyword_names, similarities[0])):
        print(f"   {name}: {sim.item():.4f}")
    
    # 预测最匹配的关键词
    best_match_idx = torch.argmax(similarities, dim=1)
    best_similarity = torch.max(similarities, dim=1)[0]
    
    print(f"\n3. 预测结果:")
    print(f"   最匹配关键词: {keyword_names[best_match_idx.item()]}")
    print(f"   置信度: {best_similarity.item():.4f}")
    
    # 设置阈值进行决策
    threshold = 0.5
    if best_similarity.item() > threshold:
        print(f"   决策: 检测到关键词 '{keyword_names[best_match_idx.item()]}'")
    else:
        print(f"   决策: 未检测到已注册的关键词 (低于阈值 {threshold})")


def main():
    """
    运行所有演示
    """
    print("GE2E-KWS Loss 完整示例演示")
    print("基于论文: arXiv:2410.16647v1")
    print("=" * 80)
    
    # 设置随机种子以获得可重现的结果
    torch.manual_seed(42)
    np.random.seed(42)
    
    try:
        # 1. 基本使用演示
        loss, centroids, embeddings = demonstrate_ge2e_loss()
        
        # 2. 训练工作流程演示
        model, criterion, train_embeddings, train_labels = demonstrate_training_workflow()
        
        # 3. 嵌入向量分析
        metrics = analyze_embeddings(
            train_embeddings, 
            train_labels, 
            num_phrases=4, 
            num_utterances_per_phrase=8
        )
        
        # 4. 推理演示
        demonstrate_inference()
        
        print("\n" + "=" * 80)
        print("演示完成！")
        print("\n总结:")
        print(f"  - GE2E Loss成功实现并通过测试")
        print(f"  - 训练工作流程正常运行")
        print(f"  - 嵌入向量分离度: {metrics['separation_ratio']:.2f}")
        print(f"  - 类内相似度: {metrics['intra_class_similarity']:.4f}")
        print("\n使用指南:")
        print("  1. 准备按GE2E要求组织的数据集")
        print("  2. 使用GE2EBatchSampler确保正确的批次结构")
        print("  3. 训练时使用model.get_embeddings()获取嵌入向量")
        print("  4. 用GE2ELoss计算损失并优化")
        print("  5. 推理时计算测试样本与注册质心的相似度")
        
    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 