"""
GE2E-KWS Loss的PyTorch实现

基于论文 arXiv:2410.16647v1 中描述的Generalized End-to-End (GE2E) Loss。
该损失函数专门用于关键词检测任务中的嵌入向量学习。

作者: EdgeVoice项目
日期: 2024
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional


class GE2ELoss(nn.Module):
    """
    GE2E-KWS Loss的PyTorch实现，基于论文 arXiv:2410.16647v1.
    
    该损失函数用于训练关键词检测模型的嵌入向量，通过最大化测试样本与对应关键词质心的
    余弦相似度，同时最小化与其他关键词质心的相似度来进行优化。
    """
    
    def __init__(self, init_w: float = 10.0, init_b: float = -5.0):
        """
        初始化GE2E Loss模块.
        
        Args:
            init_w: 缩放因子w的初始值，用于缩放余弦相似度
            init_b: 偏置b的初始值，用于调整相似度基线
        """
        super(GE2ELoss, self).__init__()
        
        # 初始化可学习的缩放因子w和偏置b
        # 这些参数有助于模型更好地分离嵌入向量
        self.w = nn.Parameter(torch.tensor(init_w, dtype=torch.float32))
        self.b = nn.Parameter(torch.tensor(init_b, dtype=torch.float32))
        
        # 用于数值稳定性的小常数
        self.eps = 1e-8
        
    def forward(self, embeddings: torch.Tensor, 
                num_phrases: int, 
                num_utterances_per_phrase: int) -> torch.Tensor:
        """
        计算GE2E Loss.
        
        Args:
            embeddings: 模型的输出嵌入向量, 形状为 (num_phrases * num_utterances_per_phrase, embedding_dim)
            num_phrases: 批次中的关键词/意图数量 (X)
            num_utterances_per_phrase: 每个关键词的音频数量 (Y)
            
        Returns:
            loss: 该批次的平均损失值 (一个标量)
        """
        # 验证输入形状
        batch_size, embedding_dim = embeddings.shape
        if batch_size != num_phrases * num_utterances_per_phrase:
            raise ValueError(
                f"嵌入向量批次大小 {batch_size} 与 num_phrases*num_utterances "
                f"{num_phrases * num_utterances_per_phrase} 不匹配"
            )
        
        if num_utterances_per_phrase < 2:
            raise ValueError("每个关键词至少需要2条音频才能进行注册/测试分离")
        
        # 1. 重塑嵌入向量以分离关键词和音频
        # 形状: (X, Y, dim)
        embeddings_reshaped = embeddings.view(num_phrases, num_utterances_per_phrase, embedding_dim)
        
        # 2. 分离注册集和测试集
        # 注册集：前半部分用于计算质心
        # 测试集：后半部分用于计算损失
        num_enroll = num_utterances_per_phrase // 2
        num_test = num_utterances_per_phrase - num_enroll
        
        enroll_embeddings = embeddings_reshaped[:, :num_enroll, :]  # (X, Y/2, dim)
        test_embeddings = embeddings_reshaped[:, num_enroll:, :]    # (X, Y-Y/2, dim)
        
        # 3. 计算每个关键词的质心
        # 质心是注册集中所有嵌入向量的算术平均值，然后进行L2归一化
        # 形状: (X, dim)
        centroids = F.normalize(enroll_embeddings.mean(dim=1), p=2, dim=1, eps=self.eps)
        
        # 4. 准备计算相似度矩阵
        # 将测试集展平，方便进行矩阵运算
        # 形状: (X * num_test, dim)
        test_embeddings_flat = test_embeddings.reshape(-1, embedding_dim)
        
        # 对测试嵌入进行L2归一化，用于计算余弦相似度
        test_embeddings_normalized = F.normalize(test_embeddings_flat, p=2, dim=1, eps=self.eps)
        
        # 5. 高效计算相似度矩阵 (所有质心 vs 所有测试嵌入)
        # 质心(X, dim) x 测试嵌入转置(dim, X * num_test) -> 相似度矩阵(X, X * num_test)
        similarity_matrix = torch.matmul(centroids, test_embeddings_normalized.T)
        
        # 6. 应用可学习的缩放和偏置
        # 这有助于模型学习合适的相似度范围
        similarity_matrix = self.w * similarity_matrix + self.b
        
        # 7. 计算每个质心的损失
        total_loss = 0.0
        
        for i in range(num_phrases):
            # 对于第i个质心，其正样本在展平的测试集中的索引范围
            start_idx = i * num_test
            end_idx = start_idx + num_test
            
            # 正样本相似度得分（属于同一关键词的测试样本）
            positive_scores = similarity_matrix[i, start_idx:end_idx]
            
            # 负样本相似度得分（属于其他关键词的测试样本）
            # 通过创建mask来排除正样本
            mask = torch.ones_like(similarity_matrix[i], dtype=torch.bool)
            mask[start_idx:end_idx] = False
            negative_scores = similarity_matrix[i, mask]
            
            # 使用logsumexp计算损失，这在数值上更稳定
            # 损失公式: log(sum(exp(negative))) - log(sum(exp(positive)))
            # 目标是最大化正样本相似度，最小化负样本相似度
            loss_i = torch.logsumexp(negative_scores, dim=0) - torch.logsumexp(positive_scores, dim=0)
            total_loss += loss_i
        
        # 返回批次的平均损失
        return total_loss / num_phrases
    
    def compute_centroids(self, embeddings: torch.Tensor, 
                         num_phrases: int, 
                         num_utterances_per_phrase: int) -> torch.Tensor:
        """
        单独计算质心，用于推理或分析
        
        Args:
            embeddings: 嵌入向量
            num_phrases: 关键词数量
            num_utterances_per_phrase: 每个关键词的音频数量
            
        Returns:
            centroids: 归一化的质心向量，形状为 (num_phrases, embedding_dim)
        """
        batch_size, embedding_dim = embeddings.shape
        if batch_size != num_phrases * num_utterances_per_phrase:
            raise ValueError("嵌入向量批次大小与参数不匹配")
        
        # 重塑并计算质心
        embeddings_reshaped = embeddings.view(num_phrases, num_utterances_per_phrase, embedding_dim)
        centroids = F.normalize(embeddings_reshaped.mean(dim=1), p=2, dim=1, eps=self.eps)
        
        return centroids
    
    def compute_similarity_matrix(self, centroids: torch.Tensor, 
                                test_embeddings: torch.Tensor) -> torch.Tensor:
        """
        计算质心与测试嵌入之间的相似度矩阵
        
        Args:
            centroids: 质心向量，形状为 (num_centroids, embedding_dim)
            test_embeddings: 测试嵌入，形状为 (num_test, embedding_dim)
            
        Returns:
            similarity_matrix: 相似度矩阵，形状为 (num_centroids, num_test)
        """
        # 归一化
        centroids_norm = F.normalize(centroids, p=2, dim=1, eps=self.eps)
        test_norm = F.normalize(test_embeddings, p=2, dim=1, eps=self.eps)
        
        # 计算余弦相似度
        similarity = torch.matmul(centroids_norm, test_norm.T)
        
        # 应用缩放和偏置
        return self.w * similarity + self.b


class GE2EBatchSampler:
    """
    用于GE2E Loss的专用批次采样器
    
    该采样器确保每个批次包含指定数量的关键词，每个关键词包含指定数量的音频样本。
    这是GE2E Loss正确工作的前提条件。
    """
    
    def __init__(self, labels: list, 
                 num_phrases_per_batch: int = 8,
                 num_utterances_per_phrase: int = 10,
                 shuffle: bool = True):
        """
        初始化批次采样器
        
        Args:
            labels: 所有样本的标签列表
            num_phrases_per_batch: 每个批次中的关键词数量
            num_utterances_per_phrase: 每个关键词的音频数量
            shuffle: 是否在每个epoch开始时打乱数据
        """
        self.labels = labels
        self.num_phrases_per_batch = num_phrases_per_batch
        self.num_utterances_per_phrase = num_utterances_per_phrase
        self.shuffle = shuffle
        
        # 按标签组织样本索引
        self.label_to_indices = {}
        for idx, label in enumerate(labels):
            if label not in self.label_to_indices:
                self.label_to_indices[label] = []
            self.label_to_indices[label].append(idx)
        
        # 过滤掉样本数量不足的标签
        self.valid_labels = [
            label for label, indices in self.label_to_indices.items()
            if len(indices) >= num_utterances_per_phrase
        ]
        
        if len(self.valid_labels) < num_phrases_per_batch:
            raise ValueError(
                f"有效标签数量 {len(self.valid_labels)} 少于每批次所需的关键词数量 {num_phrases_per_batch}"
            )
    
    def __iter__(self):
        """生成批次索引"""
        if self.shuffle:
            # 打乱有效标签
            valid_labels = self.valid_labels.copy()
            np.random.shuffle(valid_labels)
        else:
            valid_labels = self.valid_labels
        
        # 生成批次
        for i in range(0, len(valid_labels), self.num_phrases_per_batch):
            # 选择当前批次的标签
            batch_labels = valid_labels[i:i + self.num_phrases_per_batch]
            
            if len(batch_labels) < self.num_phrases_per_batch:
                # 如果最后一个批次标签不足，跳过
                continue
            
            batch_indices = []
            
            for label in batch_labels:
                # 为每个标签随机选择指定数量的样本
                available_indices = self.label_to_indices[label].copy()
                if self.shuffle:
                    np.random.shuffle(available_indices)
                
                # 选择样本
                selected_indices = available_indices[:self.num_utterances_per_phrase]
                batch_indices.extend(selected_indices)
            
            yield batch_indices
    
    def __len__(self):
        """返回批次数量"""
        return len(self.valid_labels) // self.num_phrases_per_batch


def test_ge2e_loss():
    """
    测试GE2E Loss的基本功能
    """
    print("开始测试GE2E Loss...")
    
    # 设置参数
    batch_size = 32  # 4个关键词 * 8条音频
    embedding_dim = 128
    num_phrases = 4
    num_utterances_per_phrase = 8
    
    # 创建模拟嵌入向量
    embeddings = torch.randn(batch_size, embedding_dim)
    
    # 创建损失函数
    criterion = GE2ELoss()
    
    try:
        # 计算损失
        loss = criterion(embeddings, num_phrases, num_utterances_per_phrase)
        print(f"损失值: {loss.item():.4f}")
        
        # 测试反向传播
        loss.backward()
        print("反向传播成功")
        
        # 测试质心计算
        centroids = criterion.compute_centroids(embeddings, num_phrases, num_utterances_per_phrase)
        print(f"质心形状: {centroids.shape}")
        
        print("GE2E Loss测试通过！")
        
    except Exception as e:
        print(f"测试失败: {e}")
        raise


if __name__ == "__main__":
    test_ge2e_loss() 