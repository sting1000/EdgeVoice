"""
使用GE2E Loss训练StreamingConformer模型的示例脚本

该脚本展示了如何将GE2E Loss集成到现有的训练流程中，
实现基于嵌入向量学习的关键词检测模型训练。

作者: EdgeVoice项目
日期: 2024
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import pandas as pd
from typing import List, Tuple, Dict
import argparse
import logging
from tqdm import tqdm

# 导入项目模块
from models.streaming_conformer import StreamingConformer
from models.ge2e_loss import GE2ELoss, GE2EBatchSampler
from config import *


# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GE2EDataset(Dataset):
    """
    专门用于GE2E训练的数据集类
    
    该数据集需要支持按标签进行采样，以便GE2EBatchSampler能够正确工作。
    """
    
    def __init__(self, data_dir: str, annotation_file: str, feature_extractor=None):
        """
        初始化数据集
        
        Args:
            data_dir: 音频数据目录
            annotation_file: 标注文件路径
            feature_extractor: 特征提取器（可选）
        """
        self.data_dir = data_dir
        self.feature_extractor = feature_extractor
        
        # 读取标注文件
        self.df = pd.read_csv(annotation_file)
        
        # 验证必要的列是否存在
        required_columns = ['file_path', 'intent']
        for col in required_columns:
            if col not in self.df.columns:
                raise ValueError(f"标注文件中缺少必要的列: {col}")
        
        # 创建标签到索引的映射
        self.unique_labels = sorted(self.df['intent'].unique())
        self.label_to_idx = {label: idx for idx, label in enumerate(self.unique_labels)}
        self.idx_to_label = {idx: label for label, idx in self.label_to_idx.items()}
        
        # 统计每个标签的样本数量
        label_counts = self.df['intent'].value_counts()
        logger.info(f"数据集统计:")
        for label, count in label_counts.items():
            logger.info(f"  {label}: {count} 个样本")
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        """
        获取一个样本
        
        Args:
            idx: 样本索引
            
        Returns:
            feature: 音频特征 [seq_len, feature_dim]
            label: 标签索引
            label_name: 标签名称
        """
        row = self.df.iloc[idx]
        file_path = os.path.join(self.data_dir, row['file_path'])
        
        # 加载和处理音频特征
        if self.feature_extractor:
            feature = self.feature_extractor(file_path)
        else:
            # 这里应该实现实际的特征提取逻辑
            # 为了示例，我们生成随机特征
            feature = torch.randn(100, FEATURE_DIM)  # [seq_len, feature_dim]
        
        # 获取标签
        label_name = row['intent']
        label = self.label_to_idx[label_name]
        
        return feature, label, label_name
    
    def get_labels(self) -> List[int]:
        """
        获取所有样本的标签列表，用于GE2EBatchSampler
        
        Returns:
            labels: 标签索引列表
        """
        return [self.label_to_idx[intent] for intent in self.df['intent']]


def collate_fn(batch):
    """
    自定义的批次整理函数
    
    Args:
        batch: 批次数据列表
        
    Returns:
        features: 填充后的特征张量 [batch_size, max_seq_len, feature_dim]
        labels: 标签张量 [batch_size]
        label_names: 标签名称列表
    """
    features, labels, label_names = zip(*batch)
    
    # 获取最大序列长度
    max_seq_len = max(f.size(0) for f in features)
    feature_dim = features[0].size(1)
    batch_size = len(features)
    
    # 创建填充后的张量
    padded_features = torch.zeros(batch_size, max_seq_len, feature_dim)
    
    for i, feature in enumerate(features):
        seq_len = feature.size(0)
        padded_features[i, :seq_len, :] = feature
    
    labels_tensor = torch.tensor(labels, dtype=torch.long)
    
    return padded_features, labels_tensor, list(label_names)


class GE2ETrainer:
    """
    GE2E Loss训练器
    """
    
    def __init__(self, model: StreamingConformer, 
                 criterion: GE2ELoss,
                 optimizer: optim.Optimizer,
                 device: torch.device,
                 num_phrases_per_batch: int = 4,
                 num_utterances_per_phrase: int = 8):
        """
        初始化训练器
        
        Args:
            model: StreamingConformer模型
            criterion: GE2E Loss函数
            optimizer: 优化器
            device: 计算设备
            num_phrases_per_batch: 每批次关键词数量
            num_utterances_per_phrase: 每个关键词的音频数量
        """
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.num_phrases_per_batch = num_phrases_per_batch
        self.num_utterances_per_phrase = num_utterances_per_phrase
        
        # 将模型和损失函数移到指定设备
        self.model.to(device)
        self.criterion.to(device)
    
    def train_epoch(self, dataloader: DataLoader, epoch: int) -> float:
        """
        训练一个epoch
        
        Args:
            dataloader: 数据加载器
            epoch: 当前epoch数
            
        Returns:
            avg_loss: 平均损失
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch}")
        
        for batch_idx, (features, labels, label_names) in enumerate(progress_bar):
            # 将数据移到设备
            features = features.to(self.device)
            labels = labels.to(self.device)
            
            # 清零梯度
            self.optimizer.zero_grad()
            
            # 获取嵌入向量
            embeddings = self.model.get_embeddings(features)
            
            # 计算GE2E Loss
            try:
                loss = self.criterion(
                    embeddings, 
                    self.num_phrases_per_batch, 
                    self.num_utterances_per_phrase
                )
                
                # 反向传播
                loss.backward()
                
                # 梯度裁剪（可选）
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                # 更新参数
                self.optimizer.step()
                
                # 累积损失
                total_loss += loss.item()
                num_batches += 1
                
                # 更新进度条
                progress_bar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'Avg Loss': f'{total_loss/num_batches:.4f}',
                    'w': f'{self.criterion.w.item():.2f}',
                    'b': f'{self.criterion.b.item():.2f}'
                })
                
            except Exception as e:
                logger.error(f"批次 {batch_idx} 训练失败: {e}")
                continue
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        return avg_loss
    
    def validate(self, dataloader: DataLoader) -> Dict[str, float]:
        """
        验证模型
        
        Args:
            dataloader: 验证数据加载器
            
        Returns:
            metrics: 验证指标字典
        """
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for features, labels, label_names in dataloader:
                features = features.to(self.device)
                labels = labels.to(self.device)
                
                try:
                    # 获取嵌入向量
                    embeddings = self.model.get_embeddings(features)
                    
                    # 计算损失
                    loss = self.criterion(
                        embeddings,
                        self.num_phrases_per_batch,
                        self.num_utterances_per_phrase
                    )
                    
                    total_loss += loss.item()
                    num_batches += 1
                    
                except Exception as e:
                    logger.error(f"验证批次失败: {e}")
                    continue
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        
        return {
            'val_loss': avg_loss,
            'w_param': self.criterion.w.item(),
            'b_param': self.criterion.b.item()
        }


def main():
    """
    主训练函数
    """
    parser = argparse.ArgumentParser(description='使用GE2E Loss训练StreamingConformer')
    parser.add_argument('--data_dir', type=str, default='./data', help='数据目录')
    parser.add_argument('--annotation_file', type=str, default='./data/annotations.csv', 
                       help='标注文件路径')
    parser.add_argument('--batch_size', type=int, default=32, help='批次大小(总样本数)')
    parser.add_argument('--num_phrases', type=int, default=4, help='每批次关键词数量')
    parser.add_argument('--num_utterances', type=int, default=8, help='每个关键词的音频数量')
    parser.add_argument('--epochs', type=int, default=100, help='训练轮数')
    parser.add_argument('--lr', type=float, default=1e-4, help='学习率')
    parser.add_argument('--save_dir', type=str, default='./saved_models', help='模型保存目录')
    parser.add_argument('--resume', type=str, default=None, help='恢复训练的检查点路径')
    
    args = parser.parse_args()
    
    # 验证批次参数
    if args.batch_size != args.num_phrases * args.num_utterances:
        logger.warning(
            f"批次大小 {args.batch_size} 与 num_phrases*num_utterances "
            f"{args.num_phrases * args.num_utterances} 不匹配，将自动调整"
        )
        args.batch_size = args.num_phrases * args.num_utterances
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"使用设备: {device}")
    
    # 创建数据集
    logger.info("加载数据集...")
    dataset = GE2EDataset(args.data_dir, args.annotation_file)
    
    # 创建GE2E批次采样器
    batch_sampler = GE2EBatchSampler(
        labels=dataset.get_labels(),
        num_phrases_per_batch=args.num_phrases,
        num_utterances_per_phrase=args.num_utterances,
        shuffle=True
    )
    
    # 创建数据加载器
    dataloader = DataLoader(
        dataset,
        batch_sampler=batch_sampler,
        collate_fn=collate_fn,
        num_workers=4
    )
    
    logger.info(f"数据加载器创建完成，共 {len(dataloader)} 个批次")
    
    # 创建模型
    logger.info("初始化模型...")
    model = StreamingConformer(
        input_dim=FEATURE_DIM,
        hidden_dim=HIDDEN_DIM,
        num_classes=len(dataset.unique_labels),
        num_layers=NUM_LAYERS,
        dropout=DROPOUT
    )
    
    # 创建GE2E Loss
    criterion = GE2ELoss(init_w=10.0, init_b=-5.0)
    
    # 创建优化器
    optimizer = optim.Adam(
        list(model.parameters()) + list(criterion.parameters()),
        lr=args.lr,
        weight_decay=1e-4
    )
    
    # 创建学习率调度器
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)
    
    # 创建训练器
    trainer = GE2ETrainer(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        num_phrases_per_batch=args.num_phrases,
        num_utterances_per_phrase=args.num_utterances
    )
    
    # 恢复训练（如果指定）
    start_epoch = 0
    best_loss = float('inf')
    
    if args.resume and os.path.exists(args.resume):
        logger.info(f"从 {args.resume} 恢复训练...")
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        criterion.load_state_dict(checkpoint['criterion_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        best_loss = checkpoint['best_loss']
        logger.info(f"恢复训练从第 {start_epoch} 个epoch开始")
    
    # 创建保存目录
    os.makedirs(args.save_dir, exist_ok=True)
    
    # 训练循环
    logger.info("开始训练...")
    for epoch in range(start_epoch, args.epochs):
        # 训练一个epoch
        avg_loss = trainer.train_epoch(dataloader, epoch)
        
        # 更新学习率
        scheduler.step()
        
        # 验证（使用同一个数据加载器，实际应用中应该使用单独的验证集）
        val_metrics = trainer.validate(dataloader)
        
        logger.info(
            f"Epoch {epoch}: "
            f"Train Loss: {avg_loss:.4f}, "
            f"Val Loss: {val_metrics['val_loss']:.4f}, "
            f"w: {val_metrics['w_param']:.2f}, "
            f"b: {val_metrics['b_param']:.2f}, "
            f"LR: {scheduler.get_last_lr()[0]:.6f}"
        )
        
        # 保存最佳模型
        if val_metrics['val_loss'] < best_loss:
            best_loss = val_metrics['val_loss']
            best_model_path = os.path.join(args.save_dir, 'best_ge2e_model.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'criterion_state_dict': criterion.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_loss': best_loss,
                'val_metrics': val_metrics
            }, best_model_path)
            logger.info(f"保存最佳模型到 {best_model_path}")
        
        # 定期保存检查点
        if (epoch + 1) % 10 == 0:
            checkpoint_path = os.path.join(args.save_dir, f'checkpoint_epoch_{epoch}.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'criterion_state_dict': criterion.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_loss': best_loss,
                'val_metrics': val_metrics
            }, checkpoint_path)
            logger.info(f"保存检查点到 {checkpoint_path}")
    
    logger.info("训练完成！")


if __name__ == "__main__":
    main() 