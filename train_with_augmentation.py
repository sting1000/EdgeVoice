#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
使用数据增强训练语音意图识别模型
"""

import os
import time
import random
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm
from transformers import DistilBertTokenizer

from config import *
from models.fast_classifier import FastIntentClassifier
from models.precise_classifier import PreciseIntentClassifier
from augmented_dataset import prepare_augmented_dataloader


def set_seed(seed):
    """设置随机种子以确保可重现性"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def train_model(data_dir=DATA_DIR, annotation_file=None, model_type='fast', 
               epochs=20, learning_rate=LEARNING_RATE, batch_size=BATCH_SIZE,
               save_dir="saved_models", augment=True, augment_prob=0.5, 
               use_cache=True, seed=42):
    """
    训练模型（支持数据增强）
    
    参数:
        data_dir: 数据目录
        annotation_file: 注释文件路径
        model_type: 模型类型，'fast'或'precise'
        epochs: 训练轮数
        learning_rate: 学习率
        batch_size: 批处理大小
        save_dir: 保存目录
        augment: 是否启用数据增强
        augment_prob: 数据增强概率
        use_cache: 是否使用音频缓存
        seed: 随机种子
    
    返回:
        训练好的模型
    """
    # 设置随机种子
    set_seed(seed)
    
    # 确保保存目录存在
    os.makedirs(save_dir, exist_ok=True)
    
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 加载分词器（如果是精确模型）
    tokenizer = None
    if model_type == 'precise':
        try:
            tokenizer = DistilBertTokenizer.from_pretrained(DISTILBERT_MODEL_PATH)
            print(f"已从本地路径加载分词器: {DISTILBERT_MODEL_PATH}")
        except:
            tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
            print("已从在线资源加载分词器")
    
    # 准备数据加载器
    print(f"{'启用' if augment else '禁用'}数据增强，增强概率: {augment_prob if augment else 0}")
    data_loader = prepare_augmented_dataloader(
        annotation_file=annotation_file,
        data_dir=data_dir,
        batch_size=batch_size,
        tokenizer=tokenizer,
        augment=augment,
        augment_prob=augment_prob,
        use_cache=use_cache,
        shuffle=True
    )
    
    # 获取意图类别
    intents = data_loader.dataset.intents
    num_intents = len(intents)
    print(f"意图类别数: {num_intents}")
    for i, intent in enumerate(intents):
        print(f"  {i}: {intent}")
    
    # 初始化模型
    if model_type == 'fast':
        model = FastIntentClassifier(num_intents=num_intents)
        model_save_path = os.path.join(save_dir, "fast_intent_model.pth")
    else:  # precise
        try:
            model = PreciseIntentClassifier(
                num_intents=num_intents,
                pretrained_path=DISTILBERT_MODEL_PATH
            )
            print(f"已从本地路径初始化精确分类器: {DISTILBERT_MODEL_PATH}")
        except:
            model = PreciseIntentClassifier(num_intents=num_intents)
            print("已从在线资源初始化精确分类器")
        model_save_path = os.path.join(save_dir, "precise_intent_model.pth")
    
    model = model.to(device)
    print(f"模型类型: {model_type}")
    
    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # 记录训练历史
    history = {
        'train_loss': [],
        'train_acc': []
    }
    
    # 训练循环
    print(f"开始训练，总轮数: {epochs}")
    start_time = time.time()
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        # 创建进度条
        progress_bar = tqdm(data_loader, desc=f"Epoch {epoch+1}/{epochs}")
        
        for batch in progress_bar:
            # 准备数据
            if model_type == 'fast':
                inputs = batch['audio_feature'].to(device)
                targets = batch['intent_label'].to(device)
                
                # 前向传播
                outputs = model(inputs)
            else:  # precise
                audio_features = batch['audio_feature'].to(device)
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                targets = batch['intent_label'].to(device)
                
                # 前向传播
                outputs = model(audio_features, input_ids, attention_mask)
            
            # 计算损失
            loss = criterion(outputs, targets)
            
            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # 统计
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
            
            # 更新进度条
            progress_bar.set_postfix({
                'loss': running_loss / (progress_bar.n + 1),
                'acc': 100 * correct / total
            })
        
        # 计算本轮平均损失和准确率
        epoch_loss = running_loss / len(data_loader)
        epoch_acc = 100 * correct / total
        
        # 记录历史
        history['train_loss'].append(epoch_loss)
        history['train_acc'].append(epoch_acc)
        
        # 输出本轮结果
        print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.2f}%")
    
    # 计算总训练时间
    total_time = time.time() - start_time
    print(f"训练完成！总时间: {total_time:.2f}秒")
    
    # 保存模型
    torch.save(model.state_dict(), model_save_path)
    print(f"模型已保存至: {model_save_path}")
    
    # 绘制训练曲线
    plot_training_curves(history, save_dir, model_type, augment)
    
    return model


def plot_training_curves(history, save_dir, model_type, augment=True):
    """绘制训练曲线并保存"""
    # 创建图表
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
    
    # 绘制损失曲线
    ax1.plot(history['train_loss'])
    ax1.set_title('模型损失')
    ax1.set_ylabel('损失')
    ax1.set_xlabel('Epoch')
    ax1.grid(True)
    
    # 绘制准确率曲线
    ax2.plot(history['train_acc'])
    ax2.set_title('模型准确率')
    ax2.set_ylabel('准确率 (%)')
    ax2.set_xlabel('Epoch')
    ax2.grid(True)
    
    plt.tight_layout()
    
    # 保存图表
    aug_str = "with_aug" if augment else "no_aug"
    plt.savefig(os.path.join(save_dir, f"{model_type}_training_curves_{aug_str}.png"))
    plt.close()


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="使用数据增强训练意图识别模型")
    
    parser.add_argument(
        "--annotation_file", 
        type=str, 
        required=True,
        help="训练数据集的注释CSV文件路径"
    )
    
    parser.add_argument(
        "--data_dir", 
        type=str, 
        default=DATA_DIR,
        help="音频文件目录"
    )
    
    parser.add_argument(
        "--model_type", 
        type=str, 
        default="fast",
        choices=["fast", "precise"],
        help="模型类型: 'fast'或'precise'"
    )
    
    parser.add_argument(
        "--epochs", 
        type=int, 
        default=20,
        help="训练轮数"
    )
    
    parser.add_argument(
        "--batch_size", 
        type=int, 
        default=BATCH_SIZE,
        help="批处理大小"
    )
    
    parser.add_argument(
        "--learning_rate", 
        type=float, 
        default=LEARNING_RATE,
        help="学习率"
    )
    
    parser.add_argument(
        "--save_dir", 
        type=str, 
        default="saved_models",
        help="模型保存目录"
    )
    
    parser.add_argument(
        "--seed", 
        type=int, 
        default=42,
        help="随机种子，用于可重复性"
    )
    
    parser.add_argument(
        "--no_augment", 
        action="store_true",
        help="禁用数据增强（默认启用）"
    )
    
    parser.add_argument(
        "--augment_prob", 
        type=float, 
        default=0.5,
        help="数据增强概率，即每个样本被增强的概率"
    )
    
    parser.add_argument(
        "--no_cache", 
        action="store_true",
        help="禁用音频缓存（默认启用）"
    )
    
    return parser.parse_args()


def main():
    """主函数"""
    args = parse_args()
    
    print("=" * 50)
    print("使用数据增强训练意图识别模型")
    print("=" * 50)
    print(f"注释文件: {args.annotation_file}")
    print(f"数据目录: {args.data_dir}")
    print(f"模型类型: {args.model_type}")
    print(f"训练轮数: {args.epochs}")
    print(f"批处理大小: {args.batch_size}")
    print(f"学习率: {args.learning_rate}")
    print(f"模型保存目录: {args.save_dir}")
    print(f"是否数据增强: {not args.no_augment}")
    if not args.no_augment:
        print(f"增强概率: {args.augment_prob}")
    print(f"是否缓存音频: {not args.no_cache}")
    print("=" * 50)
    
    # 训练模型
    train_model(
        data_dir=args.data_dir,
        annotation_file=args.annotation_file,
        model_type=args.model_type,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        save_dir=args.save_dir,
        augment=not args.no_augment,
        augment_prob=args.augment_prob,
        use_cache=not args.no_cache,
        seed=args.seed
    )
    
    print("训练完成!")


if __name__ == "__main__":
    main() 