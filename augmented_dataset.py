#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
增强音频意图数据集 - 支持实时数据增强
"""

import os
import random
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import librosa
import warnings
from tqdm import tqdm
from data_utils import AudioIntentDataset, prepare_audio_features, load_audio
from audio_augmentation import AudioAugmenter
from config import *


class AugmentedAudioIntentDataset(AudioIntentDataset):
    """
    支持实时数据增强的音频意图数据集类
    继承自基础AudioIntentDataset类
    """
    
    def __init__(self, annotation_file, data_dir=DATA_DIR, 
                 tokenizer=None, sample_rate=SAMPLE_RATE, max_length=MAX_LENGTH, 
                 augment=True, augment_prob=0.5, use_cache=True, 
                 analyze_audio=False, verbose=True):
        """
        初始化增强音频意图数据集
        
        参数:
            annotation_file: 注释文件路径
            data_dir: 音频文件目录
            tokenizer: 用于文本tokenize的分词器
            sample_rate: 采样率
            max_length: 最大序列长度
            augment: 是否启用数据增强
            augment_prob: 应用增强的概率
            use_cache: 是否缓存音频数据
            analyze_audio: 是否分析音频长度分布
            verbose: 是否显示详细信息
        """
        # 调用父类初始化方法
        super().__init__(annotation_file, data_dir, tokenizer, sample_rate, 
                        max_length, analyze_audio, verbose)
        
        # 增强器参数
        self.augment = augment
        self.augment_prob = augment_prob
        self.use_cache = use_cache
        
        # 初始化音频增强器
        self.augmenter = AudioAugmenter(sample_rate=sample_rate)
        
        # 初始化音频缓存
        self.audio_cache = {}
        
        # 显示数据集信息
        if verbose and augment:
            print(f"已启用实时数据增强，增强概率: {augment_prob}")
    
    def __getitem__(self, idx):
        """
        获取数据集中的一个样本
        
        参数:
            idx: 样本索引
        
        返回:
            样本数据字典
        """
        # 获取样本路径和标签
        row = self.df.iloc[idx]
        file_path = os.path.join(self.data_dir, row['file_path'])
        intent_label = self.intents.index(row['intent'])
        
        # 获取音频数据
        if self.use_cache and file_path in self.audio_cache:
            # 从缓存加载
            audio = self.audio_cache[file_path].copy()
        else:
            # 从文件加载
            try:
                audio, _ = load_audio(file_path, self.sample_rate)
                
                # 添加到缓存
                if self.use_cache:
                    self.audio_cache[file_path] = audio.copy()
            except Exception as e:
                warnings.warn(f"无法加载音频 {file_path}: {str(e)}")
                # 返回一个空音频（1秒静音）作为后备
                audio = np.zeros(self.sample_rate)
        
        # 应用数据增强（根据概率）
        if self.augment and random.random() < self.augment_prob:
            # 随机选择一种增强方法
            aug_method = random.choice(['pitch', 'speed', 'volume', 'noise', 'combo'])
            
            if aug_method == 'pitch':
                # 音高变化
                audio, _ = self.augmenter.pitch_shift(audio)
            elif aug_method == 'speed':
                # 语速变化
                audio, _ = self.augmenter.time_stretch(audio)
            elif aug_method == 'volume':
                # 音量变化
                audio, _ = self.augmenter.adjust_volume(audio)
            elif aug_method == 'noise':
                # 添加噪声
                audio, _ = self.augmenter.add_noise(audio)
            else:  # combo
                # 组合多种增强
                if random.random() > 0.5:
                    audio, _ = self.augmenter.pitch_shift(audio)
                if random.random() > 0.5:
                    audio, _ = self.augmenter.time_stretch(audio)
                if random.random() > 0.5:
                    audio, _ = self.augmenter.adjust_volume(audio)
        
        # 提取特征
        audio_feature = prepare_audio_features(
            audio, self.sample_rate, apply_normalization=True
        )
        
        # 准备返回数据
        sample = {'audio_feature': audio_feature, 'intent_label': intent_label}
        
        # 如果有transcript字段，准备文本数据
        if 'transcript' in row and self.tokenizer is not None:
            transcript = str(row['transcript'])
            encoding = self.tokenizer(
                transcript, 
                return_tensors='pt',
                max_length=self.max_length,
                padding='max_length',
                truncation=True
            )
            
            # 添加到样本中
            sample['input_ids'] = encoding['input_ids'].squeeze(0)
            sample['attention_mask'] = encoding['attention_mask'].squeeze(0)
        
        return sample


def prepare_augmented_dataloader(annotation_file, batch_size=BATCH_SIZE, 
                               tokenizer=None, data_dir=DATA_DIR,
                               augment=True, augment_prob=0.5, use_cache=True, 
                               shuffle=True, num_workers=4):
    """
    创建增强数据集的DataLoader
    
    参数:
        annotation_file: 注释文件路径
        batch_size: 批处理大小
        tokenizer: 分词器
        data_dir: 数据目录
        augment: 是否启用增强
        augment_prob: 增强概率
        use_cache: 是否使用缓存
        shuffle: 是否打乱数据
        num_workers: 数据加载线程数
        
    返回:
        增强数据的DataLoader
    """
    dataset = AugmentedAudioIntentDataset(
        annotation_file=annotation_file,
        data_dir=data_dir,
        tokenizer=tokenizer,
        augment=augment,
        augment_prob=augment_prob,
        use_cache=use_cache,
        verbose=True
    )
    
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return dataloader


if __name__ == "__main__":
    # 测试数据增强功能
    from transformers import DistilBertTokenizer
    import matplotlib.pyplot as plt
    
    # 加载分词器
    try:
        tokenizer = DistilBertTokenizer.from_pretrained(DISTILBERT_MODEL_PATH)
    except:
        tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    
    # 创建增强数据集
    dataset = AugmentedAudioIntentDataset(
        annotation_file="data/annotations.csv",
        tokenizer=tokenizer,
        augment=True,
        augment_prob=1.0,  # 总是增强，用于测试
        analyze_audio=True
    )
    
    # 测试获取样本
    sample_idx = random.randint(0, len(dataset) - 1)
    original_sample = dataset.__getitem__(sample_idx)
    
    # 禁用增强以获取原始样本
    dataset.augment = False
    original_sample = dataset.__getitem__(sample_idx)
    original_audio = original_sample['audio_feature']
    
    # 启用增强以获取增强样本
    dataset.augment = True
    augmented_samples = [dataset.__getitem__(sample_idx) for _ in range(5)]
    
    # 可视化原始和增强样本
    plt.figure(figsize=(12, 8))
    
    # 绘制原始样本
    plt.subplot(6, 1, 1)
    plt.plot(original_audio.numpy())
    plt.title("原始音频")
    plt.grid(True)
    
    # 绘制增强样本
    for i, sample in enumerate(augmented_samples):
        plt.subplot(6, 1, i + 2)
        plt.plot(sample['audio_feature'].numpy())
        plt.title(f"增强样本 #{i+1}")
        plt.grid(True)
    
    plt.tight_layout()
    plt.savefig("augmentation_test.png")
    plt.close()
    
    print(f"共测试了 {len(augmented_samples)} 个增强样本")
    print(f"原始样本信息: 特征形状 {original_audio.shape}, 标签 {original_sample['intent_label']}")
    print(f"可视化结果已保存至 augmentation_test.png")
    
    # 测试DataLoader
    dataloader = prepare_augmented_dataloader(
        annotation_file="data/annotations.csv",
        batch_size=4,
        tokenizer=tokenizer,
        augment=True
    )
    
    # 获取一个批次
    batch = next(iter(dataloader))
    print(f"批次信息: 音频特征形状 {batch['audio_feature'].shape}, 标签 {batch['intent_label']}")
    
    if 'input_ids' in batch:
        print(f"文本特征: input_ids 形状 {batch['input_ids'].shape}") 