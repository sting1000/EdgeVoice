#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import torch
import numpy as np
import pandas as pd
import random
import librosa
from torch.utils.data import Dataset, DataLoader
from utils.feature_extraction import extract_features, streaming_feature_extractor, random_crop_audio
from config import *
import pickle

class StreamingAudioDataset(Dataset):
    """
    流式音频数据集，支持两种模式：
    1. 完整音频模式（用于预训练）
    2. 流式模式（用于流式微调）
    """
    
    def __init__(self, annotation_file, data_dir=DATA_DIR, streaming_mode=False,
                 chunk_size=STREAMING_CHUNK_SIZE, step_size=STREAMING_STEP_SIZE,
                 use_random_crop=False, cache_dir=None):
        """
        初始化数据集
        
        Args:
            annotation_file: 标注文件路径
            data_dir: 音频文件目录
            streaming_mode: 是否启用流式模式
            chunk_size: 流式处理的块大小（帧数）
            step_size: 流式处理的步长（帧数）
            use_random_crop: 是否使用随机裁剪增强
            cache_dir: 特征缓存目录，加速训练
        """
        self.data_dir = data_dir
        self.streaming_mode = streaming_mode
        self.chunk_size = chunk_size
        self.step_size = step_size
        self.use_random_crop = use_random_crop
        
        # 加载标注
        self.df = pd.read_csv(annotation_file)
        
        # 从标注中获取类别标签
        self.intent_labels = sorted(self.df['intent'].unique())
        self.label_to_id = {label: i for i, label in enumerate(self.intent_labels)}
        
        # 特征缓存
        self.cache_dir = cache_dir
        self.feature_cache = {}
        if cache_dir is not None:
            os.makedirs(cache_dir, exist_ok=True)
    
    def __len__(self):
        return len(self.df)
    
    def _get_cache_path(self, idx, is_streaming):
        """获取缓存文件路径"""
        if self.cache_dir is None:
            return None
            
        mode = "streaming" if is_streaming else "full"
        return os.path.join(self.cache_dir, f"{idx}_{mode}.pkl")
    
    def _load_cached_features(self, cache_path):
        """从缓存加载特征"""
        if cache_path is None or not os.path.exists(cache_path):
            return None
            
        try:
            with open(cache_path, 'rb') as f:
                return pickle.load(f)
        except:
            # 缓存加载失败，删除损坏的缓存
            if os.path.exists(cache_path):
                os.remove(cache_path)
            return None
    
    def _save_to_cache(self, cache_path, data):
        """保存特征到缓存"""
        if cache_path is None:
            return
            
        try:
            with open(cache_path, 'wb') as f:
                pickle.dump(data, f)
        except:
            # 缓存保存失败，忽略错误继续执行
            pass
    
    def _load_audio(self, idx):
        """加载音频文件"""
        row = self.df.iloc[idx]
        file_path = os.path.join(self.data_dir, row['file_path'])
        
        try:
            audio, sr = librosa.load(file_path, sr=TARGET_SAMPLE_RATE)
            # 随机裁剪增强
            if self.use_random_crop and not self.streaming_mode:
                audio = random_crop_audio(audio, sr)
            return audio, sr
        except Exception as e:
            print(f"加载音频文件失败: {file_path}, 错误: {e}")
            # 返回一个短的空音频作为备选
            return np.zeros(TARGET_SAMPLE_RATE), TARGET_SAMPLE_RATE
    
    def __getitem__(self, idx):
        """获取数据集项"""
        row = self.df.iloc[idx]
        label_id = self.label_to_id[row['intent']]
        
        # 尝试从缓存加载特征
        cache_path = self._get_cache_path(idx, self.streaming_mode)
        cached_data = self._load_cached_features(cache_path)
        
        if cached_data is not None:
            if self.streaming_mode:
                return {
                    'chunk_features': cached_data['chunk_features'],
                    'label': label_id
                }
            else:
                return {
                    'features': cached_data['features'],
                    'label': label_id
                }
        
        # 如果没有缓存，加载音频并提取特征
        audio, sr = self._load_audio(idx)
        
        if self.streaming_mode:
            # 流式模式：提取分块特征
            chunk_features = streaming_feature_extractor(
                audio, sr, self.chunk_size, self.step_size
            )
            
            # 如果没有有效的特征块，创建一个空特征作为备选
            if len(chunk_features) == 0:
                chunk_features = [np.zeros((1, N_MFCC * 3))]
            
            # 缓存特征
            if cache_path is not None:
                self._save_to_cache(cache_path, {
                    'chunk_features': chunk_features
                })
            
            return {
                'chunk_features': chunk_features,
                'label': label_id
            }
        else:
            # 完整音频模式：提取全局特征
            features = extract_features(audio, sr)
            
            # 缓存特征
            if cache_path is not None:
                self._save_to_cache(cache_path, {
                    'features': features
                })
            
            return {
                'features': features,
                'label': label_id
            }

def collate_streaming_batch(batch):
    """
    流式模式的批处理函数
    处理不同长度的特征序列，保持批大小不变
    """
    # 为每个样本收集所有块特征
    all_features = []
    all_labels = []
    
    for item in batch:
        # 如果样本有多个chunk，我们只使用第一个
        # 这样可以保持批大小不变
        if len(item['chunk_features']) > 0:
            all_features.append(item['chunk_features'][0])
            all_labels.append(item['label'])
    
    # 转换为tensor
    features = torch.FloatTensor(np.array(all_features))
    labels = torch.LongTensor(all_labels)
    
    return features, labels

def collate_full_batch(batch):
    """
    完整音频模式的批处理函数
    """
    features = []
    labels = []
    
    max_length = max(item['features'].shape[0] for item in batch)
    
    for item in batch:
        # 填充到最大长度
        curr_feat = item['features']
        curr_len = curr_feat.shape[0]
        
        if curr_len < max_length:
            # 填充零
            padded_feat = np.zeros((max_length, curr_feat.shape[1]))
            padded_feat[:curr_len, :] = curr_feat
            features.append(padded_feat)
        else:
            features.append(curr_feat)
            
        labels.append(item['label'])
    
    # 转换为tensor
    features = torch.FloatTensor(np.array(features))
    labels = torch.LongTensor(labels)
    
    return features, labels

def prepare_streaming_dataloader(annotation_file, data_dir=DATA_DIR, batch_size=32, 
                              streaming_mode=False, use_random_crop=False, 
                              cache_dir=None, shuffle=True, seed=42):
    """
    准备用于流式训练的数据加载器
    
    Args:
        annotation_file: 标注文件路径
        data_dir: 音频文件目录
        batch_size: 批大小
        streaming_mode: 是否启用流式模式
        use_random_crop: 是否使用随机裁剪增强
        cache_dir: 特征缓存目录
        shuffle: 是否打乱数据
        seed: 随机种子
        
    Returns:
        dataloader: 数据加载器
        intent_labels: 意图标签列表
    """
    # 设置随机种子
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # 创建数据集
    dataset = StreamingAudioDataset(
        annotation_file=annotation_file,
        data_dir=data_dir,
        streaming_mode=streaming_mode,
        use_random_crop=use_random_crop,
        cache_dir=cache_dir
    )
    
    # 选择正确的批处理函数
    collate_fn = collate_streaming_batch if streaming_mode else collate_full_batch
    
    # 创建数据加载器
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn,
        num_workers=0,  # 可根据系统调整
        pin_memory=torch.cuda.is_available()
    )
    
    return dataloader, dataset.intent_labels 