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
from utils.feature_extraction import add_feature_jitter, add_feature_mask
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
                 use_random_crop=False, cache_dir=None, use_feature_augmentation=False,
                 jitter_ratio=0.02, mask_ratio=0.05):
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
            use_feature_augmentation: 是否启用特征增强（抖动、掩码）
            jitter_ratio: 特征抖动比例
            mask_ratio: 特征掩码比例
        """
        self.data_dir = data_dir
        self.streaming_mode = streaming_mode
        self.chunk_size = int(chunk_size)
        self.step_size = int(step_size)
        self.use_random_crop = use_random_crop
        self.use_feature_augmentation = use_feature_augmentation
        self.jitter_ratio = jitter_ratio
        self.mask_ratio = mask_ratio
        
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
    
    def _apply_feature_augmentation(self, features, jitter_ratio=None, mask_ratio=None):
        """
        应用特征增强（抖动和掩码）
        
        Args:
            features: 输入特征，可能是单个特征数组或特征数组列表
            jitter_ratio: 抖动比例，如果为None则使用类默认值
            mask_ratio: 掩码比例，如果为None则使用类默认值
            
        Returns:
            增强后的特征
        """
        if not self.use_feature_augmentation:
            return features
            
        # 使用传入的参数或类默认值
        jitter_ratio = jitter_ratio if jitter_ratio is not None else self.jitter_ratio
        mask_ratio = mask_ratio if mask_ratio is not None else self.mask_ratio
            
        # 不同的数据结构需要不同的处理方式
        if isinstance(features, list):
            # 处理chunk_features列表
            augmented_features = []
            for feat in features:
                # 应用抖动
                feat_aug = add_feature_jitter(feat, jitter_ratio)
                # 应用掩码
                feat_aug = add_feature_mask(feat_aug, mask_ratio)
                augmented_features.append(feat_aug)
            return augmented_features
        else:
            # 处理单个特征数组
            features = add_feature_jitter(features, jitter_ratio)
            features = add_feature_mask(features, mask_ratio)
            return features
    
    # 直接读取特征向量txt文件，而不是load wav
    def read_feature_txt(self, txt_path):
        # 假设每行是tab或空格分隔的数字，读成二维数组，返回np.ndarray: (n_frames, feature_dim)
        file_path = os.path.join(self.data_dir, txt_path)
        array = np.loadtxt(file_path, dtype=np.uint16)
        return array.view(np.float16)

    def __getitem__(self, idx):
        """获取数据集项"""
        row = self.df.iloc[idx]
        label_id = self.label_to_id[row['intent']]

        file_path = row['file_path']
        gender = row['gender']
        transcript = row['transcript']
        
        # 判断后缀是.txt，则直接读取特征向量
        if file_path.endswith('.txt'):
            features = self.read_feature_txt(file_path)
        else:
            # 处理.wav文件 - 加载音频并提取特征
            try:
                # 检查缓存
                cache_path = self._get_cache_path(idx, self.streaming_mode)
                cached_result = self._load_cached_features(cache_path)
                
                if cached_result is not None:
                    if self.streaming_mode:
                        chunk_features = cached_result
                        if self.use_feature_augmentation:
                            chunk_features = self._apply_feature_augmentation(chunk_features)
                        return {
                            'chunk_features': chunk_features,
                            'label': label_id,
                            'file_path': file_path,
                            'gender': gender,
                            'transcript': transcript
                        }
                    else:
                        features = cached_result
                        if self.use_feature_augmentation:
                            features = self._apply_feature_augmentation(features)
                        return {
                            'features': features,
                            'label': label_id,
                            'file_path': file_path,
                            'gender': gender,
                            'transcript': transcript
                        }
                
                # 加载音频文件
                audio, sr = self._load_audio(idx)
                
                # 提取特征
                if self.streaming_mode:
                    # 流式模式 - 使用streaming_feature_extractor
                    chunk_features, _ = streaming_feature_extractor(
                        audio, sr, 
                        chunk_size=self.chunk_size, 
                        step_size=self.step_size
                    )
                    
                    # 确保至少有一个特征块
                    if len(chunk_features) == 0:
                        chunk_features = [np.zeros((self.chunk_size, N_MFCC * 3))]
                    
                    # 保存到缓存
                    self._save_to_cache(cache_path, chunk_features)
                    
                    # 应用特征增强
                    if self.use_feature_augmentation:
                        chunk_features = self._apply_feature_augmentation(chunk_features)
                    
                    return {
                        'chunk_features': chunk_features,
                        'label': label_id,
                        'file_path': file_path,
                        'gender': gender,
                        'transcript': transcript
                    }
                else:
                    # 非流式模式 - 使用extract_features
                    features = extract_features(audio, sr)
                    
                    # 确保特征不为空
                    if features is None or features.shape[0] == 0:
                        features = np.zeros((10, N_MFCC * 3))  # 默认10帧
                    
                    # 保存到缓存
                    self._save_to_cache(cache_path, features)
                    
                    # 应用特征增强
                    if self.use_feature_augmentation:
                        features = self._apply_feature_augmentation(features)
                    
                    return {
                        'features': features,
                        'label': label_id,
                        'file_path': file_path,
                        'gender': gender,
                        'transcript': transcript
                    }
                    
            except Exception as e:
                print(f"处理音频文件时出错 {file_path}: {e}")
                # 返回默认特征以避免训练中断
                if self.streaming_mode:
                    default_chunk_features = [np.zeros((self.chunk_size, N_MFCC * 3))]
                    return {
                        'chunk_features': default_chunk_features,
                        'label': label_id,
                        'file_path': file_path,
                        'gender': gender,
                        'transcript': transcript
                    }
                else:
                    default_features = np.zeros((10, N_MFCC * 3))
                    return {
                        'features': default_features,
                        'label': label_id,
                        'file_path': file_path,
                        'gender': gender,
                        'transcript': transcript
                    }

        # 原来的.txt文件处理逻辑
        # 数据增强
        # if self.use_feature_augmentation:
        #     features = self._apply_feature_augmentation(features)
        if self.streaming_mode:
            chunk_features = []
            num_frames = features.shape[0]
            for start in range(0, num_frames, self.step_size):
                end = start + self.chunk_size
                chunk = features[start:end]
                if chunk.shape[0] < self.chunk_size:
                    pad_width = self.chunk_size - chunk.shape[0]
                    chunk = np.pad(chunk, ((0, pad_width), (0, 0)), mode='constant')
                chunk_features.append(chunk)
            if len(chunk_features) == 0:
                chunk_features = [np.zeros((self.chunk_size, features.shape[1]))]
            if self.use_feature_augmentation:
                chunk_features = self._apply_feature_augmentation(chunk_features)
            return {
                'chunk_features': chunk_features,
                'label': label_id,
                'file_path': file_path,
                'gender': gender,
                'transcript': transcript
            }
        else:
            # 非流式直接用全体特征
            if self.use_feature_augmentation:
                features = self._apply_feature_augmentation(features)
            return {
                'features': features,
                'label': label_id,
                'file_path': file_path,
                'gender': gender,
                'transcript': transcript
            }

def collate_streaming_batch(batch):
    """
    流式模式的批处理函数
    处理不同长度的特征序列，保持批大小不变
    优化后的版本，减少内存复制和转换开销
    """
    # 为每个样本收集所有块特征
    all_features = []
    all_labels = []
    
    # 优化：预分配最大可能的空间
    max_items = len(batch)
    all_features = [None] * max_items
    all_labels = [None] * max_items
    
    # 填充实际有效的项目
    valid_count = 0
    for item in batch:
        # 如果样本有多个chunk，我们只使用第一个
        # 这样可以保持批大小不变
        if len(item['chunk_features']) > 0:
            all_features[valid_count] = item['chunk_features'][0]
            all_labels[valid_count] = item['label']
            valid_count += 1
    
    # 如果有无效样本，调整列表大小
    if valid_count < max_items:
        all_features = all_features[:valid_count]
        all_labels = all_labels[:valid_count]
    
    # 如果没有有效样本，返回空批次
    if valid_count == 0:
        return torch.FloatTensor(), torch.LongTensor()
    
    # 批量转换为numpy数组，然后一次性转为tensor，减少多次转换开销
    features = np.stack(all_features)
    features = torch.from_numpy(features).float()
    labels = torch.tensor(all_labels, dtype=torch.long)
    
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
                              cache_dir=None, shuffle=True, seed=42, 
                              use_feature_augmentation=False, jitter_ratio=0.02, mask_ratio=0.05):
    """
    准备用于流式训练的数据加载器
    
    Args:
        annotation_file: 标注文件路径
        data_dir: 音频数据目录
        batch_size: 批处理大小
        streaming_mode: 是否使用流式模式
        use_random_crop: 是否使用随机裁剪
        cache_dir: 特征缓存目录
        shuffle: 是否打乱数据顺序
        seed: 随机种子，用于重现性
        use_feature_augmentation: 是否启用特征增强
        jitter_ratio: 特征抖动比例
        mask_ratio: 特征掩码比例
        
    Returns:
        data_loader: 数据加载器
        intent_labels: 意图标签列表
    """
    # 设置随机种子
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
    
    # 创建数据集
    dataset = StreamingAudioDataset(
        annotation_file=annotation_file,
        data_dir=data_dir,
        streaming_mode=streaming_mode,
        use_random_crop=use_random_crop,
        cache_dir=cache_dir,
        use_feature_augmentation=use_feature_augmentation,
        jitter_ratio=jitter_ratio,
        mask_ratio=mask_ratio
    )
    
    # 选择合适的批处理函数
    collate_fn = collate_streaming_batch if streaming_mode else collate_full_batch
    
    # 创建数据加载器
    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn,
        num_workers=2,
        pin_memory=True,
        drop_last=False
    )
    
    return data_loader, dataset.intent_labels 