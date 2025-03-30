#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import numpy as np
import random
from librosa.effects import time_stretch, pitch_shift
import torch.nn.functional as F
from config import *

def augment_streaming_features(features, phase='train'):
    """针对流式特征的增强策略
    
    Args:
        features: 输入特征 [batch_size, seq_len, feat_dim] 或 numpy数组
        phase: 'train' 或 'test'，仅在训练阶段应用增强
        
    Returns:
        augmented: 增强后的特征，与输入相同形状
    """
    # 仅在训练阶段增强
    if phase != 'train':
        return features
    
    # 转换为torch张量（如果是numpy数组）
    is_numpy = isinstance(features, np.ndarray)
    if is_numpy:
        features = torch.FloatTensor(features)
    
    if features.dim() == 2:
        # 单个特征序列 [seq_len, feat_dim]
        features = features.unsqueeze(0)  # 添加batch维度
        batch_size, seq_len, feat_dim = 1, features.size(1), features.size(2)
        single_sample = True
    else:
        # 批量特征 [batch_size, seq_len, feat_dim]
        batch_size, seq_len, feat_dim = features.shape
        single_sample = False
    
    # 创建增强特征副本
    augmented = features.clone()
    
    # 1. 频谱增强 - 随机掩码特征维度(SpecAugment风格)
    if random.random() < 0.5:
        mask_count = random.randint(1, 5)
        for _ in range(mask_count):
            f_start = random.randint(0, feat_dim - 5)
            f_width = random.randint(1, 3)
            augmented[:, :, f_start:f_start+f_width] = 0
    
    # 2. 时间增强 - 随机掩码时间帧
    if random.random() < 0.4:
        mask_count = random.randint(1, 3)
        for _ in range(mask_count):
            t_start = random.randint(0, seq_len - 10)
            t_width = random.randint(2, 5)
            if t_start + t_width < seq_len:
                augmented[:, t_start:t_start+t_width, :] = 0
    
    # 3. 添加高斯噪声
    if random.random() < 0.5:
        noise_level = random.uniform(0.001, 0.005)
        noise = torch.randn_like(augmented) * noise_level
        augmented = augmented + noise
    
    # 4. 时间扭曲(随机延长或缩短某些帧)
    if random.random() < 0.3 and seq_len > 10:
        # 创建随机的扭曲因子
        warp_factor = random.uniform(0.9, 1.1)
        time_indices = torch.arange(seq_len, dtype=torch.float32)
        warped_indices = time_indices * warp_factor
        warped_indices = torch.clamp(warped_indices, 0, seq_len-1)
        
        # 线性插值
        warped_features = torch.zeros_like(augmented)
        for i in range(seq_len):
            idx = warped_indices[i]
            idx_floor = torch.floor(idx).long()
            idx_ceil = torch.ceil(idx).long()
            
            if idx_floor == idx_ceil:
                warped_features[:, i, :] = augmented[:, idx_floor, :]
            else:
                w_floor = idx_ceil - idx
                w_ceil = idx - idx_floor
                warped_features[:, i, :] = w_floor * augmented[:, idx_floor, :] + w_ceil * augmented[:, idx_ceil, :]
        
        augmented = warped_features
    
    # 5. 特征值抖动 - 随机调整特征值
    if random.random() < 0.4:
        jitter_factor = random.uniform(0.98, 1.02)
        augmented = augmented * jitter_factor
    
    # 还原原始形状
    if single_sample and not is_numpy:
        augmented = augmented.squeeze(0)
    
    # 如果原始输入是numpy数组，转换回numpy
    if is_numpy:
        augmented = augmented.numpy()
        if single_sample:
            augmented = augmented.squeeze(0)
    
    return augmented

def mixup_features(features, labels, alpha=0.2):
    """
    应用MixUp数据增强
    
    Args:
        features: 输入特征 [batch_size, seq_len, feature_dim]
        labels: 标签 [batch_size]
        alpha: Beta分布参数
        
    Returns:
        mixed_features: 混合后的特征
        labels_a: 第一组标签
        labels_b: 第二组标签
        lam: 混合系数
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    
    batch_size = features.size(0)
    
    # 生成随机索引，用于混合
    index = torch.randperm(batch_size).to(features.device)
    
    # 特征混合
    mixed_features = lam * features + (1 - lam) * features[index, :]
    
    # 返回原始标签和混合标签
    return mixed_features, labels, labels[index], lam

def time_mask(features, max_mask_len=10, mask_num=2):
    """时间掩码增强
    
    Args:
        features: 特征张量 [batch_size, seq_len, feat_dim]
        max_mask_len: 最大掩码长度
        mask_num: 掩码数量
        
    Returns:
        masekd_features: 掩码后的特征
    """
    masked_features = features.clone()
    batch_size, seq_len, feat_dim = masked_features.shape
    
    for i in range(batch_size):
        for _ in range(mask_num):
            mask_len = random.randint(1, max_mask_len)
            mask_start = random.randint(0, seq_len - mask_len)
            masked_features[i, mask_start:mask_start+mask_len, :] = 0
    
    return masked_features

def freq_mask(features, max_mask_len=5, mask_num=2):
    """频率掩码增强
    
    Args:
        features: 特征张量 [batch_size, seq_len, feat_dim]
        max_mask_len: 最大掩码长度
        mask_num: 掩码数量
        
    Returns:
        masekd_features: 掩码后的特征
    """
    masked_features = features.clone()
    batch_size, seq_len, feat_dim = masked_features.shape
    
    for i in range(batch_size):
        for _ in range(mask_num):
            mask_len = random.randint(1, max_mask_len)
            mask_start = random.randint(0, feat_dim - mask_len)
            masked_features[i, :, mask_start:mask_start+mask_len] = 0
    
    return masked_features

def add_gaussian_noise(features, mean=0, std=0.005):
    """添加高斯噪声
    
    Args:
        features: 特征张量 [batch_size, seq_len, feat_dim]
        mean: 噪声均值
        std: 噪声标准差
        
    Returns:
        noisy_features: 添加噪声后的特征
    """
    noise = torch.randn_like(features) * std + mean
    return features + noise

def time_warp(features, warp_factor=5):
    """应用时间扭曲增强
    
    Args:
        features: 输入特征 [batch_size, seq_len, feature_dim]
        warp_factor: 扭曲因子
        
    Returns:
        warped_features: 扭曲后的特征
    """
    if isinstance(features, np.ndarray):
        features = torch.from_numpy(features)
    
    device = features.device
    
    if len(features.shape) == 2:  # [seq_len, feature_dim]
        features = features.unsqueeze(0)  # [1, seq_len, feature_dim]
        single_sample = True
    else:
        single_sample = False
    
    batch_size, time_steps, feature_dim = features.shape
    
    warped_features = torch.zeros_like(features)
    
    for i in range(batch_size):
        # 创建随机时间扭曲映射
        time_warp_idx = torch.linspace(0, time_steps - 1, time_steps)
        
        # 添加扭曲
        warp = torch.randn(1) * warp_factor
        warp_idx = torch.arange(time_steps, dtype=torch.float32)
        warp_idx = warp_idx + warp * torch.sin(warp_idx.float() * 2 * np.pi / time_steps)
        warp_idx = torch.clamp(warp_idx, 0, time_steps - 1)
        
        # 对每个时间步进行插值
        for t in range(time_steps):
            idx = warp_idx[t].long()
            next_idx = min(idx + 1, time_steps - 1)
            alpha = warp_idx[t] - idx
            
            warped_features[i, t] = (1 - alpha) * features[i, idx] + alpha * features[i, next_idx]
    
    if single_sample:
        warped_features = warped_features.squeeze(0)
    
    return warped_features.to(device)

def add_feature_jitter(features, jitter_ratio=0.05):
    """添加随机抖动增强
    
    Args:
        features: 输入特征 [seq_len, feature_dim] 或 [batch_size, seq_len, feature_dim]
        jitter_ratio: 抖动强度
        
    Returns:
        jittered_features: 添加抖动后的特征
    """
    if isinstance(features, torch.Tensor):
        device = features.device
        features_np = features.cpu().numpy()
        is_tensor = True
    else:
        features_np = features
        is_tensor = False
    
    # 计算抖动值
    noise = np.random.normal(0, jitter_ratio * np.std(features_np), features_np.shape)
    
    # 添加抖动
    jittered_features = features_np + noise
    
    if is_tensor:
        return torch.tensor(jittered_features, device=device)
    else:
        return jittered_features

def add_feature_mask(features, mask_ratio=0.05, mask_value=0):
    """应用特征掩码增强
    
    Args:
        features: 输入特征 [seq_len, feature_dim] 或 [batch_size, seq_len, feature_dim]
        mask_ratio: 掩码比例
        mask_value: 填充值
    
    Returns:
        masked_features: 掩码后的特征
    """
    if isinstance(features, torch.Tensor):
        device = features.device
        features_np = features.cpu().numpy()
        is_tensor = True
    else:
        features_np = features
        is_tensor = False
    
    # 复制原始特征
    masked_features = features_np.copy()
    
    # 获取形状
    if masked_features.ndim == 3:  # [batch_size, seq_len, feature_dim]
        batch_size, seq_len, feature_dim = masked_features.shape
        
        # 对每个样本应用掩码
        for i in range(batch_size):
            # 时间维度掩码
            time_mask_size = max(1, int(seq_len * mask_ratio))
            time_mask_start = random.randint(0, seq_len - time_mask_size)
            masked_features[i, time_mask_start:time_mask_start+time_mask_size, :] = mask_value
            
            # 特征维度掩码
            feat_mask_size = max(1, int(feature_dim * mask_ratio))
            feat_mask_start = random.randint(0, feature_dim - feat_mask_size)
            masked_features[i, :, feat_mask_start:feat_mask_start+feat_mask_size] = mask_value
            
    else:  # [seq_len, feature_dim]
        seq_len, feature_dim = masked_features.shape
        
        # 时间维度掩码
        time_mask_size = max(1, int(seq_len * mask_ratio))
        time_mask_start = random.randint(0, seq_len - time_mask_size)
        masked_features[time_mask_start:time_mask_start+time_mask_size, :] = mask_value
        
        # 特征维度掩码
        feat_mask_size = max(1, int(feature_dim * mask_ratio))
        feat_mask_start = random.randint(0, feature_dim - feat_mask_size)
        masked_features[:, feat_mask_start:feat_mask_start+feat_mask_size] = mask_value
    
    if is_tensor:
        return torch.tensor(masked_features, device=device)
    else:
        return masked_features

def spec_augment(features, freq_mask_param=5, time_mask_param=10, n_freq_masks=2, n_time_masks=2):
    """SpecAugment风格的数据增强
    
    Args:
        features: 输入特征 [batch_size, seq_len, feature_dim]
        freq_mask_param: 频率掩码最大宽度
        time_mask_param: 时间掩码最大宽度
        n_freq_masks: 频率掩码数量
        n_time_masks: 时间掩码数量
        
    Returns:
        augmented_features: 增强后的特征
    """
    if isinstance(features, np.ndarray):
        features = torch.from_numpy(features)
    
    device = features.device
    augmented_features = features.clone()
    
    if len(augmented_features.shape) == 2:  # [seq_len, feature_dim]
        augmented_features = augmented_features.unsqueeze(0)  # [1, seq_len, feature_dim]
        single_sample = True
    else:
        single_sample = False
    
    batch_size, time_steps, feature_dim = augmented_features.shape
    
    # 对每个批次样本应用增强
    for i in range(batch_size):
        # 应用频率掩码
        for _ in range(n_freq_masks):
            f_mask_width = torch.randint(0, freq_mask_param, (1,)).item()
            f_start = torch.randint(0, feature_dim - f_mask_width, (1,)).item()
            augmented_features[i, :, f_start:f_start + f_mask_width] = 0
        
        # 应用时间掩码
        for _ in range(n_time_masks):
            t_mask_width = torch.randint(0, time_mask_param, (1,)).item()
            t_start = torch.randint(0, time_steps - t_mask_width, (1,)).item()
            augmented_features[i, t_start:t_start + t_mask_width, :] = 0
    
    if single_sample:
        augmented_features = augmented_features.squeeze(0)
    
        # 创建随机扭曲点
        center = random.randint(seq_len // 4, seq_len * 3 // 4)
        warp_size = random.randint(1, max_warp)
        
        # 创建扭曲后的索引
        src_indices = torch.arange(seq_len, dtype=torch.float32)
        
        # 在扭曲点周围应用扭曲
        if random.random() < 0.5:  # 50%几率扩展或压缩
            # 扩展 (center附近的索引拉伸)
            warped_indices = src_indices.clone()
            for j in range(center, seq_len):
                warped_indices[j] = min(j + (j - center) * warp_size / seq_len, seq_len - 1)
        else:
            # 压缩 (center附近的索引压缩)
            warped_indices = src_indices.clone()
            for j in range(center, seq_len):
                warped_indices[j] = j - (j - center) * warp_size / seq_len
        
        # 线性插值应用扭曲
        for j in range(seq_len):
            idx = warped_indices[j]
            idx_floor = int(np.floor(idx))
            idx_ceil = min(int(np.ceil(idx)), seq_len - 1)
            
            if idx_floor == idx_ceil:
                warped_features[i, j, :] = features[i, idx_floor, :]
            else:
                w_floor = idx_ceil - idx
                w_ceil = idx - idx_floor
                warped_features[i, j, :] = w_floor * features[i, idx_floor, :] + w_ceil * features[i, idx_ceil, :]
    
    return warped_features

def apply_augmentations(features, phase='train', augment_prob=0.7):
    """应用组合增强策略
    
    Args:
        features: 特征张量 [batch_size, seq_len, feat_dim]
        phase: 'train' 或 'test'
        augment_prob: 整体应用增强的概率
        
    Returns:
        augmented_features: 增强后的特征
    """
    if phase != 'train' or random.random() > augment_prob:
        return features
    
    # 应用随机增强组合
    if random.random() < 0.5:
        features = add_gaussian_noise(features, std=random.uniform(0.001, 0.005))
    
    if random.random() < 0.3:
        features = time_mask(features, max_mask_len=random.randint(1, 5))
    
    if random.random() < 0.3:
        features = freq_mask(features, max_mask_len=random.randint(1, 3))
    
    if random.random() < 0.2:
        features = time_warp(features, max_warp=random.randint(2, 4))
    
    return features 