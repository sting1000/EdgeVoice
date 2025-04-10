import torch
import numpy as np
import random
from librosa.effects import time_stretch, pitch_shift
import torch.nn.functional as F

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
    """MixUp数据增强
    
    Args:
        features: 特征张量 [batch_size, ...]
        labels: 标签张量 [batch_size]
        alpha: mixup强度参数
        
    Returns:
        mixed_features: 混合特征
        mixed_labels: 混合标签
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    
    batch_size = features.size(0)
    index = torch.randperm(batch_size).to(features.device)
    
    # 混合特征
    mixed_features = lam * features + (1 - lam) * features[index, ...]
    
    # 混合标签 - 返回混合标签和混合权重
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

def time_warp(features, max_warp=5):
    """时间扭曲增强
    
    Args:
        features: 特征张量 [batch_size, seq_len, feat_dim]
        max_warp: 最大扭曲步长
        
    Returns:
        warped_features: 扭曲后的特征
    """
    batch_size, seq_len, feat_dim = features.shape
    if seq_len < 10:  # 序列太短，避免扭曲
        return features
    
    warped_features = torch.zeros_like(features)
    
    for i in range(batch_size):
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