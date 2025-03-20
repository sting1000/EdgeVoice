#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import librosa
import random
from config import *

class FeatureExtractor:  
    """
    特征提取器类，提供面向对象的特征提取接口
    """
    def __init__(self, sample_rate=TARGET_SAMPLE_RATE,   
                 n_mfcc=N_MFCC, n_fft=N_FFT,   
                 hop_length=HOP_LENGTH, context_frames=CONTEXT_FRAMES,
                 enhanced_features=False):  
        self.sample_rate = sample_rate  
        self.n_mfcc = n_mfcc  
        self.n_fft = n_fft  
        self.hop_length = hop_length  
        self.context_frames = context_frames  
        self.enhanced_features = enhanced_features
        
    def extract_mfcc(self, audio):  
        """提取MFCC特征"""  
        mfcc = librosa.feature.mfcc(y=audio, sr=self.sample_rate,   
                                   n_mfcc=self.n_mfcc, n_fft=self.n_fft,   
                                   hop_length=self.hop_length)  
        
        # 计算delta和delta-delta特征  
        delta_mfcc = librosa.feature.delta(mfcc)  
        delta2_mfcc = librosa.feature.delta(mfcc, order=2)  
        
        # 合并特征  
        features = np.concatenate([mfcc, delta_mfcc, delta2_mfcc], axis=0)  
        
        # 转置以便每行对应一帧  
        features = features.T  # (n_frames, n_features)  
        
        return features
    
    def extract_enhanced_features(self, audio):
        """提取增强型特征，包括MFCC、梅尔谱和特定声学特征"""
        # 基础特征 - MFCC
        mfcc = librosa.feature.mfcc(y=audio, sr=self.sample_rate,   
                                  n_mfcc=self.n_mfcc, n_fft=self.n_fft,   
                                  hop_length=self.hop_length)
        
        # 计算能量特征
        energy = np.sum(np.abs(librosa.stft(audio, n_fft=self.n_fft, hop_length=self.hop_length))**2, axis=0)
        energy = energy.reshape(1, -1)  # 调整形状以便拼接
        
        # 计算梅尔谱图特征
        mel_spec = librosa.feature.melspectrogram(
            y=audio, 
            sr=self.sample_rate, 
            n_fft=self.n_fft, 
            hop_length=self.hop_length, 
            n_mels=20  # 使用20个梅尔频带
        )
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        
        # 计算声音响度特征
        spectral_contrast = librosa.feature.spectral_contrast(
            y=audio, 
            sr=self.sample_rate, 
            n_fft=self.n_fft, 
            hop_length=self.hop_length
        )
        
        # 计算声谱平坦度
        spectral_flatness = librosa.feature.spectral_flatness(
            y=audio, 
            n_fft=self.n_fft, 
            hop_length=self.hop_length
        )
        
        # 计算声谱质心（区分不同声音音色的重要特征）
        spectral_centroid = librosa.feature.spectral_centroid(
            y=audio, 
            sr=self.sample_rate, 
            n_fft=self.n_fft, 
            hop_length=self.hop_length
        )
        
        # 计算过零率特征（区分浊音和清音）
        zcr = librosa.feature.zero_crossing_rate(
            audio, 
            frame_length=self.n_fft, 
            hop_length=self.hop_length
        )
        
        # 计算delta和delta-delta特征
        delta_mfcc = librosa.feature.delta(mfcc)  
        delta2_mfcc = librosa.feature.delta(mfcc, order=2)
        
        # 合并所有特征
        features = np.concatenate([
            mfcc, 
            delta_mfcc, 
            delta2_mfcc, 
            energy, 
            mel_spec_db[:5],  # 使用前5个梅尔频带特征
            spectral_contrast,
            spectral_flatness,
            spectral_centroid,
            zcr
        ], axis=0)
        
        # 转置以便每行对应一帧
        features = features.T  # (n_frames, n_features)
        
        return features
        
    def add_context(self, features, context_size=None):  
        """添加上下文帧信息，形成最终特征"""  
        if context_size is None:  
            context_size = self.context_frames  
            
        num_frames, feat_dim = features.shape  
        
        # 如果特征帧数太少，使用填充  
        if num_frames < 2 * context_size + 1:  
            pad_size = (2 * context_size + 1) - num_frames  
            features = np.pad(features, ((0, pad_size), (0, 0)), 'constant')  
            num_frames = features.shape[0]  
        
        # 构建上下文特征  
        context_features = []  
        for i in range(context_size, num_frames - context_size):  
            # 提取当前帧的上下文  
            context = []  
            for j in range(i - context_size, i + context_size + 1):  
                context.append(features[j])  
            
            # 合并上下文特征  
            context_feat = np.concatenate(context)  
            context_features.append(context_feat)  
        
        return np.array(context_features)  
        
    def extract_features(self, audio):  
        """提取特征向量，可包含上下文信息"""  
        if self.enhanced_features:
            features = self.extract_enhanced_features(audio)
        else:
            features = self.extract_mfcc(audio)
        
        # 添加上下文信息  
        if self.context_frames > 0:  
            features = self.add_context(features, self.context_frames)  
            
        return features
    
    def get_feature_dim(self):
        """获取特征维度"""
        if self.enhanced_features:
            # 计算增强特征的维度
            # MFCC(16) + Delta(16) + Delta2(16) + Energy(1) + Mel Spectrogram(5) 
            # + Spectral Contrast(7) + Flatness(1) + Centroid(1) + ZCR(1)
            base_dim = N_MFCC * 3 + 1 + 5 + 7 + 1 + 1 + 1  # 总共64维
        else:
            # 基本特征：MFCC + Delta + Delta2
            base_dim = self.n_mfcc * 3  # 通常是16*3=48维
        
        if self.context_frames > 0:
            # 如果有上下文特征，每帧的特征维度乘以上下文窗口大小
            feature_dim = base_dim * (2 * self.context_frames + 1)
        else:
            feature_dim = base_dim
        
        return feature_dim

# 函数式接口，保持与原来utils/feature_extraction.py兼容性
def extract_features(audio, sr, n_mfcc=N_MFCC):
    """
    从音频数据提取MFCC特征及其Delta和Delta-Delta
    
    Args:
        audio: 音频数据numpy数组
        sr: 采样率
        n_mfcc: MFCC系数数量(默认为16，产生48维特征)
        
    Returns:
        features: 包含MFCC、Delta、Delta-Delta的特征矩阵 [n_frames, n_mfcc*3]
    """
    # 确保采样率一致
    if sr != TARGET_SAMPLE_RATE:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=TARGET_SAMPLE_RATE)
    
    # 提取MFCC特征
    mfcc = librosa.feature.mfcc(
        y=audio, 
        sr=TARGET_SAMPLE_RATE,
        n_mfcc=n_mfcc,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH
    )
    
    # 计算一阶差分(Delta)特征
    delta = librosa.feature.delta(mfcc, order=1)
    
    # 计算二阶差分(Delta-Delta)特征
    delta2 = librosa.feature.delta(mfcc, order=2)
    
    # 转置特征矩阵为 [n_frames, n_features]
    mfcc = mfcc.T
    delta = delta.T
    delta2 = delta2.T
    
    # 合并所有特征
    features = np.hstack((mfcc, delta, delta2))
    
    return features

def extract_features_streaming(audio_chunk, sr, n_mfcc=N_MFCC, prev_frames=None):
    """
    流式提取MFCC特征，适用于实时处理
    增强版本，提升健壮性
    
    Args:
        audio_chunk: 当前音频数据块
        sr: 采样率
        n_mfcc: MFCC系数数量
        prev_frames: 前一时刻的特征帧(用于计算Delta)
        
    Returns:
        features: 当前块的特征
        last_frames: 需要保存的最后几帧(用于下一块计算)
    """
    # 确保采样率一致
    if sr != TARGET_SAMPLE_RATE:
        audio_chunk = librosa.resample(audio_chunk, orig_sr=sr, target_sr=TARGET_SAMPLE_RATE)
    
    # 需要2帧上下文来计算delta
    context_size = 2
    
    # 处理空音频输入
    if len(audio_chunk) == 0:
        empty_features = np.zeros((1, n_mfcc * 3))
        return empty_features, np.zeros(HOP_LENGTH * context_size)
    
    # 确保音频长度足够
    min_len = HOP_LENGTH * 3  # 至少需要3帧
    if len(audio_chunk) < min_len:
        # 填充
        audio_chunk = np.pad(audio_chunk, (0, min_len - len(audio_chunk)))
    
    # 如果有前一时刻的帧，拼接以保证Delta计算的连续性
    if prev_frames is not None and len(prev_frames) > 0:
        audio_with_context = np.concatenate([prev_frames, audio_chunk])
        
        # 提取MFCC
        mfcc = librosa.feature.mfcc(
            y=audio_with_context, 
            sr=TARGET_SAMPLE_RATE,
            n_mfcc=n_mfcc,
            n_fft=N_FFT,
            hop_length=HOP_LENGTH
        )
        
        # 确保生成了足够的帧
        if mfcc.shape[1] < 3:
            # 不够帧，退化为没有上下文的处理方式
            mfcc = librosa.feature.mfcc(
                y=audio_chunk, 
                sr=TARGET_SAMPLE_RATE,
                n_mfcc=n_mfcc,
                n_fft=N_FFT,
                hop_length=HOP_LENGTH
            )
            # 判断是否有足够的帧计算delta
            if mfcc.shape[1] >= 3:
                delta = librosa.feature.delta(mfcc, width=3, order=1)
                delta2 = librosa.feature.delta(mfcc, width=3, order=2)
            else:
                # 仍然不够，创建零矩阵
                delta = np.zeros_like(mfcc)
                delta2 = np.zeros_like(mfcc)
                
            mfcc = mfcc.T
            delta = delta.T
            delta2 = delta2.T
        else:
            # 计算差分特征（对于小批量音频使用较小的宽度）
            width = min(3, mfcc.shape[1] - 2)  # 确保宽度不超过特征长度
            delta = librosa.feature.delta(mfcc, width=width, order=1)
            delta2 = librosa.feature.delta(mfcc, width=width, order=2)
            
            # 转置和拼接
            mfcc = mfcc.T
            delta = delta.T
            delta2 = delta2.T
            
            # 只取当前块对应的帧
            # 计算前一块对应的帧数
            prev_frames_count = int(len(prev_frames) / HOP_LENGTH)
            
            # 如果帧数计算不准确，采用安全措施
            if prev_frames_count >= mfcc.shape[0]:
                prev_frames_count = max(0, mfcc.shape[0] - 1)
            
            # 去除上下文帧
            mfcc = mfcc[prev_frames_count:]
            delta = delta[prev_frames_count:]
            delta2 = delta2[prev_frames_count:]
    else:
        # 首次处理，没有上下文帧
        mfcc = librosa.feature.mfcc(
            y=audio_chunk, 
            sr=TARGET_SAMPLE_RATE,
            n_mfcc=n_mfcc,
            n_fft=N_FFT,
            hop_length=HOP_LENGTH
        )
        
        # 对于小批量音频使用较小的宽度
        width = min(3, mfcc.shape[1] - 2)  # 确保宽度不超过特征长度
        if width > 0:
            delta = librosa.feature.delta(mfcc, width=width, order=1)
            delta2 = librosa.feature.delta(mfcc, width=width, order=2)
        else:
            # 如果帧数太少，无法计算delta，则使用零矩阵
            delta = np.zeros_like(mfcc)
            delta2 = np.zeros_like(mfcc)
        
        mfcc = mfcc.T
        delta = delta.T
        delta2 = delta2.T
    
    # 合并特征
    features = np.hstack((mfcc, delta, delta2))
    
    # 如果特征为空，创建一个默认特征
    if features.shape[0] == 0:
        features = np.zeros((1, n_mfcc * 3))
    
    # 保存最后context_size帧的音频数据，用于下一块处理
    context_samples = int(context_size * HOP_LENGTH)
    last_frames = audio_chunk[-context_samples:] if len(audio_chunk) >= context_samples else audio_chunk
    
    return features, last_frames

def standardize_features(features):
    """
    标准化特征，使其均值为0，方差为1
    
    Args:
        features: 特征矩阵 [n_frames, n_features]
        
    Returns:
        standardized_features: 标准化后的特征
    """
    # 沿时间轴计算均值和标准差
    mean = np.mean(features, axis=0)
    std = np.std(features, axis=0)
    
    # 避免除以0
    std = np.where(std == 0, 1e-10, std)
    
    # 标准化
    standardized_features = (features - mean) / std
    
    return standardized_features 

def random_crop_audio(audio, sr, min_duration=0.5, max_duration=3.0):
    """
    随机裁剪音频片段，增强模型对不同长度输入的鲁棒性
    
    Args:
        audio: 音频数据numpy数组
        sr: 采样率
        min_duration: 最小裁剪时长(秒)
        max_duration: 最大裁剪时长(秒)
        
    Returns:
        cropped_audio: 裁剪后的音频
    """
    # 原始音频长度(秒)
    original_duration = len(audio) / sr
    
    # 确保最大裁剪长度不超过原始音频长度
    max_duration = min(max_duration, original_duration)
    
    # 如果原始音频已经很短，直接返回
    if original_duration <= min_duration:
        return audio
    
    # 随机选择裁剪长度
    crop_duration = random.uniform(min_duration, max_duration)
    
    # 计算裁剪样本数
    crop_samples = int(crop_duration * sr)
    
    # 随机选择裁剪起始点
    max_start_idx = len(audio) - crop_samples
    start_idx = random.randint(0, max_start_idx)
    
    # 裁剪音频
    cropped_audio = audio[start_idx:start_idx + crop_samples]
    
    return cropped_audio

def streaming_feature_extractor(audio, sr, chunk_size=STREAMING_CHUNK_SIZE, step_size=STREAMING_STEP_SIZE):
    """
    模拟流式处理，逐块提取特征
    
    Args:
        audio: 音频数据numpy数组
        sr: 采样率
        chunk_size: 每个块的帧数
        step_size: 处理步长(帧数)
        
    Returns:
        chunk_features: 列表，包含每个块的特征
    """
    # 确保采样率一致
    if sr != TARGET_SAMPLE_RATE:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=TARGET_SAMPLE_RATE)
    
    # 计算块大小和步长(采样点数)
    chunk_samples = int(chunk_size * HOP_LENGTH)
    step_samples = int(step_size * HOP_LENGTH)
    
    # 存储特征和最后的上下文帧
    chunk_features = []
    prev_frames = None
    
    # 确保音频长度足够
    min_samples = HOP_LENGTH * 3  # 至少需要3帧才能计算delta
    if len(audio) < min_samples:
        # 如果音频太短，填充到最小长度
        audio = np.pad(audio, (0, min_samples - len(audio)))
    
    # 逐块处理
    for i in range(0, len(audio), step_samples):
        # 获取当前音频块
        end_idx = min(i + chunk_samples, len(audio))
        audio_chunk = audio[i:end_idx]
        
        # 如果剩余音频过短，填充到最小长度
        if len(audio_chunk) < HOP_LENGTH * 3:
            if i + chunk_samples > len(audio):
                # 只有最后一块需要填充
                audio_chunk = np.pad(audio_chunk, (0, HOP_LENGTH * 3 - len(audio_chunk)))
            else:
                # 跳过非最后一块但长度不足的情况
                continue
        
        # 提取流式特征 - 使用安全版本
        features, prev_frames = safe_extract_features_streaming(audio_chunk, sr, prev_frames=prev_frames)
        
        # 添加到结果列表
        if len(features) > 0:  # 确保提取到了有效特征
            chunk_features.append(features)
    
    # 确保至少有一个特征块
    if len(chunk_features) == 0:
        # 创建一个默认特征块
        chunk_features = [np.zeros((1, N_MFCC * 3))]
    
    return chunk_features, None

def safe_extract_features_streaming(audio_chunk, sr, prev_frames=None):
    """
    安全地调用extract_features_streaming，确保总是返回正确的元组
    
    Args:
        audio_chunk: 当前音频数据块
        sr: 采样率
        prev_frames: 前一时刻的特征帧
        
    Returns:
        (features, last_frames): 即使发生错误也保证返回两个值的元组
    """
    try:
        return extract_features_streaming(audio_chunk, sr, prev_frames=prev_frames)
    except Exception as e:
        print(f"特征提取错误 (安全处理): {e}")
        # 返回默认值
        empty_features = np.zeros((1, N_MFCC * 3))
        return empty_features, audio_chunk[-min(len(audio_chunk), HOP_LENGTH*2):] if len(audio_chunk) > 0 else np.zeros(HOP_LENGTH*2) 