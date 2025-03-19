#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import librosa
from config import *

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
    
    # 如果有前一时刻的帧，拼接以保证Delta计算的连续性
    if prev_frames is not None:
        # 需要2帧上下文来计算delta
        context_size = 2
        audio_with_context = np.concatenate([prev_frames, audio_chunk])
        
        # 提取MFCC
        mfcc = librosa.feature.mfcc(
            y=audio_with_context, 
            sr=TARGET_SAMPLE_RATE,
            n_mfcc=n_mfcc,
            n_fft=N_FFT,
            hop_length=HOP_LENGTH
        )
        
        # 计算差分特征
        delta = librosa.feature.delta(mfcc, order=1)
        delta2 = librosa.feature.delta(mfcc, order=2)
        
        # 转置和拼接
        mfcc = mfcc.T
        delta = delta.T
        delta2 = delta2.T
        
        # 只取当前块对应的帧
        # 计算前一块对应的帧数
        prev_frames_count = int(len(prev_frames) / HOP_LENGTH)
        
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
        
        delta = librosa.feature.delta(mfcc, order=1)
        delta2 = librosa.feature.delta(mfcc, order=2)
        
        mfcc = mfcc.T
        delta = delta.T
        delta2 = delta2.T
    
    # 合并特征
    features = np.hstack((mfcc, delta, delta2))
    
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