#!/usr/bin/env python
# -*- coding: utf-8 -*-
# utils/asr_utils.py

"""
ASR相关工具函数，辅助WeNet ASR模型的数据处理、转写和评估。
"""

import os
import json
import time
import numpy as np
import torch
import librosa

def ensure_sample_rate(audio, original_sr, target_sr=16000):
    """确保音频采样率正确
    
    Args:
        audio: 音频数据
        original_sr: 原始采样率
        target_sr: 目标采样率
        
    Returns:
        重采样后的音频
    """
    if original_sr != target_sr:
        audio = librosa.resample(audio, orig_sr=original_sr, target_sr=target_sr)
    return audio

def wav_to_fbank(wav, sample_rate=16000, n_mels=80, frame_length=25, frame_shift=10):
    """将音频转换为Fbank特征
    
    Args:
        wav: 音频数据
        sample_rate: 采样率
        n_mels: Mel滤波器数量
        frame_length: 帧长（毫秒）
        frame_shift: 帧移（毫秒）
    
    Returns:
        fbank特征 [T, F]
    """
    # 将帧长和帧移从毫秒转换为采样点
    frame_length = int(sample_rate * frame_length / 1000)
    frame_shift = int(sample_rate * frame_shift / 1000)
    
    # 提取fbank特征
    fbank = librosa.feature.melspectrogram(
        y=wav, 
        sr=sample_rate,
        n_fft=frame_length,
        hop_length=frame_shift,
        n_mels=n_mels
    )
    
    # 转换为分贝
    fbank = librosa.power_to_db(fbank, ref=np.max)
    
    # 转置为[T, F]格式
    fbank = fbank.T
    
    return fbank

def normalize_fbank(fbank, mean=None, std=None):
    """对Fbank特征进行规范化
    
    Args:
        fbank: Fbank特征 [T, F]
        mean: 均值，如果为None则计算
        std: 标准差，如果为None则计算
    
    Returns:
        规范化后的特征, mean, std
    """
    if mean is None:
        mean = np.mean(fbank, axis=0)
    if std is None:
        std = np.std(fbank, axis=0)
    
    # 防止除零
    std = np.maximum(std, 1e-10)
    
    # 规范化
    normalized = (fbank - mean) / std
    
    return normalized, mean, std

def save_asr_result(result_dir, audio_id, text, confidence, metadata=None):
    """保存ASR转写结果
    
    Args:
        result_dir: 结果保存目录
        audio_id: 音频ID或时间戳
        text: 转写文本
        confidence: 置信度
        metadata: 额外元数据信息
    """
    os.makedirs(result_dir, exist_ok=True)
    
    result = {
        "id": audio_id,
        "text": text,
        "confidence": float(confidence),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    
    if metadata:
        result.update(metadata)
    
    # 保存为JSON格式
    result_file = os.path.join(result_dir, f"asr_result_{audio_id}.json")
    with open(result_file, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    
    return result_file

def cer_calculate(hypotheses, references):
    """计算字错误率(CER)
    
    Args:
        hypotheses: 预测文本列表
        references: 参考文本列表
    
    Returns:
        字错误率(CER)
    """
    total_chars = 0
    error_chars = 0
    
    for hyp, ref in zip(hypotheses, references):
        hyp_chars = list(hyp)
        ref_chars = list(ref)
        
        # 计算编辑距离
        d = _levenshtein_distance(hyp_chars, ref_chars)
        error_chars += d
        total_chars += len(ref_chars)
    
    if total_chars == 0:
        return 0
    
    cer = error_chars / total_chars
    return cer

def _levenshtein_distance(s1, s2):
    """计算Levenshtein距离（编辑距离）
    
    Args:
        s1: 序列1
        s2: 序列2
    
    Returns:
        编辑距离
    """
    if len(s1) > len(s2):
        s1, s2 = s2, s1
    
    distances = range(len(s1) + 1)
    for i2, c2 in enumerate(s2):
        distances_ = [i2+1]
        for i1, c1 in enumerate(s1):
            if c1 == c2:
                distances_.append(distances[i1])
            else:
                distances_.append(1 + min((distances[i1], distances[i1 + 1], distances_[-1])))
        distances = distances_
    
    return distances[-1]

def prepare_asr_data(audio_dir, annotation_file, output_dir):
    """准备ASR训练数据
    
    Args:
        audio_dir: 音频目录
        annotation_file: 标注文件
        output_dir: 输出目录
    
    Returns:
        数据文件数量
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 读取标注文件
    annotations = []
    with open(annotation_file, "r", encoding="utf-8") as f:
        # 假设标注文件为CSV格式，格式为: file_path,text
        # 跳过标题行
        header = f.readline()
        for line in f:
            parts = line.strip().split(",")
            if len(parts) >= 2:
                file_path = parts[0]
                text = parts[1]
                annotations.append((file_path, text))
    
    # 创建WeNet格式数据
    wav_scp = os.path.join(output_dir, "wav.scp")
    text_file = os.path.join(output_dir, "text")
    
    with open(wav_scp, "w", encoding="utf-8") as wav_f, \
         open(text_file, "w", encoding="utf-8") as text_f:
        
        for file_path, text in annotations:
            # 生成文件ID
            file_id = os.path.splitext(os.path.basename(file_path))[0]
            
            # 写入wav.scp
            full_path = os.path.join(audio_dir, file_path)
            wav_f.write(f"{file_id} {full_path}\n")
            
            # 写入text
            text_f.write(f"{file_id} {text}\n")
    
    print(f"准备完成 {len(annotations)} 条数据")
    return len(annotations) 