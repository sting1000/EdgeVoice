#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import argparse
import torch
import numpy as np
import time
import librosa
from tqdm import tqdm

from config import *
from models.fast_classifier import FastIntentClassifier
from utils.feature_extraction import extract_features

def parse_args():
    parser = argparse.ArgumentParser(description="EdgeVoice流式处理演示")
    parser.add_argument("--model_path", type=str, required=True, help="Fast模型路径")
    parser.add_argument("--audio_file", type=str, required=True, help="测试音频文件路径")
    parser.add_argument("--chunk_size", type=int, default=STREAMING_CHUNK_SIZE, help="每次处理的帧数")
    parser.add_argument("--step_size", type=int, default=STREAMING_STEP_SIZE, help="流式处理的步长")
    return parser.parse_args()

def load_model(model_path):
    """加载Fast模型"""
    # 加载模型配置和权重
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FastIntentClassifier(input_size=N_MFCC*3)  # 16*3=48维特征
    
    # 尝试加载模型
    try:
        # 首先尝试加载完整的模型字典
        checkpoint = torch.load(model_path, map_location=device)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            # 如果是训练过程中保存的完整检查点
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"成功加载模型检查点: {model_path}")
            intent_labels = checkpoint.get('intent_labels', INTENT_CLASSES)
            print(f"类别标签: {intent_labels}")
        else:
            # 直接加载状态字典
            model.load_state_dict(checkpoint)
            print(f"成功加载模型状态字典: {model_path}")
    except Exception as e:
        print(f"加载模型时发生错误: {e}")
        print("使用初始化模型继续...")
    
    model.to(device)
    model.eval()  # 设置为评估模式
    
    return model, device

def create_feature_chunks(features, chunk_size, step_size):
    """将特征序列分割为重叠的数据块"""
    chunks = []
    for i in range(0, features.shape[0] - chunk_size + 1, step_size):
        chunk = features[i:i+chunk_size]
        chunks.append(chunk)
    
    # 处理最后一个可能不完整的块
    if (features.shape[0] % step_size) != 0:
        last_chunk = features[-chunk_size:]
        chunks.append(last_chunk)
        
    return chunks

def simulate_streaming(model, features, chunk_size, step_size, device):
    """模拟流式处理"""
    # 准备特征块
    feature_chunks = create_feature_chunks(features, chunk_size, step_size)
    
    # 初始化状态
    cached_states = None
    
    # 记录每个时间步的结果
    results = []
    confidences = []
    processing_times = []
    
    print(f"\n模拟流式处理 {len(feature_chunks)} 个特征块...\n")
    
    # 对每个数据块进行处理
    for i, chunk in enumerate(tqdm(feature_chunks)):
        # 转换为tensor
        x = torch.FloatTensor(chunk).unsqueeze(0).to(device)  # [1, chunk_size, feature_dim]
        
        # 记录处理时间
        start_time = time.time()
        
        # 流式预测
        with torch.no_grad():
            pred, conf, cached_states = model.predict_streaming(x, cached_states)
        
        # 计算处理时间
        elapsed = (time.time() - start_time) * 1000  # 毫秒
        processing_times.append(elapsed)
        
        # 保存结果
        results.append(pred.item())
        confidences.append(conf.item())
        
        # 显示当前预测
        pred_class = INTENT_CLASSES[pred.item()]
        print(f"块 {i+1}/{len(feature_chunks)}: 预测={pred_class}, 置信度={conf.item():.4f}, 处理时间={elapsed:.2f}ms")
        
        # 如果置信度超过阈值，可以提前停止
        if conf.item() > FAST_CONFIDENCE_THRESHOLD:
            print(f"\n置信度达到阈值 {FAST_CONFIDENCE_THRESHOLD}，提前结束预测\n")
            break
    
    # 输出统计信息
    final_pred = INTENT_CLASSES[results[-1]]
    final_conf = confidences[-1]
    avg_time = np.mean(processing_times)
    total_time = np.sum(processing_times)
    
    print("\n流式处理统计:")
    print(f"最终预测: {final_pred}")
    print(f"最终置信度: {final_conf:.4f}")
    print(f"平均每块处理时间: {avg_time:.2f}ms")
    print(f"总处理时间: {total_time:.2f}ms")
    print(f"对应每帧(10ms)处理时间: {avg_time/chunk_size:.2f}ms")
    
    return results, confidences, processing_times

def main():
    args = parse_args()
    
    # 加载模型
    print("加载模型...")
    model, device = load_model(args.model_path)
    
    # 加载音频并提取特征
    print(f"加载音频文件: {args.audio_file}")
    audio, sr = librosa.load(args.audio_file, sr=TARGET_SAMPLE_RATE)
    
    print("提取特征...")
    features = extract_features(audio, sr)
    
    # 模拟流式处理
    simulate_streaming(model, features, args.chunk_size, args.step_size, device)

if __name__ == "__main__":
    main() 