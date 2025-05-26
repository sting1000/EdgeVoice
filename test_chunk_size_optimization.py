#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Chunk Size优化测试脚本
测试不同chunk size对流式语音识别性能的影响
"""

import os
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict

from models.streaming_conformer import StreamingConformer
from streaming_dataset import StreamingAudioDataset
from config import *

def test_chunk_size_performance(model_path, test_annotation_file, data_dir, 
                               chunk_sizes=[50, 100, 150, 200, 250, 300], 
                               device=DEVICE):
    """测试不同chunk size的性能
    
    Args:
        model_path: 模型路径
        test_annotation_file: 测试集标注文件
        data_dir: 数据目录
        chunk_sizes: 要测试的chunk size列表
        device: 设备
        
    Returns:
        results: 测试结果字典
    """
    # 加载模型
    model = StreamingConformer(
        input_dim=N_MFCC * 3,
        hidden_dim=CONFORMER_HIDDEN_SIZE,
        num_classes=len(INTENT_CLASSES),
        num_layers=CONFORMER_LAYERS,
        num_heads=CONFORMER_ATTENTION_HEADS,
        dropout=CONFORMER_DROPOUT,
        kernel_size=CONFORMER_CONV_KERNEL_SIZE,
        expansion_factor=CONFORMER_FF_EXPANSION_FACTOR
    )
    
    # 加载权重
    checkpoint = torch.load(model_path, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model = model.to(device)
    model.eval()
    
    results = {}
    
    for chunk_size in chunk_sizes:
        print(f"\n测试 Chunk Size: {chunk_size} 帧 ({chunk_size * 0.01:.2f}秒)")
        
        # 创建测试数据集
        test_dataset = StreamingAudioDataset(
            annotation_file=test_annotation_file,
            data_dir=data_dir,
            streaming_mode=True,
            chunk_size=chunk_size,
            step_size=chunk_size // 2  # 50%重叠
        )
        
        # 测试指标
        correct = 0
        total = 0
        confidences = []
        latencies = []
        prediction_changes = []
        
        # 逐样本测试
        for i in tqdm(range(len(test_dataset)), desc=f"Chunk {chunk_size}"):
            sample = test_dataset[i]
            chunk_features = sample['chunk_features']
            true_label = sample['label']
            
            # 重置模型状态
            model.reset_streaming_state()
            cached_states = None
            
            # 模拟流式处理
            predictions = []
            confs = []
            
            for chunk in chunk_features:
                chunk_tensor = torch.FloatTensor(chunk).unsqueeze(0).to(device)
                
                with torch.no_grad():
                    pred, conf, cached_states = model.predict_streaming(chunk_tensor, cached_states)
                
                predictions.append(pred.item())
                confs.append(conf.item())
            
            # 最终预测
            if predictions:
                final_pred = predictions[-1]
                final_conf = confs[-1]
                
                # 统计
                total += 1
                correct += (final_pred == true_label)
                confidences.append(final_conf)
                
                # 计算预测变化次数
                changes = sum(1 for j in range(1, len(predictions)) 
                             if predictions[j] != predictions[j-1])
                prediction_changes.append(changes)
                
                # 计算延迟（到达最终预测的时间）
                latency = len(predictions) * chunk_size * 0.01  # 秒
                latencies.append(latency)
        
        # 计算指标
        accuracy = correct / total if total > 0 else 0
        avg_confidence = np.mean(confidences) if confidences else 0
        avg_latency = np.mean(latencies) if latencies else 0
        avg_changes = np.mean(prediction_changes) if prediction_changes else 0
        
        results[chunk_size] = {
            'accuracy': accuracy,
            'avg_confidence': avg_confidence,
            'avg_latency': avg_latency,
            'avg_prediction_changes': avg_changes,
            'total_samples': total,
            'chunk_duration_ms': chunk_size * 10  # 毫秒
        }
        
        print(f"准确率: {accuracy:.3f}")
        print(f"平均置信度: {avg_confidence:.3f}")
        print(f"平均延迟: {avg_latency:.3f}秒")
        print(f"平均预测变化: {avg_changes:.1f}次")
    
    return results

def analyze_audio_duration_distribution(annotation_file, data_dir):
    """分析音频时长分布
    
    Args:
        annotation_file: 标注文件
        data_dir: 数据目录
        
    Returns:
        duration_stats: 时长统计信息
    """
    import librosa
    
    df = pd.read_csv(annotation_file)
    durations = []
    
    print("分析音频时长分布...")
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        file_path = os.path.join(data_dir, row['file_path'])
        if os.path.exists(file_path):
            try:
                audio, sr = librosa.load(file_path, sr=TARGET_SAMPLE_RATE)
                duration = len(audio) / sr
                durations.append(duration)
            except:
                continue
    
    durations = np.array(durations)
    
    duration_stats = {
        'mean': np.mean(durations),
        'median': np.median(durations),
        'std': np.std(durations),
        'min': np.min(durations),
        'max': np.max(durations),
        'percentiles': {
            '25%': np.percentile(durations, 25),
            '50%': np.percentile(durations, 50),
            '75%': np.percentile(durations, 75),
            '90%': np.percentile(durations, 90),
            '95%': np.percentile(durations, 95),
            '99%': np.percentile(durations, 99)
        },
        'durations': durations
    }
    
    return duration_stats

def plot_results(results, duration_stats, save_path="chunk_size_analysis.png"):
    """绘制分析结果
    
    Args:
        results: 测试结果
        duration_stats: 时长统计
        save_path: 保存路径
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Chunk Size 优化分析', fontsize=16, fontweight='bold')
    
    chunk_sizes = list(results.keys())
    accuracies = [results[cs]['accuracy'] for cs in chunk_sizes]
    confidences = [results[cs]['avg_confidence'] for cs in chunk_sizes]
    latencies = [results[cs]['avg_latency'] for cs in chunk_sizes]
    changes = [results[cs]['avg_prediction_changes'] for cs in chunk_sizes]
    durations_ms = [results[cs]['chunk_duration_ms'] for cs in chunk_sizes]
    
    # 1. 准确率 vs Chunk Size
    axes[0, 0].plot(durations_ms, accuracies, 'bo-', linewidth=2, markersize=8)
    axes[0, 0].set_xlabel('Chunk Duration (ms)')
    axes[0, 0].set_ylabel('Accuracy')
    axes[0, 0].set_title('准确率 vs Chunk Size')
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. 置信度 vs Chunk Size
    axes[0, 1].plot(durations_ms, confidences, 'go-', linewidth=2, markersize=8)
    axes[0, 1].set_xlabel('Chunk Duration (ms)')
    axes[0, 1].set_ylabel('Average Confidence')
    axes[0, 1].set_title('平均置信度 vs Chunk Size')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. 延迟 vs Chunk Size
    axes[0, 2].plot(durations_ms, latencies, 'ro-', linewidth=2, markersize=8)
    axes[0, 2].set_xlabel('Chunk Duration (ms)')
    axes[0, 2].set_ylabel('Average Latency (s)')
    axes[0, 2].set_title('平均延迟 vs Chunk Size')
    axes[0, 2].grid(True, alpha=0.3)
    
    # 4. 预测变化次数 vs Chunk Size
    axes[1, 0].plot(durations_ms, changes, 'mo-', linewidth=2, markersize=8)
    axes[1, 0].set_xlabel('Chunk Duration (ms)')
    axes[1, 0].set_ylabel('Average Prediction Changes')
    axes[1, 0].set_title('平均预测变化次数 vs Chunk Size')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 5. 音频时长分布
    axes[1, 1].hist(duration_stats['durations'], bins=50, alpha=0.7, color='skyblue', edgecolor='black')
    axes[1, 1].axvline(duration_stats['mean'], color='red', linestyle='--', label=f'Mean: {duration_stats["mean"]:.2f}s')
    axes[1, 1].axvline(duration_stats['median'], color='green', linestyle='--', label=f'Median: {duration_stats["median"]:.2f}s')
    axes[1, 1].axvline(duration_stats['percentiles']['95%'], color='orange', linestyle='--', label=f'95%: {duration_stats["percentiles"]["95%"]:.2f}s')
    axes[1, 1].set_xlabel('Duration (s)')
    axes[1, 1].set_ylabel('Count')
    axes[1, 1].set_title('音频时长分布')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    # 6. 综合性能评分
    # 归一化指标并计算综合得分
    norm_acc = np.array(accuracies) / max(accuracies)
    norm_conf = np.array(confidences) / max(confidences)
    norm_latency = 1 - (np.array(latencies) / max(latencies))  # 延迟越低越好
    norm_changes = 1 - (np.array(changes) / max(changes))  # 变化越少越好
    
    # 加权综合得分
    composite_score = 0.4 * norm_acc + 0.3 * norm_conf + 0.2 * norm_latency + 0.1 * norm_changes
    
    axes[1, 2].plot(durations_ms, composite_score, 'ko-', linewidth=2, markersize=8)
    axes[1, 2].set_xlabel('Chunk Duration (ms)')
    axes[1, 2].set_ylabel('Composite Score')
    axes[1, 2].set_title('综合性能评分')
    axes[1, 2].grid(True, alpha=0.3)
    
    # 标记最佳chunk size
    best_idx = np.argmax(composite_score)
    best_chunk_ms = durations_ms[best_idx]
    axes[1, 2].axvline(best_chunk_ms, color='red', linestyle='--', 
                       label=f'Best: {best_chunk_ms}ms')
    axes[1, 2].legend()
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    return best_chunk_ms

def main():
    """主函数"""
    # 配置
    model_path = "saved_models/streaming_conformer.pt"
    test_annotation_file = "data/split/test_annotations.csv"
    data_dir = "data"
    
    # 检查文件存在性
    if not os.path.exists(model_path):
        print(f"错误：模型文件不存在 {model_path}")
        return
    
    if not os.path.exists(test_annotation_file):
        print(f"错误：测试标注文件不存在 {test_annotation_file}")
        return
    
    print("=== Chunk Size 优化分析 ===")
    
    # 1. 分析音频时长分布
    print("\n1. 分析音频时长分布...")
    duration_stats = analyze_audio_duration_distribution(test_annotation_file, data_dir)
    
    print(f"音频时长统计:")
    print(f"  平均时长: {duration_stats['mean']:.2f}秒")
    print(f"  中位数: {duration_stats['median']:.2f}秒")
    print(f"  标准差: {duration_stats['std']:.2f}秒")
    print(f"  95%分位数: {duration_stats['percentiles']['95%']:.2f}秒")
    
    # 2. 基于时长分布确定测试的chunk sizes
    # 转换为帧数（10ms/帧）
    chunk_sizes = [
        50,   # 0.5秒
        100,  # 1.0秒
        150,  # 1.5秒
        200,  # 2.0秒（你当前的192接近这个值）
        250,  # 2.5秒
        300,  # 3.0秒
        400,  # 4.0秒
        500   # 5.0秒
    ]
    
    print(f"\n2. 测试不同Chunk Size的性能...")
    print(f"测试的Chunk Sizes: {[f'{cs}帧({cs*0.01:.1f}s)' for cs in chunk_sizes]}")
    
    # 3. 性能测试
    results = test_chunk_size_performance(
        model_path=model_path,
        test_annotation_file=test_annotation_file,
        data_dir=data_dir,
        chunk_sizes=chunk_sizes
    )
    
    # 4. 结果分析和可视化
    print("\n3. 生成分析报告...")
    best_chunk_ms = plot_results(results, duration_stats)
    
    # 5. 输出建议
    print(f"\n=== 优化建议 ===")
    print(f"最佳Chunk Size: {best_chunk_ms}ms ({best_chunk_ms/10}帧)")
    print(f"你当前设置的192帧 = {192*10}ms")
    
    if best_chunk_ms/10 > 192:
        print(f"建议增加chunk size到 {int(best_chunk_ms/10)} 帧")
    elif best_chunk_ms/10 < 192:
        print(f"建议减少chunk size到 {int(best_chunk_ms/10)} 帧")
    else:
        print("当前设置已经很接近最优值")
    
    # 6. 保存详细结果
    results_df = pd.DataFrame(results).T
    results_df.to_csv("chunk_size_optimization_results.csv")
    print(f"\n详细结果已保存到: chunk_size_optimization_results.csv")

if __name__ == "__main__":
    main() 