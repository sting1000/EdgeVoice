#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
模型分析脚本
用于分析模型在数据集上的性能，创建混淆矩阵并分析易混淆的类别
"""

import os
import sys
import time
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, classification_report
import librosa

from config import *
from inference import IntentInferenceEngine
from augmented_dataset import prepare_augmented_dataloader, standardize_audio_length

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='语音意图模型分析工具')
    parser.add_argument('--data_dir', type=str, default=DATA_DIR, help='数据目录')
    parser.add_argument('--annotation_file', type=str, default=os.path.join(DATA_DIR, 'annotations.csv'), 
                       help='注释文件路径')
    parser.add_argument('--model_path', type=str, default=os.path.join(MODEL_DIR, 'fast_model.pth'), 
                       help='模型文件路径')
    parser.add_argument('--batch_size', type=int, default=8, help='批处理大小')
    parser.add_argument('--confidence_threshold', type=float, default=0.6, 
                       help='快速模型置信度阈值')
    parser.add_argument('--precise_model_path', type=str, default=None,
                       help='精确模型文件路径（可选）')
    parser.add_argument('--output_dir', type=str, default='analysis_results',
                       help='分析结果输出目录')
    parser.add_argument('--show_plots', action='store_true', 
                       help='是否显示图表（而不仅保存）')
    
    return parser.parse_args()

def plot_confusion_matrix(cm, class_names, output_path=None, title='Confusion Matrix', show=False):
    """绘制混淆矩阵热图"""
    # 重置字体设置，使用系统默认字体
    plt.rcParams.update({'font.family': 'sans-serif'})
    plt.rcParams['axes.unicode_minus'] = False   # 正确显示负号
    
    plt.figure(figsize=(10, 8))
    
    # 将混淆矩阵转换为百分比
    cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
    
    # 创建热图
    ax = sns.heatmap(
        cm_percent, 
        annot=cm,  # 显示原始计数
        fmt='d',   # 整数格式
        cmap='Blues', 
        xticklabels=class_names,
        yticklabels=class_names
    )
    
    # 设置坐标轴标签
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title(title)
    
    # 调整标签布局
    plt.tight_layout()
    
    # 保存图表
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"混淆矩阵已保存到: {output_path}")
    
    # 显示图表
    if show:
        plt.show()
    else:
        plt.close()

def plot_class_accuracy(class_metrics, output_path=None, title='Class Accuracy', show=False):
    """绘制各类别准确率柱状图"""
    # 重置字体设置，使用系统默认字体
    plt.rcParams.update({'font.family': 'sans-serif'})
    plt.rcParams['axes.unicode_minus'] = False   # 正确显示负号
    
    classes = list(class_metrics.keys())
    accuracies = [metrics['accuracy'] for metrics in class_metrics.values()]
    
    # 按准确率排序
    sorted_indices = np.argsort(accuracies)
    classes = [classes[i] for i in sorted_indices]
    accuracies = [accuracies[i] for i in sorted_indices]
    
    plt.figure(figsize=(12, 6))
    bars = plt.barh(classes, accuracies, color='skyblue')
    
    # 添加数值标签
    for i, bar in enumerate(bars):
        plt.text(
            bar.get_width() + 0.01, 
            bar.get_y() + bar.get_height()/2, 
            f'{accuracies[i]:.2f}', 
            va='center'
        )
    
    plt.xlabel('Accuracy')
    plt.ylabel('Class')
    plt.title(title)
    plt.xlim(0, 1.1)  # 限制x轴范围
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    
    # 保存图表
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"类别准确率图表已保存到: {output_path}")
    
    # 显示图表
    if show:
        plt.show()
    else:
        plt.close()

def plot_confusion_pairs(class_metrics, output_path=None, min_rate=0.1, title='Confusion Pairs', show=False):
    """绘制主要混淆类别对图表"""
    # 重置字体设置，使用系统默认字体
    plt.rcParams.update({'font.family': 'sans-serif'})
    plt.rcParams['axes.unicode_minus'] = False   # 正确显示负号
    
    confusion_pairs = []
    
    # 收集所有混淆率大于阈值的类别对
    for true_class, metrics in class_metrics.items():
        for confused_class, rate in metrics['confusion_rates']:
            if rate >= min_rate:
                confusion_pairs.append((true_class, confused_class, rate))
    
    # 按混淆率排序
    confusion_pairs.sort(key=lambda x: x[2], reverse=True)
    
    # 如果没有混淆对，返回
    if not confusion_pairs:
        print(f"没有混淆率大于 {min_rate} 的类别对")
        return
    
    # 提取数据
    true_classes = [p[0] for p in confusion_pairs]
    confused_classes = [p[1] for p in confusion_pairs]
    rates = [p[2] for p in confusion_pairs]
    
    # 创建标签
    labels = [f"{t} → {c}" for t, c in zip(true_classes, confused_classes)]
    
    plt.figure(figsize=(12, max(6, len(confusion_pairs) * 0.4)))
    bars = plt.barh(labels, rates, color='salmon')
    
    # 添加数值标签
    for i, bar in enumerate(bars):
        plt.text(
            bar.get_width() + 0.01, 
            bar.get_y() + bar.get_height()/2, 
            f'{rates[i]:.2f}', 
            va='center'
        )
    
    plt.xlabel('Confusion Rate')
    plt.ylabel('Class Pairs')
    plt.title(title)
    plt.xlim(0, min(1.1, max(rates) + 0.1))  # 限制x轴范围
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    
    # 保存图表
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"混淆类别对图表已保存到: {output_path}")
    
    # 显示图表
    if show:
        plt.show()
    else:
        plt.close()

def analyze_model(args):
    """分析模型性能"""
    # 确保输出目录存在
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 初始化推理引擎
    print(f"加载模型: {args.model_path}")
    engine = IntentInferenceEngine(
        fast_model_path=args.model_path,
        precise_model_path=args.precise_model_path,
        fast_confidence_threshold=args.confidence_threshold
    )
    
    # 获取模型所需的特征维度
    try:
        input_size = 39  # 默认特征维度
        enhanced_features = False
        context_frames = 0
        
        # 尝试从模型中读取特征配置
        if hasattr(engine, 'fast_model') and engine.fast_model is not None:
            # 检查模型的输入尺寸
            if hasattr(engine.fast_model, 'input_projection'):
                input_size = engine.fast_model.input_projection.in_features
                print(f"从模型读取特征维度: {input_size}")
        
        # 尝试从模型加载的checkpoint读取特征配置
        try:
            checkpoint = torch.load(args.model_path, map_location='cpu')
            if isinstance(checkpoint, dict):
                enhanced_features = checkpoint.get('enhanced_features', False)
                context_frames = checkpoint.get('context_frames', 0)
                if 'input_size' in checkpoint:
                    input_size = checkpoint['input_size']
                    print(f"从模型checkpoint读取特征维度: {input_size}")
            
            print(f"特征配置: 增强特征={enhanced_features}, 上下文帧数={context_frames}")
        except Exception as e:
            print(f"读取checkpoint时出错: {e}")
    except Exception as e:
        print(f"获取模型配置时出错: {e}")
    
    # 定义特征提取器，与模型匹配
    def feature_extractor(audio, sr, **kwargs):
        # 标准化音频长度
        audio = standardize_audio_length(audio, sr)
        
        # 提取MFCC特征和动态特征
        mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
        delta_mfccs = librosa.feature.delta(mfccs)
        delta2_mfccs = librosa.feature.delta(mfccs, order=2)
        
        # 合并基本特征
        features = np.concatenate([mfccs, delta_mfccs, delta2_mfccs], axis=0)
        
        # 转置为(time, features)格式
        features = features.T
        
        # 如果需要添加上下文
        if context_frames > 0:
            window_size = 2 * context_frames + 1
            padded = np.pad(features, ((context_frames, context_frames), (0, 0)), mode='constant')
            context_features = []
            
            for i in range(len(features)):
                # 提取窗口上下文
                window = padded[i:i+window_size]
                # 展平
                context_feat = window.flatten()
                context_features.append(context_feat)
            
            features = np.array(context_features)
        
        return features
    
    # 使用数据加载器
    print("\n数据加载准备...")
    train_loader, train_labels = prepare_augmented_dataloader(
        annotation_file=args.annotation_file,
        data_dir=args.data_dir,
        feature_extractor=feature_extractor,
        batch_size=args.batch_size,
        augment=False,  # 分析时不使用增强
        use_cache=True
    )
    
    # 运行分析
    print("\n开始模型评估...")
    device = torch.device(DEVICE)
    model = engine.fast_model.to(device)
    model.eval()
    
    # 存储预测结果
    predictions = []
    true_labels = []
    confidences = []
    file_paths = []
    
    # 分批评估
    total_samples = len(train_loader.dataset)
    
    with torch.no_grad():
        for batch in tqdm(train_loader, desc="评估进度"):
            features = batch['features'].to(device)
            labels = batch['label'].to(device)
            file_path = batch['file_path']
            
            # 前向传播
            outputs = model(features)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            
            # 获取预测结果
            confidence, pred = torch.max(probs, dim=1)
            
            # 存储结果
            predictions.extend(pred.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())
            confidences.extend(confidence.cpu().numpy())
            file_paths.extend(file_path)
    
    # 转换标签从索引到类别名称
    prediction_classes = [train_labels[p] for p in predictions]
    true_label_classes = [train_labels[t] for t in true_labels]
    
    # 计算整体准确率
    correct = sum(1 for p, t in zip(prediction_classes, true_label_classes) if p == t)
    accuracy = correct / len(true_label_classes) if true_label_classes else 0
    print(f"\n整体准确率: {accuracy:.4f} ({correct}/{len(true_label_classes)})")
    
    # 创建混淆矩阵
    unique_labels = sorted(set(true_label_classes) | set(prediction_classes))
    
    # 创建混淆矩阵
    cm = confusion_matrix(true_label_classes, prediction_classes, labels=unique_labels)
    
    # 计算每个类别的准确率和错误率
    class_metrics = {}
    for i, label in enumerate(unique_labels):
        if sum(cm[i]) > 0:  # 避免除以零
            class_accuracy = cm[i, i] / sum(cm[i])
            # 找出最容易混淆的类别（除了自身外错误率最高的）
            confusion_rates = [(unique_labels[j], cm[i, j] / sum(cm[i])) 
                              for j in range(len(unique_labels)) if j != i and cm[i, j] > 0]
            confusion_rates.sort(key=lambda x: x[1], reverse=True)
            
            class_metrics[label] = {
                'accuracy': class_accuracy,
                'confusion_rates': confusion_rates
            }
    
    # 输出分类报告
    class_report = classification_report(true_label_classes, prediction_classes, labels=unique_labels)
    print("\n分类报告:")
    print(class_report)
    
    # 创建混淆矩阵热图
    cm_output_path = os.path.join(args.output_dir, 'confusion_matrix.png')
    plot_confusion_matrix(
        cm=cm,
        class_names=unique_labels,
        output_path=cm_output_path,
        show=args.show_plots
    )
    
    # 创建类别准确率图表
    acc_output_path = os.path.join(args.output_dir, 'class_accuracy.png')
    plot_class_accuracy(
        class_metrics=class_metrics,
        output_path=acc_output_path,
        show=args.show_plots
    )
    
    # 创建混淆类别对图表
    pairs_output_path = os.path.join(args.output_dir, 'confusion_pairs.png')
    plot_confusion_pairs(
        class_metrics=class_metrics,
        output_path=pairs_output_path,
        min_rate=0.1,  # 只显示混淆率大于10%的类别对
        show=args.show_plots
    )
    
    # 保存分析结果到文本文件
    results_file = os.path.join(args.output_dir, 'analysis_results.txt')
    with open(results_file, 'w', encoding='utf-8') as f:
        f.write("模型分析结果\n")
        f.write("=" * 50 + "\n\n")
        
        f.write(f"模型文件: {args.model_path}\n")
        f.write(f"数据集: {args.annotation_file}\n")
        f.write(f"分析时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("类别准确率:\n")
        for label, metrics in sorted(class_metrics.items(), 
                                    key=lambda x: x[1]['accuracy'], reverse=True):
            f.write(f"  {label}: {metrics['accuracy']:.4f}\n")
        
        f.write("\n主要混淆类别对 (混淆率 > 10%):\n")
        for label, metrics in class_metrics.items():
            high_confusion = [(cl, rt) for cl, rt in metrics['confusion_rates'] if rt > 0.1]
            if high_confusion:
                f.write(f"  {label}:\n")
                for confused_label, rate in high_confusion:
                    f.write(f"    → {confused_label}: {rate:.4f} ({rate*100:.1f}%)\n")
    
    print(f"分析结果已保存到: {results_file}")
    
    return class_metrics, cm, unique_labels

def main():
    """主函数"""
    args = parse_args()
    
    print("=" * 50)
    print("EdgeVoice 模型分析工具")
    print("=" * 50)
    
    print("\n使用以下参数:")
    print(f"  数据目录: {args.data_dir}")
    print(f"  注释文件: {args.annotation_file}")
    print(f"  模型文件: {args.model_path}")
    print(f"  置信度阈值: {args.confidence_threshold}")
    print(f"  输出目录: {args.output_dir}")
    
    try:
        analyze_model(args)
    except Exception as e:
        print(f"分析过程中出错: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 