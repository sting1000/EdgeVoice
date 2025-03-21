#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import argparse
import torch
import numpy as np
import librosa
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import seaborn as sns

from models.fast_classifier import FastIntentClassifier
from utils.feature_extraction import streaming_feature_extractor
from config import *

def load_streaming_model(model_path):
    """
    加载流式训练的模型
    
    Args:
        model_path: 模型文件路径
        
    Returns:
        model: 加载的模型
        intent_labels: 意图标签列表
    """
    # 加载模型
    checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
    
    # 获取意图标签
    intent_labels = checkpoint['intent_labels']
    
    # 创建模型实例
    input_size = N_MFCC * 3
    model = FastIntentClassifier(input_size=input_size, num_classes=len(intent_labels))
    
    # 加载模型权重
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    return model, intent_labels

def evaluate_streaming_model(model, intent_labels, annotation_file, data_dir=DATA_DIR,
                            chunk_size=STREAMING_CHUNK_SIZE, step_size=STREAMING_STEP_SIZE,
                            confidence_threshold=0.9, use_majority_voting=True, device='cpu'):
    """
    评估流式模型性能
    
    Args:
        model: 模型实例
        intent_labels: 意图标签列表
        annotation_file: 标注文件路径
        data_dir: 音频文件目录
        chunk_size: 流式处理的块大小
        step_size: 流式处理的步长
        confidence_threshold: 早停置信度阈值
        use_majority_voting: 是否使用多数投票
        device: 运行设备
        
    Returns:
        metrics: 评估指标
    """
    # 将模型放到指定设备
    model = model.to(device)
    model.eval()
    
    # 加载测试数据
    df = pd.read_csv(annotation_file)
    
    # 转换标签到ID
    label_to_id = {label: i for i, label in enumerate(intent_labels)}
    id_to_label = {i: label for i, label in enumerate(intent_labels)}
    
    # 评估指标
    all_true_labels = []
    all_pred_labels = []
    all_confidences = []
    early_stopping_count = 0
    
    # 存储每个块的预测结果（用于分析）
    chunk_predictions = {}
    
    # 流式处理的延迟统计
    latencies = []
    end_to_decision_times = []
    
    # 处理每个音频文件
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="评估中"):
        file_path = os.path.join(data_dir, row['file_path'])
        true_label = label_to_id[row['intent']]
        
        try:
            # 加载音频
            audio, sr = librosa.load(file_path, sr=TARGET_SAMPLE_RATE)
            
            # 提取chunk特征
            try:
                # 尝试解包两个返回值
                chunk_features, _ = streaming_feature_extractor(audio, sr, chunk_size, step_size)
            except ValueError:
                # 如果返回一个值，则直接赋值给chunk_features
                chunk_features = streaming_feature_extractor(audio, sr, chunk_size, step_size)
            
            if len(chunk_features) == 0:
                print(f"警告: 文件没有提取到有效特征: {file_path}, 跳过评估")
                continue
                
            # 初始化模型状态
            cached_states = None
            file_predictions = []
            
            # 模拟流式处理
            early_stopped = False
            start_time = len(audio) / sr  # 音频总时长
            decision_time = start_time
            
            for i, features in enumerate(chunk_features):
                # 安全检查
                if features is None or len(features) == 0:
                    print(f"警告: 跳过空特征块 {i} (文件: {file_path})")
                    continue
                    
                try:
                    features_tensor = torch.FloatTensor(features).unsqueeze(0).to(device)
                    
                    with torch.no_grad():
                        # 使用安全的前向传播
                        try:
                            # 尝试获取3个返回值
                            pred, conf, cached_states = model.predict_streaming(features_tensor, cached_states)
                        except ValueError as e:
                            print(f"处理块 {i} 时出错 (文件: {file_path}): {e}")
                            # 尝试只获取两个返回值
                            pred, conf = model.predict_streaming(features_tensor, cached_states)
                            # 保留前一个缓存状态
                        
                        # 记录预测结果
                        pred_label = pred.item()
                        confidence = conf.item()
                        file_predictions.append((pred_label, confidence))
                        
                        # 计算当前时间点
                        current_time = (i * step_size * HOP_LENGTH) / sr
                        
                        # 检查是否满足早停条件
                        if confidence > confidence_threshold and not early_stopped:
                            early_stopped = True
                            decision_time = current_time
                            early_stopping_count += 1
                
                except Exception as e:
                    print(f"处理块 {i} 时出错 (文件: {file_path}): {e}")
                    # 跳过有问题的块
                    continue
            
            # 如果没有有效预测，跳过
            if not file_predictions:
                print(f"警告: 文件没有生成有效预测: {file_path}, 跳过评估")
                continue
                
            # 计算延迟（从音频开始到得出决策的时间）
            latency = decision_time
            latencies.append(latency)
            
            # 计算从音频结束到决策的时间（负值表示在音频结束前做出决策）
            end_to_decision_time = decision_time - start_time
            end_to_decision_times.append(end_to_decision_time)
            
            # 存储块预测结果
            chunk_predictions[row['file_path']] = file_predictions
            
            # 最终预测结果
            if use_majority_voting:
                # 多数投票（加权）
                vote_count = np.zeros(len(intent_labels))
                for pred, conf in file_predictions:
                    vote_count[pred] += conf
                
                final_pred = np.argmax(vote_count)
                final_conf = vote_count[final_pred] / len(file_predictions)
            else:
                # 使用最后一个块的预测
                final_pred, final_conf = file_predictions[-1]
            
            # 记录真实标签和预测结果
            all_true_labels.append(true_label)
            all_pred_labels.append(final_pred)
            all_confidences.append(final_conf)
            
        except Exception as e:
            print(f"处理文件时出错: {file_path}, 错误: {e}")
    
    # 计算评估指标
    accuracy = accuracy_score(all_true_labels, all_pred_labels)
    report = classification_report(all_true_labels, all_pred_labels, 
                                  target_names=intent_labels, output_dict=True,
                                  zero_division=0)
    
    # 计算早停比例
    early_stopping_ratio = early_stopping_count / len(df) if len(df) > 0 else 0
    
    # 计算平均延迟
    average_latency = np.mean(latencies) if latencies else 0
    
    # 分析每个句子的预测变化
    prediction_stability = analyze_prediction_stability(chunk_predictions, intent_labels)
    
    # 检查是否有有效的预测结果
    if not all_true_labels or not all_pred_labels:
        print("警告：没有有效的预测结果，可能是由于处理音频文件时出错")
        # 创建一个空的分类报告
        report = {label: {'precision': 0, 'recall': 0, 'f1-score': 0, 'support': 0} for label in intent_labels}
        report.update({'accuracy': 0, 'macro avg': {'precision': 0, 'recall': 0, 'f1-score': 0, 'support': 0},
                      'weighted avg': {'precision': 0, 'recall': 0, 'f1-score': 0, 'support': 0}})
        accuracy = 0
    
    # 绘制混淆矩阵
    if all_true_labels and all_pred_labels:
        cm = confusion_matrix(all_true_labels, all_pred_labels)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=intent_labels, yticklabels=intent_labels)
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.title('Confusion Matrix')
        plt.tight_layout()
        plt.savefig('streaming_confusion_matrix.png')
        plt.close()
    else:
        print("警告：无法绘制混淆矩阵，没有足够的预测结果")
    
    # 汇总指标
    metrics = {
        'accuracy': accuracy,
        'classification_report': report,
        'early_stopping_ratio': early_stopping_ratio,
        'average_latency': average_latency,
        'average_end_to_decision_time': np.mean(end_to_decision_times) if end_to_decision_times else 0,
        'prediction_stability': prediction_stability
    }
    
    return metrics

def analyze_prediction_stability(chunk_predictions, intent_labels):
    """
    分析预测的稳定性
    
    Args:
        chunk_predictions: 每个文件各块的预测结果
        intent_labels: 意图标签列表
        
    Returns:
        stability_metrics: 稳定性指标
    """
    # 存储指标
    prediction_changes = []
    final_confidence = []
    
    # 检查是否有预测数据
    if not chunk_predictions:
        print("警告：没有有效的预测数据用于稳定性分析")
        return {
            'average_changes': 0,
            'average_final_confidence': 0
        }
    
    # 分析每个文件
    for file_path, predictions in chunk_predictions.items():
        if len(predictions) <= 1:
            continue
            
        # 计算预测变化次数
        changes = 0
        prev_pred = predictions[0][0]
        
        for i in range(1, len(predictions)):
            curr_pred = predictions[i][0]
            if curr_pred != prev_pred:
                changes += 1
                prev_pred = curr_pred
        
        # 记录变化次数和最终置信度
        prediction_changes.append(changes)
        final_confidence.append(predictions[-1][1])
    
    # 计算平均变化次数和平均最终置信度
    avg_changes = np.mean(prediction_changes) if prediction_changes else 0
    avg_final_confidence = np.mean(final_confidence) if final_confidence else 0
    
    # 只在有数据时绘制图表
    if prediction_changes:
        # 绘制预测变化分布
        plt.figure(figsize=(10, 5))
        plt.hist(prediction_changes, bins=range(max(prediction_changes) + 2), alpha=0.7)
        plt.xlabel('Prediction Changes')
        plt.ylabel('File Count')
        plt.title('Prediction Stability Analysis')
        plt.grid(True, alpha=0.3)
        plt.savefig('prediction_stability.png')
        plt.close()
    else:
        print("警告：没有足够的预测变化数据来绘制稳定性分析图")
    
    return {
        'average_changes': avg_changes,
        'average_final_confidence': avg_final_confidence
    }

def plot_streaming_metrics(metrics, save_path='streaming_metrics.png'):
    """
    绘制流式评估指标
    
    Args:
        metrics: 评估指标
        save_path: 保存路径
    """
    # 检查是否有有效的评估指标
    if metrics['accuracy'] == 0 and not any(metrics['classification_report'].get(label, {}).get('f1-score', 0) 
                                          for label in metrics['classification_report'] 
                                          if label not in ['accuracy', 'macro avg', 'weighted avg']):
        print("警告：没有有效的评估指标数据，跳过绘图")
        return
    
    # 提取分类报告中的F1分数
    f1_scores = []
    classes = []
    
    for label, values in metrics['classification_report'].items():
        if label not in ['accuracy', 'macro avg', 'weighted avg']:
            classes.append(label)
            f1_scores.append(values['f1-score'])
    
    # 绘制F1分数
    plt.figure(figsize=(15, 10))
    
    # F1分数柱状图
    plt.subplot(2, 2, 1)
    plt.bar(classes, f1_scores, color='skyblue')
    plt.xlabel('Classes')
    plt.ylabel('F1 Score')
    plt.title('F1 Score for Each Class')
    plt.xticks(rotation=45)
    
    # 总体准确率和早停比例
    plt.subplot(2, 2, 2)
    metrics_labels = ['Accuracy', 'Early Stopping Ratio']
    metrics_values = [metrics['accuracy'], metrics['early_stopping_ratio']]
    plt.bar(metrics_labels, metrics_values, color=['green', 'orange'])
    plt.ylim(0, 1)
    plt.title('Overall Performance Metrics')
    
    # 延迟时间（秒）
    plt.subplot(2, 2, 3)
    delay_labels = ['Start-to-Decision', 'End-to-Decision']
    delay_values = [metrics['average_latency'], metrics['average_end_to_decision_time']]
    plt.bar(delay_labels, delay_values, color=['red', 'darkred'])
    plt.ylabel('Time (seconds)')
    plt.title('Average Decision Delay')
    
    # 预测稳定性
    plt.subplot(2, 2, 4)
    stability = metrics['prediction_stability']
    plt.bar(['Average Changes', 'Average Final Confidence'], 
           [stability['average_changes'], stability['average_final_confidence']],
           color=['purple', 'brown'])
    plt.title('Prediction Stability Metrics')
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='评估流式模型')
    parser.add_argument('--model_path', type=str, required=True, help='模型文件路径')
    parser.add_argument('--annotation_file', type=str, required=True, help='测试集标注文件')
    parser.add_argument('--data_dir', type=str, default=DATA_DIR, help='音频文件目录')
    parser.add_argument('--chunk_size', type=int, default=STREAMING_CHUNK_SIZE, help='流式处理块大小')
    parser.add_argument('--step_size', type=int, default=STREAMING_STEP_SIZE, help='流式处理步长')
    parser.add_argument('--confidence_threshold', type=float, default=0.9, help='早停置信度阈值')
    parser.add_argument('--no_majority_voting', action='store_true', help='禁用多数投票')
    
    return parser.parse_args()

def main():
    """主函数"""
    args = parse_args()
    
    # 确定设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'使用设备: {device}')
    
    # 加载模型
    print(f'加载模型: {args.model_path}')
    model, intent_labels = load_streaming_model(args.model_path)
    print(f'识别类别: {intent_labels}')
    
    # 评估模型
    print('开始评估流式模型...')
    try:
        metrics = evaluate_streaming_model(
            model=model,
            intent_labels=intent_labels,
            annotation_file=args.annotation_file,
            data_dir=args.data_dir,
            chunk_size=args.chunk_size,
            step_size=args.step_size,
            confidence_threshold=args.confidence_threshold,
            use_majority_voting=not args.no_majority_voting,
            device=device
        )
        
        # 打印评估结果
        print('\n评估结果:')
        print(f'准确率: {metrics["accuracy"]:.4f}')
        print(f'早停比例: {metrics["early_stopping_ratio"]:.4f}')
        print(f'从音频开始到决策的平均延迟: {metrics["average_latency"]:.4f} 秒')
        print(f'从音频结束到决策的平均时间: {metrics["average_end_to_decision_time"]:.4f} 秒' + 
              (' (负值表示在音频结束前已完成决策)' if metrics["average_end_to_decision_time"] < 0 else ''))
        print('\n类别报告:')
        for label, values in metrics['classification_report'].items():
            if isinstance(values, dict):
                print(f'{label}: F1={values["f1-score"]:.4f}, Precision={values["precision"]:.4f}, Recall={values["recall"]:.4f}')
        
        # 预测稳定性
        stability = metrics['prediction_stability']
        print('\n预测稳定性:')
        print(f'平均变化次数: {stability["average_changes"]:.2f}')
        print(f'平均最终置信度: {stability["average_final_confidence"]:.4f}')
        
        # 绘制指标图表
        plot_streaming_metrics(metrics)
        print('\n评估图表已保存为 streaming_metrics.png 和 streaming_confusion_matrix.png')
        print('预测稳定性分析已保存为 prediction_stability.png')
    except Exception as e:
        print(f"评估过程中发生错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main() 