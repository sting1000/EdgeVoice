# evaluate.py
import os
import time
import argparse
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import classification_report, accuracy_score, precision_recall_fscore_support, confusion_matrix
import matplotlib.pyplot as plt
from datetime import datetime

from config import *
from data_utils import prepare_dataloader
from models.fast_classifier import FastIntentClassifier
from models.precise_classifier import PreciseIntentClassifier
from inference import IntentInferenceEngine

def evaluate_models(data_dir, annotation_file, fast_model_path, precise_model_path=None, output_dir='evaluation_results'):
    """
    评估fast和precise模型的性能，并将结果输出到Excel文件中
    
    Args:
        data_dir: 数据目录
        annotation_file: 注释文件路径
        fast_model_path: 快速分类器模型路径
        precise_model_path: 精确分类器模型路径（可选）
        output_dir: 输出目录
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 创建推理引擎
    inference_engine = IntentInferenceEngine(fast_model_path, precise_model_path)
    
    # 准备测试数据加载器
    print("准备测试数据...")
    fast_loader = prepare_dataloader(data_dir, annotation_file, batch_size=32, mode='fast')
    
    # 如果有精确模型，也准备对应的数据加载器
    if precise_model_path:
        precise_loader = prepare_dataloader(data_dir, annotation_file, batch_size=32, mode='precise')
    
    # 评估指标收集
    results = {
        'model': [],
        'accuracy': [],
        'precision': [],
        'recall': [],
        'f1': [],
        'avg_inference_time': [],
        'total_samples': [],
        'inference_times': [],
        'class_precisions': [],
        'class_recalls': [],
        'class_f1s': [],
    }
    
    # 1. 评估快速分类器
    print("\n开始评估快速分类器...")
    model_results = evaluate_fast_model(inference_engine, fast_loader)
    
    # 添加结果
    results['model'].append('Fast Model')
    results['accuracy'].append(model_results['accuracy'])
    results['precision'].append(model_results['precision'])
    results['recall'].append(model_results['recall'])
    results['f1'].append(model_results['f1'])
    results['avg_inference_time'].append(model_results['avg_inference_time'])
    results['total_samples'].append(model_results['total_samples'])
    results['inference_times'].append(model_results['inference_times'])
    results['class_precisions'].append(model_results['class_precision'])
    results['class_recalls'].append(model_results['class_recall'])
    results['class_f1s'].append(model_results['class_f1'])
    
    # 2. 如果有精确分类器，也评估它
    if precise_model_path:
        print("\n开始评估精确分类器...")
        model_results = evaluate_precise_model(inference_engine, precise_loader)
        
        # 添加结果
        results['model'].append('Precise Model')
        results['accuracy'].append(model_results['accuracy'])
        results['precision'].append(model_results['precision'])
        results['recall'].append(model_results['recall'])
        results['f1'].append(model_results['f1'])
        results['avg_inference_time'].append(model_results['avg_inference_time'])
        results['total_samples'].append(model_results['total_samples'])
        results['inference_times'].append(model_results['inference_times'])
        results['class_precisions'].append(model_results['class_precision'])
        results['class_recalls'].append(model_results['class_recall'])
        results['class_f1s'].append(model_results['class_f1'])
    
    # 3. 评估完整推理流程（两个模型协同）
    if precise_model_path:
        print("\n开始评估完整推理流程...")
        model_results = evaluate_full_pipeline(inference_engine, fast_loader, 
                                              confidence_threshold=FAST_CONFIDENCE_THRESHOLD)
        
        # 添加结果
        results['model'].append('Full Pipeline')
        results['accuracy'].append(model_results['accuracy'])
        results['precision'].append(model_results['precision'])
        results['recall'].append(model_results['recall'])
        results['f1'].append(model_results['f1'])
        results['avg_inference_time'].append(model_results['avg_inference_time'])
        results['total_samples'].append(model_results['total_samples'])
        results['inference_times'].append(model_results['inference_times'])
        results['class_precisions'].append(model_results['class_precision'])
        results['class_recalls'].append(model_results['class_recall'])
        results['class_f1s'].append(model_results['class_f1'])
    
    # 创建结果DataFrame
    df_summary = pd.DataFrame({
        '模型': results['model'],
        '准确率': results['accuracy'],
        '精确率(宏平均)': results['precision'],
        '召回率(宏平均)': results['recall'],
        'F1分数(宏平均)': results['f1'],
        '平均推理时间(ms)': [t * 1000 for t in results['avg_inference_time']],
        '样本数量': results['total_samples']
    })
    
    # 生成详细的类别指标
    class_metrics = []
    for i in range(len(results['model'])):
        model_name = results['model'][i]
        
        # 每个类别的详细指标
        for j, class_name in enumerate(INTENT_CLASSES):
            class_metrics.append({
                '模型': model_name,
                '意图类别': class_name,
                '精确率': results['class_precisions'][i][j],
                '召回率': results['class_recalls'][i][j],
                'F1分数': results['class_f1s'][i][j]
            })
    
    df_class_metrics = pd.DataFrame(class_metrics)
    
    # 时间戳作为文件名的一部分
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 将结果保存到Excel文件，使用多个工作表
    excel_path = os.path.join(output_dir, f'model_evaluation_{timestamp}.xlsx')
    with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
        df_summary.to_excel(writer, sheet_name='总体性能', index=False)
        df_class_metrics.to_excel(writer, sheet_name='类别详细指标', index=False)
        
        # 为每个模型创建推理时间分布工作表
        for i, model in enumerate(results['model']):
            inference_times = np.array(results['inference_times'][i]) * 1000  # 转换为毫秒
            df_times = pd.DataFrame({
                '推理时间(ms)': inference_times
            })
            df_times.to_excel(writer, sheet_name=f'{model}_推理时间', index=False)
    
    print(f"\n评估结果已保存到: {excel_path}")
    
    # 可视化
    generate_visualizations(results, output_dir, timestamp)
    
    return excel_path

def evaluate_fast_model(inference_engine, data_loader):
    """评估快速分类器模型性能"""
    device = inference_engine.device
    all_preds = []
    all_labels = []
    inference_times = []
    
    for features, labels in tqdm(data_loader):
        features = features.to(device)
        
        # 记录每个样本的推理时间
        batch_times = []
        batch_preds = []
        
        # 对每个样本进行单独处理
        for i in range(features.size(0)):
            feature = features[i:i+1]
            
            # 计时开始
            start_time = time.time()
            
            # 使用模型预测
            with torch.no_grad():
                prediction, _ = inference_engine.fast_model.predict(feature)
            
            # 计时结束
            end_time = time.time()
            inference_time = end_time - start_time
            
            # 记录结果
            batch_times.append(inference_time)
            batch_preds.append(prediction.item())
            
        all_preds.extend(batch_preds)
        all_labels.extend(labels.numpy())
        inference_times.extend(batch_times)
    
    # 计算性能指标
    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='macro')
    
    # 每个类别的指标
    class_precision, class_recall, class_f1, _ = precision_recall_fscore_support(all_labels, all_preds, 
                                                                           average=None, 
                                                                           labels=range(len(INTENT_CLASSES)))
    
    avg_inference_time = np.mean(inference_times)
    
    print(f"快速分类器评估结果:")
    print(f"准确率: {accuracy:.4f}")
    print(f"精确率(宏平均): {precision:.4f}")
    print(f"召回率(宏平均): {recall:.4f}")
    print(f"F1分数(宏平均): {f1:.4f}")
    print(f"平均推理时间: {avg_inference_time*1000:.2f}ms")
    print("\n分类报告:")
    print(classification_report(all_labels, all_preds, target_names=INTENT_CLASSES))
    
    # 返回评估结果
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'avg_inference_time': avg_inference_time,
        'total_samples': len(all_labels),
        'inference_times': inference_times,
        'class_precision': class_precision,
        'class_recall': class_recall,
        'class_f1': class_f1
    }

def evaluate_precise_model(inference_engine, data_loader):
    """评估精确分类器模型性能"""
    device = inference_engine.device
    all_preds = []
    all_labels = []
    inference_times = []
    
    for batch in tqdm(data_loader):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)
        
        # 记录每个样本的推理时间
        batch_times = []
        batch_preds = []
        
        # 对每个样本进行单独处理
        for i in range(input_ids.size(0)):
            input_id = input_ids[i:i+1]
            mask = attention_mask[i:i+1]
            
            # 计时开始
            start_time = time.time()
            
            # 使用模型预测
            with torch.no_grad():
                prediction, _ = inference_engine.precise_model.predict(input_id, mask)
            
            # 计时结束
            end_time = time.time()
            inference_time = end_time - start_time
            
            # 记录结果
            batch_times.append(inference_time)
            batch_preds.append(prediction.item())
            
        all_preds.extend(batch_preds)
        all_labels.extend(labels.cpu().numpy())
        inference_times.extend(batch_times)
    
    # 计算性能指标
    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='macro')
    
    # 每个类别的指标
    class_precision, class_recall, class_f1, _ = precision_recall_fscore_support(all_labels, all_preds, 
                                                                           average=None, 
                                                                           labels=range(len(INTENT_CLASSES)))
    
    avg_inference_time = np.mean(inference_times)
    
    print(f"精确分类器评估结果:")
    print(f"准确率: {accuracy:.4f}")
    print(f"精确率(宏平均): {precision:.4f}")
    print(f"召回率(宏平均): {recall:.4f}")
    print(f"F1分数(宏平均): {f1:.4f}")
    print(f"平均推理时间: {avg_inference_time*1000:.2f}ms")
    print("\n分类报告:")
    print(classification_report(all_labels, all_preds, target_names=INTENT_CLASSES))
    
    # 返回评估结果
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'avg_inference_time': avg_inference_time,
        'total_samples': len(all_labels),
        'inference_times': inference_times,
        'class_precision': class_precision,
        'class_recall': class_recall,
        'class_f1': class_f1
    }

def evaluate_full_pipeline(inference_engine, data_loader, confidence_threshold=FAST_CONFIDENCE_THRESHOLD):
    """评估完整推理流程（快速+精确模型协同）"""
    device = inference_engine.device
    all_preds = []
    all_labels = []
    inference_times = []
    path_counts = {'fast': 0, 'precise': 0}
    
    for features, labels in tqdm(data_loader):
        features = features.to(device)
        
        # 记录每个样本的推理时间和路径
        batch_times = []
        batch_preds = []
        
        # 对每个样本进行单独处理
        for i in range(features.size(0)):
            feature = features[i:i+1]
            
            # 计时开始
            start_time = time.time()
            
            # 1. 首先通过快速模型
            with torch.no_grad():
                fast_prediction, fast_confidence = inference_engine.fast_model.predict(feature)
            
            # 如果快速模型置信度高，使用它的预测
            if fast_confidence.item() >= confidence_threshold:
                prediction = fast_prediction.item()
                path_counts['fast'] += 1
            else:
                # 否则，我们需要通过精确模型
                # 由于这里没有文本，我们模拟ASR获取文本的过程（简化）
                # 在实际中，这里应该有一个ASR步骤获取文本
                # 我们这里简单地使用类别名称作为"ASR文本"
                asr_text = INTENT_CLASSES[labels[i]]
                
                encoding = inference_engine.tokenizer(
                    asr_text,
                    max_length=128,
                    padding='max_length',
                    truncation=True,
                    return_tensors='pt'
                )
                
                input_ids = encoding['input_ids'].to(device)
                attention_mask = encoding['attention_mask'].to(device)
                
                with torch.no_grad():
                    precise_prediction, _ = inference_engine.precise_model.predict(input_ids, attention_mask)
                
                prediction = precise_prediction.item()
                path_counts['precise'] += 1
            
            # 计时结束
            end_time = time.time()
            inference_time = end_time - start_time
            
            # 记录结果
            batch_times.append(inference_time)
            batch_preds.append(prediction)
            
        all_preds.extend(batch_preds)
        all_labels.extend(labels.numpy())
        inference_times.extend(batch_times)
    
    # 计算性能指标
    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='macro')
    
    # 每个类别的指标
    class_precision, class_recall, class_f1, _ = precision_recall_fscore_support(all_labels, all_preds, 
                                                                           average=None, 
                                                                           labels=range(len(INTENT_CLASSES)))
    
    avg_inference_time = np.mean(inference_times)
    
    print(f"完整推理流程评估结果:")
    print(f"准确率: {accuracy:.4f}")
    print(f"精确率(宏平均): {precision:.4f}")
    print(f"召回率(宏平均): {recall:.4f}")
    print(f"F1分数(宏平均): {f1:.4f}")
    print(f"平均推理时间: {avg_inference_time*1000:.2f}ms")
    print(f"路径使用情况: 快速={path_counts['fast']}次 ({path_counts['fast']/len(all_labels)*100:.1f}%), "
          f"精确={path_counts['precise']}次 ({path_counts['precise']/len(all_labels)*100:.1f}%)")
    print("\n分类报告:")
    print(classification_report(all_labels, all_preds, target_names=INTENT_CLASSES))
    
    # 返回评估结果
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'avg_inference_time': avg_inference_time,
        'total_samples': len(all_labels),
        'inference_times': inference_times,
        'class_precision': class_precision,
        'class_recall': class_recall,
        'class_f1': class_f1
    }

def generate_visualizations(results, output_dir, timestamp):
    """生成可视化图表"""
    # 1. 准确率比较柱状图
    plt.figure(figsize=(10, 6))
    plt.bar(results['model'], results['accuracy'], color=['blue', 'green', 'orange'][:len(results['model'])])
    plt.title('模型准确率比较')
    plt.ylabel('准确率')
    plt.ylim(0, 1)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig(os.path.join(output_dir, f'accuracy_comparison_{timestamp}.png'))
    
    # 2. 推理时间比较柱状图
    plt.figure(figsize=(10, 6))
    inference_times_ms = [t * 1000 for t in results['avg_inference_time']]
    plt.bar(results['model'], inference_times_ms, color=['blue', 'green', 'orange'][:len(results['model'])])
    plt.title('平均推理时间比较')
    plt.ylabel('推理时间 (ms)')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig(os.path.join(output_dir, f'inference_time_comparison_{timestamp}.png'))
    
    # 3. 每个模型的推理时间分布直方图
    for i, model in enumerate(results['model']):
        plt.figure(figsize=(10, 6))
        inference_times = np.array(results['inference_times'][i]) * 1000  # 转换为毫秒
        plt.hist(inference_times, bins=30, alpha=0.7, color='blue')
        plt.title(f'{model} 推理时间分布')
        plt.xlabel('推理时间 (ms)')
        plt.ylabel('频次')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.savefig(os.path.join(output_dir, f'{model}_inference_time_hist_{timestamp}.png'))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='评估语音意图识别模型')
    parser.add_argument('--data_dir', type=str, default=DATA_DIR, help='数据目录')
    parser.add_argument('--annotation_file', type=str, required=True, help='注释文件路径')
    parser.add_argument('--fast_model', type=str, required=True, help='一级快速分类器路径')
    parser.add_argument('--precise_model', type=str, help='二级精确分类器路径(可选)')
    parser.add_argument('--output_dir', type=str, default='evaluation_results', help='输出目录')
    args = parser.parse_args()
    
    # 评估模型并保存结果
    evaluate_models(
        args.data_dir,
        args.annotation_file,
        args.fast_model,
        args.precise_model,
        args.output_dir
    ) 