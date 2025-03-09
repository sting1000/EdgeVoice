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
import librosa
import torch.nn.functional as F
import seaborn as sns

from config import *
from data_utils import prepare_dataloader, load_audio
from augmented_dataset import standardize_audio_length, prepare_augmented_dataloader
from models.fast_classifier import FastIntentClassifier
from models.precise_classifier import PreciseIntentClassifier
from inference import IntentInferenceEngine

def evaluate_models(data_dir, annotation_file, fast_model_path, precise_model_path=None, output_dir='evaluation_results', analyze_length_impact=False):
    """评估模型性能"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(output_dir, exist_ok=True)
    
    # 初始化推理引擎
    inference_engine = IntentInferenceEngine(
        fast_model_path=fast_model_path,
        precise_model_path=precise_model_path
    )
    
    # 定义特征提取函数
    def fast_feature_extractor(audio, sr, **kwargs):
        audio = standardize_audio_length(audio, sr)
        # 提取MFCC特征和动态特征
        mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
        delta_mfccs = librosa.feature.delta(mfccs)
        delta2_mfccs = librosa.feature.delta(mfccs, order=2)
        features = np.concatenate([mfccs, delta_mfccs, delta2_mfccs], axis=0)
        return features.T  # (time, features)格式，不添加上下文
    
    # 同时进行音频长度分析
    fast_loader, fast_labels = prepare_augmented_dataloader(
        annotation_file=annotation_file, 
        data_dir=data_dir, 
        feature_extractor=fast_feature_extractor,
        batch_size=32, 
        augment=False,  # 评估时不使用增强
        shuffle=False
    )
    
    def precise_feature_extractor(audio, sr, transcript=None, **kwargs):
        # 对于精确分类器，我们需要文本特征
        # 在实际使用中，这里应该使用ASR获取文本
        # 但在评估中，我们使用标注的文本
        if transcript is None:
            transcript = "评估模式默认文本"
            
        # 使用tokenizer处理文本
        encoding = inference_engine.tokenizer(
            transcript,
            max_length=128,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0)
        }
    
    # 准备精确分类器的数据加载器
    if precise_model_path:
        precise_loader, precise_labels = prepare_augmented_dataloader(
            annotation_file=annotation_file, 
            data_dir=data_dir, 
            feature_extractor=precise_feature_extractor,
            batch_size=32, 
            augment=False,
            shuffle=False
        )
    
    print("准备测试数据...")
    
    # 存储评估结果
    all_results = []
    
    # 评估快速分类器
    print("\n开始评估快速分类器...")
    fast_results = evaluate_fast_model(inference_engine, fast_loader)
    all_results.append(fast_results)
    
    # 打印快速分类器结果
    print(f"\n快速分类器评估结果:")
    print(f"准确率: {fast_results['accuracy']:.4f}")
    print(f"精确率: {fast_results['precision']:.4f}")
    print(f"召回率: {fast_results['recall']:.4f}")
    print(f"F1分数: {fast_results['f1']:.4f}")
    print(f"平均推理时间: {fast_results['avg_inference_time_ms']:.2f}ms")
    print("\n分类报告:")
    print(fast_results['class_report'])
    
    # 评估精确分类器(如果有)
    if precise_model_path:
        print("\n开始评估精确分类器...")
        precise_results = evaluate_precise_model(inference_engine, precise_loader)
        all_results.append(precise_results)
        
        # 打印精确分类器结果
        print(f"\n精确分类器评估结果:")
        print(f"准确率: {precise_results['accuracy']:.4f}")
        print(f"精确率: {precise_results['precision']:.4f}")
        print(f"召回率: {precise_results['recall']:.4f}")
        print(f"F1分数: {precise_results['f1']:.4f}")
        print(f"平均推理时间: {precise_results['avg_inference_time_ms']:.2f}ms")
        print("\n分类报告:")
        print(precise_results['class_report'])
        
        # 评估完整流水线
        print("\n开始评估完整推理流水线...")
        pipeline_results = evaluate_full_pipeline(inference_engine, fast_loader)
        all_results.append(pipeline_results)
        
        # 打印流水线结果
        print(f"\n完整推理流水线评估结果:")
        print(f"准确率: {pipeline_results['accuracy']:.4f}")
        print(f"精确率: {pipeline_results['precision']:.4f}")
        print(f"召回率: {pipeline_results['recall']:.4f}")
        print(f"F1分数: {pipeline_results['f1']:.4f}")
        print(f"平均推理时间: {pipeline_results['avg_inference_time_ms']:.2f}ms")
        print(f"快速分类器使用率: {pipeline_results['fast_ratio']*100:.1f}%")
        print("\n分类报告:")
        print(pipeline_results['class_report'])
    
    # 分析音频长度对性能的影响
    length_impact_results = None
    if analyze_length_impact:
        print("\n分析音频长度对性能的影响...")
        length_impact_results = analyze_audio_length_impact(inference_engine, data_dir, annotation_file)
    
    # 生成可视化
    print("\n生成评估结果可视化...")
    generate_visualizations(all_results, output_dir, timestamp, length_impact_results)
    
    print(f"\n评估完成。结果保存在 {output_dir} 目录。")
    
    return all_results, length_impact_results

def evaluate_fast_model(inference_engine, data_loader):
    """评估快速分类器模型的性能"""
    all_predictions = []
    all_confidences = []
    all_labels = []
    all_inference_times = []

    loop = tqdm(data_loader, desc="评估快速分类器")
    for batch_idx, batch in enumerate(loop):
        features, labels = batch['features'], batch['label']
        
        for feature, label in zip(features, labels):
            # 单样本推理
            feature_np = feature.numpy() if not feature.is_cuda else feature.cpu().numpy()
            
            start_time = time.time()
            with torch.no_grad():
                predicted_class, confidence = inference_engine.fast_model.predict(
                    torch.FloatTensor(feature_np).unsqueeze(0).to(inference_engine.device)
                )
            inference_time = time.time() - start_time
            
            # 收集结果
            all_predictions.append(predicted_class.item())
            all_confidences.append(confidence.item())
            all_labels.append(label.item() if torch.is_tensor(label) else label)
            all_inference_times.append(inference_time)

    # 计算指标
    accuracy = accuracy_score(all_labels, all_predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_predictions, average='weighted'
    )
    avg_inference_time = np.mean(all_inference_times) * 1000  # 转换为毫秒
    
    # 生成更详细的报告
    class_report = classification_report(
        all_labels, all_predictions, 
        target_names=INTENT_CLASSES, 
        zero_division=0
    )
    
    cm = confusion_matrix(all_labels, all_predictions)
    
    results = {
        'model_type': 'fast',
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'avg_inference_time_ms': avg_inference_time,
        'class_report': class_report,
        'confusion_matrix': cm,
        'labels': INTENT_CLASSES,
        'all_predictions': all_predictions,
        'all_confidences': all_confidences,
        'all_labels': all_labels,
        'all_inference_times': all_inference_times
    }
    
    return results

def evaluate_precise_model(inference_engine, data_loader):
    """评估精确分类器模型的性能"""
    all_predictions = []
    all_confidences = []
    all_labels = []
    all_inference_times = []

    loop = tqdm(data_loader, desc="评估精确分类器")
    for batch_idx, batch in enumerate(loop):
        features = batch['features']
        labels = batch['label']
        
        for i in range(len(labels)):
            input_ids = features['input_ids'][i].unsqueeze(0).to(inference_engine.device)
            attention_mask = features['attention_mask'][i].unsqueeze(0).to(inference_engine.device)
            label = labels[i]
            
            # 单样本推理
            start_time = time.time()
            with torch.no_grad():
                outputs = inference_engine.precise_model(input_ids, attention_mask)
                logits = outputs.logits
                probs = F.softmax(logits, dim=1)
                confidence, predicted = torch.max(probs, dim=1)
            inference_time = time.time() - start_time
            
            # 收集结果
            all_predictions.append(predicted.item())
            all_confidences.append(confidence.item())
            all_labels.append(label.item() if torch.is_tensor(label) else label)
            all_inference_times.append(inference_time)

    # 计算指标
    accuracy = accuracy_score(all_labels, all_predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_predictions, average='weighted'
    )
    avg_inference_time = np.mean(all_inference_times) * 1000  # 转换为毫秒
    
    # 生成更详细的报告
    class_report = classification_report(
        all_labels, all_predictions, 
        target_names=INTENT_CLASSES, 
        zero_division=0
    )
    
    cm = confusion_matrix(all_labels, all_predictions)
    
    results = {
        'model_type': 'precise',
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'avg_inference_time_ms': avg_inference_time,
        'class_report': class_report,
        'confusion_matrix': cm,
        'labels': INTENT_CLASSES,
        'all_predictions': all_predictions,
        'all_confidences': all_confidences,
        'all_labels': all_labels,
        'all_inference_times': all_inference_times
    }
    
    return results

def evaluate_full_pipeline(inference_engine, data_loader, confidence_threshold=FAST_CONFIDENCE_THRESHOLD):
    """评估完整推理流水线的性能（一级快速+二级精确分类器）"""
    all_predictions = []
    all_confidences = []
    all_labels = []
    all_inference_times = []
    all_is_fast = []  # 记录是否使用了快速分类器

    # 使用传入的阈值覆盖引擎默认值
    original_threshold = inference_engine.fast_confidence_threshold
    inference_engine.fast_confidence_threshold = confidence_threshold
    
    loop = tqdm(data_loader, desc="评估完整流水线")
    for batch_idx, batch in enumerate(loop):
        features = batch['features']
        labels = batch['label']
        
        for i in range(len(labels)):
            # 准备特征
            if isinstance(features, torch.Tensor):
                # 对于快速分类器的特征
                feature = features[i]
                feature_np = feature.cpu().numpy() if feature.is_cuda else feature.numpy()
                feature_tensor = torch.FloatTensor(feature_np).unsqueeze(0).to(inference_engine.device)
                label = labels[i]
                transcription = None
            elif isinstance(features, dict):
                # 对于精确分类器的特征
                feature_tensor = {
                    'input_ids': features['input_ids'][i].unsqueeze(0).to(inference_engine.device),
                    'attention_mask': features['attention_mask'][i].unsqueeze(0).to(inference_engine.device)
                }
                label = labels[i]
                transcription = batch.get('transcription', [None])[i]
            else:
                print(f"不支持的特征类型: {type(features)}")
                continue
            
            # 单样本推理
            start_time = time.time()
            with torch.no_grad():
                # 使用predict方法直接处理特征
                predicted_class, confidence, is_fast = inference_engine.predict(feature_tensor)
            inference_time = time.time() - start_time
            
            # 收集结果
            all_predictions.append(predicted_class)
            all_confidences.append(confidence)
            all_labels.append(label.item() if torch.is_tensor(label) else label)
            all_inference_times.append(inference_time)
            all_is_fast.append(is_fast)

    # 恢复原始阈值
    inference_engine.fast_confidence_threshold = original_threshold
    
    # 计算指标
    accuracy = accuracy_score(all_labels, all_predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_predictions, average='weighted'
    )
    avg_inference_time = np.mean(all_inference_times) * 1000  # 转换为毫秒
    fast_ratio = sum(all_is_fast) / len(all_is_fast) if all_is_fast else 0
    
    # 生成更详细的报告
    class_report = classification_report(
        all_labels, all_predictions, 
        target_names=INTENT_CLASSES, 
        zero_division=0
    )
    
    cm = confusion_matrix(all_labels, all_predictions)
    
    results = {
        'model_type': 'pipeline',
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'avg_inference_time_ms': avg_inference_time,
        'class_report': class_report,
        'confusion_matrix': cm,
        'labels': INTENT_CLASSES,
        'all_predictions': all_predictions,
        'all_confidences': all_confidences,
        'all_labels': all_labels,
        'all_inference_times': all_inference_times,
        'all_is_fast': all_is_fast,
        'fast_ratio': fast_ratio
    }
    
    return results

def analyze_audio_length_impact(inference_engine, data_dir, annotation_file):
    """分析音频长度对模型性能的影响"""
    print("加载数据集信息...")
    # 读取注释文件
    annotations = pd.read_csv(annotation_file)
    
    # 长度分组定义
    length_bins = {
        '短(<=1s)': {'samples': [], 'labels': [], 'predictions': [], 'times': []},
        '中(1-3s)': {'samples': [], 'labels': [], 'predictions': [], 'times': []},
        '长(3-5s)': {'samples': [], 'labels': [], 'predictions': [], 'times': []},
        '超长(>5s)': {'samples': [], 'labels': [], 'predictions': [], 'times': []}
    }
    
    # 获取类别到索引的映射
    class_to_idx = {cls: i for i, cls in enumerate(INTENT_CLASSES)}
    
    print("分析不同长度的音频样本...")
    # 遍历数据集
    for idx in tqdm(range(len(annotations))):
        try:
            # 获取音频文件路径和标签
            audio_path = os.path.join(data_dir, annotations.iloc[idx]['file_path'])
            intent_label = annotations.iloc[idx]['intent']
            label_idx = class_to_idx[intent_label]
            
            # 加载音频文件
            audio, sr = load_audio(audio_path)
            duration = len(audio) / sr
            
            # 决定音频属于哪个长度组
            if duration <= 1.0:
                group = '短(<=1s)'
            elif duration <= 3.0:
                group = '中(1-3s)'
            elif duration <= 5.0:
                group = '长(3-5s)'
            else:
                group = '超长(>5s)'
            
            # 预处理音频（可选，取决于是否想评估原始音频还是预处理后的音频）
            audio = standardize_audio_length(audio, sr)
            
            # 提取特征并预测
            start_time = time.time()
            features = inference_engine.preprocessor.process(audio)
            features = inference_engine.feature_extractor.extract_features(features)
            features_tensor = torch.FloatTensor(features).unsqueeze(0).to(inference_engine.device)
            
            with torch.no_grad():
                prediction, confidence = inference_engine.fast_model.predict(features_tensor)
            
            inference_time = time.time() - start_time
            
            # 记录结果
            length_bins[group]['samples'].append(idx)
            length_bins[group]['labels'].append(label_idx)
            length_bins[group]['predictions'].append(prediction.item())
            length_bins[group]['times'].append(inference_time)
        
        except Exception as e:
            print(f"处理样本时出错 {idx}: {e}")
    
    # 计算每个长度组的性能指标
    results = []
    
    for group, data in length_bins.items():
        if len(data['samples']) > 0:
            accuracy = accuracy_score(data['labels'], data['predictions'])
            precision, recall, f1, _ = precision_recall_fscore_support(
                data['labels'], data['predictions'], average='macro'
            )
            avg_time = np.mean(data['times']) * 1000  # 转换为毫秒
            
            results.append({
                '长度组': group,
                '样本数量': len(data['samples']),
                '准确率': accuracy,
                '精确率(宏平均)': precision,
                '召回率(宏平均)': recall,
                'F1分数(宏平均)': f1,
                '平均推理时间(ms)': avg_time
            })
            
            print(f"\n{group} 组评估结果 (样本数: {len(data['samples'])}):")
            print(f"准确率: {accuracy:.4f}")
            print(f"精确率(宏平均): {precision:.4f}")
            print(f"召回率(宏平均): {recall:.4f}")
            print(f"F1分数(宏平均): {f1:.4f}")
            print(f"平均推理时间: {avg_time:.2f}ms")
    
    return results

def generate_visualizations(results, output_dir, timestamp, length_impact_results=None):
    """生成评估结果的可视化"""
    os.makedirs(output_dir, exist_ok=True)
    
    # 创建所有模型的准确率对比图
    plt.figure(figsize=(10, 6))
    models = [result['model_type'] for result in results]
    accuracies = [result['accuracy'] for result in results]
    
    plt.bar(models, accuracies, color=['blue', 'green', 'red'][:len(models)])
    plt.title('模型准确率对比')
    plt.xlabel('模型')
    plt.ylabel('准确率')
    plt.ylim(0, 1)
    
    for i, v in enumerate(accuracies):
        plt.text(i, v + 0.02, f'{v:.4f}', ha='center')
    
    plt.savefig(os.path.join(output_dir, f'{timestamp}_accuracy_comparison.png'))
    
    # 创建推理时间对比图
    plt.figure(figsize=(10, 6))
    inference_times = [result['avg_inference_time_ms'] for result in results]
    
    plt.bar(models, inference_times, color=['blue', 'green', 'red'][:len(models)])
    plt.title('模型推理时间对比 (ms)')
    plt.xlabel('模型')
    plt.ylabel('平均推理时间 (ms)')
    
    for i, v in enumerate(inference_times):
        plt.text(i, v + 0.5, f'{v:.2f}ms', ha='center')
    
    plt.savefig(os.path.join(output_dir, f'{timestamp}_inference_time_comparison.png'))
    
    # 为每个模型生成混淆矩阵图
    for result in results:
        plt.figure(figsize=(10, 8))
        cm = result['confusion_matrix']
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=INTENT_CLASSES,
                   yticklabels=INTENT_CLASSES)
        
        plt.title(f'{result["model_type"]} 混淆矩阵')
        plt.xlabel('预测类别')
        plt.ylabel('真实类别')
        plt.tight_layout()
        
        plt.savefig(os.path.join(output_dir, f'{timestamp}_{result["model_type"]}_confusion_matrix.png'))
    
    # 为每个模型生成推理时间分布图
    for result in results:
        plt.figure(figsize=(10, 6))
        inference_times = np.array(result['all_inference_times']) * 1000  # 转换为毫秒
        plt.hist(inference_times, bins=30, alpha=0.7, color='blue')
        plt.title(f'{result["model_type"]} 推理时间分布')
        plt.xlabel('推理时间 (ms)')
        plt.ylabel('样本数')
        plt.savefig(os.path.join(output_dir, f'{timestamp}_{result["model_type"]}_inference_time_distribution.png'))
    
    # 如果有pipeline结果，还需要展示二级模型的使用比例
    for result in results:
        if result['model_type'] == 'pipeline' and 'all_is_fast' in result:
            plt.figure(figsize=(8, 8))
            fast_count = sum(result['all_is_fast'])
            precise_count = len(result['all_is_fast']) - fast_count
            
            plt.pie([fast_count, precise_count], 
                   labels=['快速模型', '精确模型'],
                   autopct='%1.1f%%',
                   colors=['lightblue', 'lightgreen'])
            
            plt.title('推理路径分布')
            plt.savefig(os.path.join(output_dir, f'{timestamp}_pipeline_path_distribution.png'))
    
    # 如果有音频长度影响分析结果
    if length_impact_results:
        plt.figure(figsize=(12, 6))
        durations = length_impact_results['durations']
        accuracies = length_impact_results['accuracies']
        times = length_impact_results['times']
        
        # 创建双y轴图
        fig, ax1 = plt.subplots(figsize=(12, 6))
        
        color = 'tab:blue'
        ax1.set_xlabel('音频长度 (秒)')
        ax1.set_ylabel('准确率', color=color)
        ax1.plot(durations, accuracies, color=color, marker='o')
        ax1.tick_params(axis='y', labelcolor=color)
        
        ax2 = ax1.twinx()
        color = 'tab:red'
        ax2.set_ylabel('推理时间 (ms)', color=color)
        ax2.plot(durations, times, color=color, marker='x')
        ax2.tick_params(axis='y', labelcolor=color)
        
        plt.title('音频长度对准确率和推理时间的影响')
        fig.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{timestamp}_length_impact.png'))
        
    plt.close('all')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='评估语音意图识别模型')
    parser.add_argument('--data_dir', type=str, default=DATA_DIR, help='数据目录')
    parser.add_argument('--annotation_file', type=str, required=True, help='注释文件路径')
    parser.add_argument('--fast_model', type=str, required=True, help='一级快速分类器路径')
    parser.add_argument('--precise_model', type=str, help='二级精确分类器路径(可选)')
    parser.add_argument('--output_dir', type=str, default='evaluation_results', help='输出目录')
    parser.add_argument('--analyze_length', action='store_true', help='分析音频长度对性能的影响')
    args = parser.parse_args()
    
    # 评估模型并保存结果
    evaluate_models(
        args.data_dir,
        args.annotation_file,
        args.fast_model,
        args.precise_model,
        args.output_dir,
        args.analyze_length
    ) 