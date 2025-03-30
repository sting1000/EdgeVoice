#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import argparse
import time
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import seaborn as sns
import pandas as pd
import librosa
import pickle
import torch.nn.functional as F
from collections import defaultdict, Counter

from config import *
from models.streaming_conformer import StreamingConformer
from streaming_dataset import StreamingAudioDataset, prepare_streaming_dataloader, collate_full_batch
from utils.feature_augmentation import mixup_features, apply_augmentations
from utils.feature_extraction import streaming_feature_extractor, extract_mfcc_features, extract_streaming_features

def set_seed(seed=42):
    """设置随机种子以确保可重现性"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

class LabelSmoothingCrossEntropy(nn.Module):
    """标签平滑交叉熵损失"""
    def __init__(self, smoothing=0.1):
        super().__init__()
        self.smoothing = smoothing
        
    def forward(self, pred, target):
        n_classes = pred.size(1)
        with torch.no_grad():
            # 创建平滑标签
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (n_classes - 1))
            true_dist.scatter_(1, target.unsqueeze(1), 1.0 - self.smoothing)
        
        return torch.mean(torch.sum(-true_dist * nn.functional.log_softmax(pred, dim=1), dim=1))

def prepare_data_loaders(annotation_file, data_dir=DATA_DIR, batch_size=32, 
                          val_split=0.1, seed=42):
    """准备训练和验证数据加载器
    
    Args:
        annotation_file: 标注文件路径
        data_dir: 数据目录
        batch_size: 批大小
        val_split: 验证集比例
        seed: 随机种子
    
    Returns:
        train_loader: 训练数据加载器
        val_loader: 验证数据加载器 
        intent_labels: 意图标签列表
    """
    # 加载完整数据集
    full_dataset = StreamingAudioDataset(
        annotation_file=annotation_file,
        data_dir=data_dir,
        streaming_mode=False,  # 完整模式
        use_random_crop=False,
        use_feature_augmentation=False
    )
    
    # 获取意图标签
    intent_labels = full_dataset.intent_labels
    
    # 分割训练集和验证集
    val_size = int(len(full_dataset) * val_split)
    train_size = len(full_dataset) - val_size
    
    set_seed(seed)
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    print(f"数据集分割 - 训练集: {train_size}, 验证集: {val_size}")
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        collate_fn=collate_full_batch,
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        collate_fn=collate_full_batch,
        num_workers=4,
        pin_memory=True
    )
    
    return train_loader, val_loader, intent_labels

def train_epoch(model, dataloader, optimizer, criterion, device=DEVICE, 
                seq_length=None, use_mixup=USE_MIXUP, mixup_alpha=MIXUP_ALPHA):
    """训练一个epoch
    
    Args:
        model: 模型
        dataloader: 数据加载器
        optimizer: 优化器
        criterion: 损失函数
        device: 设备
        seq_length: 序列长度截断（用于渐进式训练）
        use_mixup: 是否使用MixUp
        mixup_alpha: MixUp参数
    
    Returns:
        epoch_loss: 平均损失
        epoch_acc: 准确率
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    # 使用tqdm进度条
    progress_bar = tqdm(dataloader, desc="训练中")
    
    for batch in progress_bar:
        # 处理batch数据，现在batch是(features, labels)形式
        features, labels = batch
        features = features.to(device)
        labels = labels.to(device)
        
        # 序列长度截断（渐进式训练）
        if seq_length is not None and features.size(1) > seq_length:
            # 随机选择序列的开始位置，避免总是使用相同的部分
            max_start = features.size(1) - seq_length
            start = random.randint(0, max_start) if max_start > 0 else 0
            features = features[:, start:start+seq_length, :]
        
        # 应用特征增强，使用更强的增强
        features = apply_augmentations(features, phase='train')
        
        # 初始化MixUp变量
        apply_mixup = use_mixup and random.random() < AUGMENT_PROB
        labels_a = labels
        labels_b = None
        lam = 1.0
        
        # MixUp数据增强
        if apply_mixup:
            features, labels_a, labels_b, lam = mixup_features(features, labels, alpha=mixup_alpha)
        
        # 清零梯度
        optimizer.zero_grad()
        
        # 前向传播
        outputs = model(features)
        
        # 计算损失
        if apply_mixup:
            # MixUp损失
            loss = lam * criterion(outputs, labels_a) + (1 - lam) * criterion(outputs, labels_b)
        else:
            loss = criterion(outputs, labels)
        
        # 反向传播和优化
        loss.backward()
        
        # 梯度裁剪，防止梯度爆炸
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=3.0)
        
        optimizer.step()
        
        # 计算准确率
        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        
        if apply_mixup:
            # MixUp下使用最高概率的标签计算准确率
            correct += (predicted == labels_a).sum().item() * lam + (predicted == labels_b).sum().item() * (1 - lam)
        else:
            correct += (predicted == labels).sum().item()
        
        # 更新进度条
        progress_bar.set_postfix({
            'loss': running_loss / (progress_bar.n + 1),
            'acc': 100 * correct / total
        })
    
    # 计算平均损失和准确率
    epoch_loss = running_loss / len(dataloader)
    epoch_acc = 100 * correct / total
    
    return epoch_loss, epoch_acc

def validate(model, dataloader, criterion, device=DEVICE, seq_length=None):
    """验证模型
    
    Args:
        model: 模型
        dataloader: 数据加载器
        criterion: 损失函数
        device: 设备
        seq_length: 序列长度截断
    
    Returns:
        val_loss: 验证损失
        val_acc: 验证准确率
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    all_preds = []
    all_labels = []
    all_confs = []  # 记录预测置信度
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="验证中"):
            # 处理batch数据，现在batch是(features, labels)形式
            features, labels = batch
            features = features.to(device)
            labels = labels.to(device)
            
            # 序列长度截断
            if seq_length is not None and features.size(1) > seq_length:
                # 使用中间部分而不是开始部分，捕获更多信息
                start = max(0, (features.size(1) - seq_length) // 2)
                features = features[:, start:start+seq_length, :]
            
            # 对验证集也进行轻微的特征增强，提高鲁棒性
            if random.random() < 0.3:  # 30%的概率应用轻微增强
                features = apply_augmentations(features, phase='val')
            
            # 前向传播
            outputs = model(features)
            
            # 计算损失
            loss = criterion(outputs, labels)
            
            # 统计
            running_loss += loss.item()
            probs = F.softmax(outputs, dim=1)
            conf, predicted = torch.max(probs, dim=1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # 收集用于计算指标
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_confs.extend(conf.cpu().numpy())
    
    # 计算平均损失和准确率
    val_loss = running_loss / len(dataloader)
    val_acc = 100 * correct / total
    
    # 计算按置信度加权的准确率
    conf_weighted_acc = 0
    total_conf = sum(all_confs)
    if total_conf > 0:
        for pred, label, conf in zip(all_preds, all_labels, all_confs):
            if pred == label:
                conf_weighted_acc += conf / total_conf
        conf_weighted_acc *= 100
    
    # 计算平均置信度
    avg_conf = sum(all_confs) / len(all_confs) if all_confs else 0
    
    print(f"平均预测置信度: {avg_conf:.4f}")
    print(f"置信度加权准确率: {conf_weighted_acc:.2f}%")
    
    return val_loss, val_acc, all_preds, all_labels

def evaluate_streaming_model(model, test_annotation_file, data_dir=DATA_DIR, 
                        confidence_threshold=STREAMING_DECISION_THRESHOLD, 
                        chunk_size=STREAMING_CHUNK_SIZE,
                        step_size=STREAMING_STEP_SIZE,
                        max_cached_frames=MAX_CACHED_FRAMES,
                        min_decision_frames=MIN_DECISION_FRAMES):
    """评估流式模型
    
    Args:
        model: 待评估模型
        test_annotation_file: 测试标注文件
        data_dir: 数据目录
        confidence_threshold: 置信度阈值，用于确定何时做出决策
        chunk_size: 每次处理的帧数
        step_size: 每次前进的步长
        max_cached_frames: 最大缓存帧数
        min_decision_frames: 最小决策所需帧数
        
    Returns:
        metrics: 包含各种评估指标的字典
    """
    # 加载测试数据集
    test_df = pd.read_csv(test_annotation_file)
    
    # 确保模型处于评估模式
    model.eval()
    model.to(DEVICE)
    
    # 初始化指标
    all_predictions = []
    all_labels = []
    all_confidences = []
    all_latencies = []
    streaming_metrics = defaultdict(list)
    
    # 跟踪每个类别的性能
    class_metrics = {cls: {"correct": 0, "total": 0, "confidence": []} for cls in INTENT_CLASSES}
    
    # 自适应决策参数
    adaptive_threshold = confidence_threshold
    smoothed_pred_counts = Counter()  # 连续预测计数
    
    # 处理每个测试样本
    for idx, row in test_df.iterrows():
        file_path = os.path.join(data_dir, row['file_path'])
        true_label = row['intent']
        true_label_idx = INTENT_CLASSES.index(true_label)
        
        # 加载完整音频用于特征提取
        try:
            audio, sr = librosa.load(file_path, sr=TARGET_SAMPLE_RATE)
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            continue
            
        # 获取完整特征序列，计算总帧数
        all_features = extract_mfcc_features(audio, sr=TARGET_SAMPLE_RATE)
        total_frames = len(all_features)
        
        # 模拟流式处理
        cached_states = None
        final_pred = None
        final_confidence = 0
        decision_made = False
        frames_processed = 0
        start_time = time.time()
        last_pred = None
        
        # 高置信度计数器和低置信度容忍度
        high_confidence_frames = 0
        low_confidence_tolerance = 0
        max_low_confidence_tolerance = 5  # 最多容忍5个低置信度帧
        
        # 连续预测跟踪
        consecutive_preds = []
        
        # 模拟每个步长的流式处理
        for i in range(0, total_frames, step_size):
            frames_processed += step_size
            
            # 提取当前chunk的特征
            end = min(i + chunk_size, total_frames)
            features = all_features[i:end]
            
            if len(features) < 2:  # 跳过极短特征
                continue
                
            # 转换为tensor
            features_tensor = torch.FloatTensor(features).unsqueeze(0).to(DEVICE)
            
            # 进行流式预测
            with torch.no_grad():
                pred, confidence, cached_states = model.predict_streaming(features_tensor, cached_states)
                
            # 记录预测
            pred_idx = pred.item()
            pred_class = INTENT_CLASSES[pred_idx]
            confidence_val = confidence.item()
            
            # 更新连续预测列表
            consecutive_preds.append((pred_idx, confidence_val))
            if len(consecutive_preds) > min_decision_frames:
                consecutive_preds.pop(0)
                
            # 置信度动态调整
            if len(consecutive_preds) >= min_decision_frames:
                # 分析连续预测的一致性
                pred_counter = Counter([p[0] for p in consecutive_preds])
                most_common_pred, count = pred_counter.most_common(1)[0]
                
                # 如果连续预测一致性高，降低置信度要求
                if count >= min_decision_frames * 0.7:  # 至少70%的连续帧预测相同
                    smoothed_pred_counts[most_common_pred] += 1
                    
                    # 根据之前的正确率动态调整阈值
                    pred_class = INTENT_CLASSES[most_common_pred]
                    if class_metrics[pred_class]["total"] > 0:
                        accuracy = class_metrics[pred_class]["correct"] / class_metrics[pred_class]["total"]
                        
                        # 对历史表现好的类别，适当降低阈值
                        if accuracy > 0.8:
                            dynamic_threshold = adaptive_threshold * 0.9
                        else:
                            dynamic_threshold = adaptive_threshold * 1.1
                    else:
                        dynamic_threshold = adaptive_threshold
                        
                    # 计算平均置信度
                    avg_confidence = sum([p[1] for p in consecutive_preds]) / len(consecutive_preds)
                    
                    # 判断是否满足决策条件
                    if avg_confidence >= dynamic_threshold and not decision_made:
                        final_pred = most_common_pred
                        final_confidence = avg_confidence
                        latency = time.time() - start_time
                        decision_made = True
                        all_latencies.append(latency)
                        break
            
            # 如果已处理了最大帧数仍未决策，则使用当前最佳预测
            if frames_processed >= max_cached_frames and not decision_made:
                # 选择置信度最高的预测
                best_pred_idx = max(consecutive_preds, key=lambda x: x[1])[0] if consecutive_preds else pred_idx
                final_pred = best_pred_idx
                final_confidence = confidence_val
                latency = time.time() - start_time
                decision_made = True
                all_latencies.append(latency)
                break
        
        # 如果处理完所有帧仍未做出决策，使用最后的预测
        if not decision_made:
            final_pred = pred_idx if 'pred_idx' in locals() else -1
            final_confidence = confidence_val if 'confidence_val' in locals() else 0
            latency = time.time() - start_time
            all_latencies.append(latency)
            
        # 记录预测结果
        all_predictions.append(final_pred)
        all_labels.append(true_label_idx)
        all_confidences.append(final_confidence)
        
        # 更新类别指标
        pred_class = INTENT_CLASSES[final_pred] if final_pred != -1 else "UNKNOWN"
        if pred_class in class_metrics:
            class_metrics[pred_class]["total"] += 1
            if final_pred == true_label_idx:
                class_metrics[pred_class]["correct"] += 1
            class_metrics[pred_class]["confidence"].append(final_confidence)
            
    # 计算整体指标
    correct = sum(1 for p, l in zip(all_predictions, all_labels) if p == l)
    accuracy = correct / len(all_labels) if all_labels else 0
    
    # 计算平均延迟
    avg_latency = sum(all_latencies) / len(all_latencies) if all_latencies else 0
    
    # 计算加权置信度准确率
    weighted_correct = sum(conf if pred == label else 0 
                          for pred, label, conf in zip(all_predictions, all_labels, all_confidences))
    weighted_accuracy = weighted_correct / sum(all_confidences) if sum(all_confidences) > 0 else 0
    
    # 计算平均置信度
    avg_confidence = sum(all_confidences) / len(all_confidences) if all_confidences else 0
    
    # 计算每个类别的指标
    class_report = {}
    for cls in INTENT_CLASSES:
        metrics = class_metrics[cls]
        cls_accuracy = metrics["correct"] / metrics["total"] if metrics["total"] > 0 else 0
        cls_avg_conf = sum(metrics["confidence"]) / len(metrics["confidence"]) if metrics["confidence"] else 0
        
        class_report[cls] = {
            "accuracy": cls_accuracy,
            "samples": metrics["total"],
            "avg_confidence": cls_avg_conf
        }
    
    # 计算混淆矩阵
    confusion = defaultdict(int)
    for pred, label in zip(all_predictions, all_labels):
        if pred != -1:  # 确保有效预测
            confusion[(INTENT_CLASSES[label], INTENT_CLASSES[pred])] += 1
    
    # 分析连续预测模式
    pred_patterns = {}
    for cls_idx, count in smoothed_pred_counts.items():
        cls_name = INTENT_CLASSES[cls_idx]
        pred_patterns[cls_name] = count
        
    # 汇总并返回指标
    metrics = {
        "accuracy": accuracy,
        "weighted_accuracy": weighted_accuracy,
        "avg_confidence": avg_confidence,
        "avg_latency": avg_latency,
        "class_report": class_report,
        "confusion_matrix": confusion,
        "prediction_patterns": pred_patterns
    }
    
    return metrics

def train_streaming_conformer(data_dir, annotation_file, model_save_path, 
                            num_epochs=30, batch_size=32, seed=42,
                            learning_rate=LEARNING_RATE, weight_decay=0.01,
                            use_mixup=USE_MIXUP, use_label_smoothing=USE_LABEL_SMOOTHING,
                            label_smoothing=LABEL_SMOOTHING,
                            progressive_training=PROGRESSIVE_TRAINING):
    """训练流式Conformer模型
    
    Args:
        data_dir: 数据目录
        annotation_file: 训练数据标注文件
        model_save_path: 模型保存路径
        num_epochs: 训练轮数
        batch_size: 批大小
        seed: 随机种子
        learning_rate: 学习率
        weight_decay: 权重衰减
        use_mixup: 是否使用MixUp
        use_label_smoothing: 是否使用标签平滑
        label_smoothing: 标签平滑系数
        progressive_training: 是否使用渐进式训练
    
    Returns:
        model: 训练好的模型
        intent_labels: 意图标签列表
    """
    # 设置随机种子
    set_seed(seed)
    
    # 准备数据加载器
    train_loader, val_loader, intent_labels = prepare_data_loaders(
        annotation_file=annotation_file,
        data_dir=data_dir,
        batch_size=batch_size,
        seed=seed
    )
    
    # 创建模型
    input_dim = N_MFCC * 3  # MFCC + delta + delta2
    model = StreamingConformer(
        input_dim=input_dim,
        hidden_dim=CONFORMER_HIDDEN_SIZE,
        num_classes=len(intent_labels),
        num_layers=CONFORMER_LAYERS,
        num_heads=CONFORMER_ATTENTION_HEADS,
        dropout=CONFORMER_DROPOUT,
        kernel_size=CONFORMER_CONV_KERNEL_SIZE,
        expansion_factor=CONFORMER_FF_EXPANSION_FACTOR
    )
    
    model = model.to(DEVICE)
    
    # 使用标签平滑的交叉熵损失
    if use_label_smoothing:
        criterion = LabelSmoothingCrossEntropy(smoothing=label_smoothing)
        print(f"使用标签平滑损失，平滑参数: {label_smoothing}")
    else:
        criterion = nn.CrossEntropyLoss()
        print("使用标准交叉熵损失")
    
    # 优化器 - 使用AdamW，更好的权重衰减管理
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=learning_rate, 
        weight_decay=weight_decay,
        betas=(0.9, 0.999),  # 标准beta
        eps=1e-8            # 数值稳定性
    )
    
    # 学习率调度器
    if USE_COSINE_SCHEDULER:
        # 余弦退火学习率调度，适合长训练
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, 
            T_max=num_epochs,  # 总共训练num_epochs轮
            eta_min=learning_rate * 0.01  # 最小学习率是初始的1%
        )
        print(f"使用余弦退火学习率调度，T_max={num_epochs}，eta_min={learning_rate * 0.01}")
    else:
        # 使用ReduceLROnPlateau调度器，根据验证表现调整学习率
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            mode='max',         # 监控验证准确率（最大化）
            factor=0.5,         # 减少学习率的因子
            patience=3,         # 3轮没提升就降低学习率
            verbose=True,       # 打印学习率变化
            min_lr=learning_rate * 0.001  # 最小学习率
        )
        print(f"使用ReduceLROnPlateau学习率调度，factor=0.5，patience=3")
    
    # 渐进式训练序列长度设置
    if progressive_training:
        # 从短到长的渐进式序列长度
        seq_lengths = [20, 30, 40, 60, 80, 100, None]  # None表示使用完整序列
        print(f"启用渐进式长度训练，序列长度进度: {seq_lengths}")
    else:
        seq_lengths = [None] * num_epochs  # 所有epoch都使用完整序列
        print("使用完整序列长度训练")
    
    # 打印模型和训练信息
    print("\n开始训练流式Conformer模型...")
    print(f"模型配置: 隐藏层={CONFORMER_HIDDEN_SIZE}, 层数={CONFORMER_LAYERS}, 头数={CONFORMER_ATTENTION_HEADS}")
    print(f"特征维度: {input_dim}, 卷积核大小: {CONFORMER_CONV_KERNEL_SIZE}")
    print(f"训练样本数: {len(train_loader.dataset)}, 验证样本数: {len(val_loader.dataset)}")
    print(f"总训练轮数: {num_epochs}")
    
    # 使用渐进式长度训练
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': [], 'lr': []}
    best_val_acc = 0
    patience = 0
    max_patience = EARLY_STOPPING_PATIENCE if USE_EARLY_STOPPING else float('inf')
    
    # 记录训练开始时间
    start_time = time.time()
    
    for epoch in range(num_epochs):
        # 设置当前序列长度
        seq_length = seq_lengths[min(epoch, len(seq_lengths)-1)]
        if seq_length is None:
            print(f"\nEpoch {epoch+1}/{num_epochs}")
            print(f"当前序列长度: 完整序列")
        else:
            print(f"\nEpoch {epoch+1}/{num_epochs}")
            print(f"当前序列长度: {seq_length}")
        
        # 显示当前学习率
        current_lr = optimizer.param_groups[0]['lr']
        print(f"当前学习率: {current_lr:.6f}")
        history['lr'].append(current_lr)
        
        # 训练一个epoch
        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, criterion, DEVICE, 
            seq_length=seq_length, use_mixup=use_mixup, mixup_alpha=MIXUP_ALPHA
        )
        
        # 验证
        val_loss, val_acc, val_preds, val_labels = validate(
            model, val_loader, criterion, DEVICE, seq_length=seq_length
        )
        
        # 记录历史
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        # 打印结果
        print(f"训练损失: {train_loss:.4f}, 训练准确率: {train_acc:.2f}%")
        print(f"验证损失: {val_loss:.4f}, 验证准确率: {val_acc:.2f}%")
        
        # 更新学习率调度器
        if USE_COSINE_SCHEDULER:
            scheduler.step()
        else:
            # ReduceLROnPlateau需要验证指标
            scheduler.step(val_acc)
        
        # 检查是否保存最佳模型
        if val_acc > best_val_acc:
            improvement = val_acc - best_val_acc
            print(f"验证准确率提升: {best_val_acc:.2f}% -> {val_acc:.2f}%")
            best_val_acc = val_acc
            patience = 0
            
            # 保存最佳模型
            checkpoint = {
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch,
                'val_acc': val_acc,
                'train_acc': train_acc,
                'intent_labels': intent_labels
            }
            
            os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
            torch.save(checkpoint, model_save_path)
            print(f"保存最佳模型到: {model_save_path}")
        else:
            patience += 1
            print(f"验证准确率未提升，当前耐心: {patience}/{max_patience}")
            
            if patience >= max_patience:
                print(f"早停: {max_patience}轮未提升，停止训练")
                break
    
    # 训练结束
    total_time = time.time() - start_time
    print(f"\n训练完成! 总时间: {total_time:.2f}秒")
    print(f"最佳验证准确率: {best_val_acc:.2f}%")
    
    # 绘制训练历史
    plot_training_history(history, os.path.dirname(model_save_path))
    
    # 加载最佳模型
    checkpoint = torch.load(model_save_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    return model, intent_labels

def plot_training_history(history, save_dir):
    """绘制训练历史
    
    Args:
        history: 训练历史字典
        save_dir: 保存目录
    """
    plt.figure(figsize=(15, 5))
    
    # 绘制损失
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='训练损失')
    plt.plot(history['val_loss'], label='验证损失')
    plt.xlabel('Epoch')
    plt.ylabel('损失')
    plt.title('训练和验证损失')
    plt.legend()
    plt.grid(True)
    
    # 绘制准确率
    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='训练准确率')
    plt.plot(history['val_acc'], label='验证准确率')
    plt.xlabel('Epoch')
    plt.ylabel('准确率(%)')
    plt.title('训练和验证准确率')
    plt.legend()
    plt.grid(True)
    
    # 保存图表
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'streaming_conformer_history.png'))
    plt.close()

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='流式Conformer模型训练和评估')
    
    # 数据参数
    parser.add_argument('--data_dir', type=str, default=DATA_DIR, help='数据目录')
    parser.add_argument('--annotation_file', type=str, required=True, help='训练数据标注文件')
    parser.add_argument('--test_annotation_file', type=str, help='测试数据标注文件（用于评估）')
    
    # 模型参数
    parser.add_argument('--model_type', type=str, default='streaming_conformer', help='模型类型')
    parser.add_argument('--model_save_path', type=str, required=True, help='模型保存路径')
    parser.add_argument('--hidden_dim', type=int, default=CONFORMER_HIDDEN_SIZE, help='隐藏层维度')
    parser.add_argument('--num_layers', type=int, default=CONFORMER_LAYERS, help='Conformer层数')
    parser.add_argument('--num_heads', type=int, default=CONFORMER_ATTENTION_HEADS, help='注意力头数')
    parser.add_argument('--kernel_size', type=int, default=CONFORMER_CONV_KERNEL_SIZE, help='卷积核大小')
    
    # 训练参数
    parser.add_argument('--num_epochs', type=int, default=NUM_EPOCHS, help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE, help='批大小')
    parser.add_argument('--learning_rate', type=float, default=LEARNING_RATE, help='学习率')
    parser.add_argument('--weight_decay', type=float, default=0.01, help='权重衰减')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    
    # 增强和训练策略
    parser.add_argument('--use_mixup', action='store_true', default=USE_MIXUP, help='是否使用MixUp增强')
    parser.add_argument('--use_label_smoothing', action='store_true', default=USE_LABEL_SMOOTHING, help='是否使用标签平滑')
    parser.add_argument('--label_smoothing', type=float, default=LABEL_SMOOTHING, help='标签平滑系数')
    parser.add_argument('--progressive_training', action='store_true', default=PROGRESSIVE_TRAINING, help='是否使用渐进式训练')
    parser.add_argument('--eval_interval', type=int, default=1, help='每多少个epoch评估一次')
    
    # 评估参数
    parser.add_argument('--evaluate', action='store_true', help='是否评估模型')
    parser.add_argument('--confidence_threshold', type=float, default=0.85, help='流式评估的置信度阈值')
    
    return parser.parse_args()

def main():
    """主函数"""
    # 解析命令行参数
    args = parse_args()
    
    # 配置
    print("\n=== 流式Conformer训练配置 ===")
    print(f"数据目录: {args.data_dir}")
    print(f"标注文件: {args.annotation_file}")
    print(f"模型保存路径: {args.model_save_path}")
    print(f"隐藏层维度: {args.hidden_dim}")
    print(f"Conformer层数: {args.num_layers}")
    print(f"注意力头数: {args.num_heads}")
    print(f"卷积核大小: {args.kernel_size}")
    print(f"训练轮数: {args.num_epochs}")
    print(f"批大小: {args.batch_size}")
    print(f"学习率: {args.learning_rate}")
    print(f"使用MixUp: {args.use_mixup}")
    print(f"标签平滑: {args.label_smoothing}")
    print(f"渐进式训练: {args.progressive_training}")
    
    # 训练模型
    model, intent_labels = train_streaming_conformer(
        data_dir=args.data_dir,
        annotation_file=args.annotation_file,
        model_save_path=args.model_save_path,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        seed=args.seed,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        use_mixup=args.use_mixup,
        use_label_smoothing=args.use_label_smoothing,
        label_smoothing=args.label_smoothing,
        progressive_training=args.progressive_training
    )
    
    # 评估模型
    if args.evaluate and args.test_annotation_file:
        print("\n=== 评估流式模型 ===")
        print(f"测试数据: {args.test_annotation_file}")
        print(f"置信度阈值: {args.confidence_threshold}")
        
        # 模型评估路径
        eval_save_path = os.path.join(os.path.dirname(args.model_save_path), "streaming_eval_results.pkl")
        
        # 流式评估
        metrics = evaluate_streaming_model(
            model=model,
            test_annotation_file=args.test_annotation_file,
            data_dir=args.data_dir,
            confidence_threshold=args.confidence_threshold
        )
        
        print("\n流式评估结果:")
        for metric, value in metrics.items():
            if isinstance(value, dict):
                print(f"{metric}:")
                for submetric, subvalue in value.items():
                    print(f"  {submetric}: {subvalue}")
            else:
                print(f"{metric}: {value}")
        
        # 保存结果
        if eval_save_path:
            with open(eval_save_path, 'wb') as f:
                pickle.dump(metrics, f)
    
    print("\n训练和评估完成!")

if __name__ == "__main__":
    main() 