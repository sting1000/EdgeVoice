#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import argparse
import time
import random
import subprocess
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
import itertools
import copy
import traceback


from config import *
from models.streaming_conformer import StreamingConformer
from streaming_dataset import StreamingAudioDataset, prepare_streaming_dataloader, collate_full_batch
from utils.feature_augmentation import mixup_features, apply_augmentations

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

class FocalLoss(nn.Module):
    """Focal Loss Function"""
    def __init__(self, alpha=[], gamma=[]):
        super().__init__()
        self.alpha = torch.tensor(alpha,  dtype=torch.float32) # 正负样本比例调节参数
        self.gamma = torch.tensor(gamma,  dtype=torch.float32) # 聚焦参数gamma
        self.CrossEntropyLoss = nn.CrossEntropyLoss
        
    def forward(self, pred, target):
        # 计算pt
        pt = nn.functional.softmax(pred, dim=1)
        target = nn.functional.one_hot(target, num_classes=len(INTENT_CLASSES)).to(torch.float32)
        pt = pt * target + (1 - pt) * (1 - target)

        # 加入聚焦参数gamma
        focalLoss = - torch.log(pt)
        if self.gamma.shape != torch.Size([0]):
            self.gamma = self.gamma.to(pt.device)
            focalLoss = ((1 - pt) ** self.gamma) * focalLoss
        else:
            focalLoss = torch.pow(1 - pt, 0) * focalLoss

        # 加入正负样本比例调节参数alpha
        if self.alpha.shape != torch.Size([0]):
            localAlpha = torch.matmul(target, self.alpha.to(focalLoss.device))
            focalLoss = focalLoss * localAlpha[:, None]
        return torch.mean(torch.sum(focalLoss))

def prepare_data_loaders(annotation_file, data_dir=DATA_DIR, batch_size=32, 
                          valid_annotation_file=None, val_split=0.1, seed=42):
    """准备训练和验证数据加载器
    
    Args:
        annotation_file: 训练数据标注文件路径
        data_dir: 数据目录
        batch_size: 批大小
        valid_annotation_file: 独立验证集标注文件路径，优先使用
        val_split: 从训练集中分割验证集的比例(仅当valid_annotation_file未指定时使用)
        seed: 随机种子
    
    Returns:
        train_loader: 训练数据加载器
        val_loader: 验证数据加载器 
        intent_labels: 意图标签列表
    """
    # 加载训练数据集
    train_dataset = StreamingAudioDataset(
        annotation_file=annotation_file,
        data_dir=data_dir,
        streaming_mode=False,  # 完整模式
        use_random_crop=False,
        use_feature_augmentation=False
    )
    
    # 获取意图标签
    intent_labels = train_dataset.intent_labels
    
    # 创建验证集
    if valid_annotation_file:
        # 使用独立验证集（不同说话者）
        val_dataset = StreamingAudioDataset(
            annotation_file=valid_annotation_file,
            data_dir=data_dir,
            streaming_mode=False,
            use_random_crop=False,
            use_feature_augmentation=False
        )
        print(f"使用独立验证集: {valid_annotation_file}")
        print(f"训练集样本数: {len(train_dataset)}, 验证集样本数: {len(val_dataset)}")
    else:
        # 从训练集分割验证集
        val_size = int(len(train_dataset) * val_split)
        train_size = len(train_dataset) - val_size
        
        set_seed(seed)
        train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])
        print(f"从训练集随机分割 - 训练集: {train_size}, 验证集: {val_size}")
    
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
        
        
        # 清零梯度
        optimizer.zero_grad()
        
        # 前向传播
        outputs = model(features)
        
        loss = criterion(outputs, labels)
        
        # 反向传播和优化
        loss.backward()
        
        # 梯度裁剪，防止梯度爆炸
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        
        optimizer.step()
        
        # 计算准确率
        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        
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
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="验证中"):
            # 处理batch数据，现在batch是(features, labels)形式
            features, labels = batch
            features = features.to(device)
            labels = labels.to(device)
            
            # 序列长度截断
            if seq_length is not None and features.size(1) > seq_length:
                # 从序列开始处截断更合理
                features = features[:, :seq_length, :]
            
            # 前向传播
            outputs = model(features)
            
            # 计算损失
            loss = criterion(outputs, labels)
            
            # 统计
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # 收集用于计算指标
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # 计算平均损失和准确率
    val_loss = running_loss / len(dataloader)
    val_acc = 100 * correct / total
    
    return val_loss, val_acc, all_preds, all_labels

def evaluate_streaming_model(model, test_annotation_file, data_dir, 
                            save_path, device=DEVICE, confidence_threshold=0.85):
    """评估流式模型，模拟真实流式推理场景
    
    Args:
        model: 模型
        test_annotation_file: 测试集标注文件
        data_dir: 数据目录
        save_path: 结果保存路径
        device: 设备
        confidence_threshold: 提前终止的置信度阈值
    
    Returns:
        accuracy: 准确率
    """
    # 创建测试数据集
    test_dataset = StreamingAudioDataset(
        annotation_file=test_annotation_file,
        data_dir=data_dir,
        streaming_mode=True,
        chunk_size=STREAMING_CHUNK_SIZE,
        step_size=STREAMING_STEP_SIZE
    )
    
    # 模型评估模式
    model.eval()
    
    # 结果统计
    correct = 0
    total = 0
    early_stops = 0
    latencies = []
    all_preds = []
    all_labels = []
    pred_changes = []
    final_confidences = []

    # 用于记录结果的列表
    wrong_results = []
    intent_labels = [
        "CAPTURE_AND_DESCRIBE",
        "OTHERS",
        "START_RECORDING",
        "STOP_RECORDING",
        "TAKE_PHOTO"
    ]

    num_classes = len(intent_labels)
    confusion = np.zeros((num_classes, num_classes), dtype=int)
    target_intent_totals = np.zeros(num_classes, dtype=int)
    target_intent_corrects = np.zeros(num_classes, dtype=int)
    others_total = 0
    others_to_non_others = 0 
    # 测试每个样本
    for i in tqdm(range(len(test_dataset)), desc="评估流式模型"):
        sample = test_dataset[i]
        chunk_features = sample['chunk_features']
        true_label = sample['label']
        file_name = sample['file_path']
        gender = sample['gender']
        transcript = sample['transcript']
        
        # 重置模型流式状态
        if hasattr(model, 'module') and hasattr(model.module, 'reset_streaming_state'):
            model.module.reset_streaming_state()
        else:
            model.reset_streaming_state()
        cached_states = None
        
        # 过程中的预测
        predictions = []
        confidences = []
        
        # 模拟流式处理
        for chunk in chunk_features:
            # 转换为张量并扩展批次维度
            chunk_tensor = torch.FloatTensor(chunk).unsqueeze(0).to(device)
            
            # 流式预测
            with torch.no_grad():
                if hasattr(model, "module") and hasattr(model.module, "predict_streaming"):
                    pred, conf, cached_states = model.module.predict_streaming(chunk_tensor, cached_states)
                else:
                    pred, conf, cached_states = model.predict_streaming(chunk_tensor, cached_states)
            
            pred_label = pred.item()
            conf_value = conf.item()
            
            # 记录预测和置信度
            predictions.append(pred_label)
            confidences.append(conf_value)
            
            # 检查是否达到提前终止条件
            if conf_value > confidence_threshold:
                early_stops += 1
                break
        
        # 最终预测（如果未提前终止，则使用最后一个预测）
        final_pred = predictions[-1]
        final_conf = confidences[-1]
        
        # 计算预测变化次数
        changes = sum(1 for i in range(1, len(predictions)) if predictions[i] != predictions[i-1])
        pred_changes.append(changes)
        final_confidences.append(final_conf)
        
        # 统计正确率
        total += 1
        correct += (final_pred == true_label)
        if final_pred != true_label:
            wrong_results.append({
                "true_label": intent_labels[true_label],
                "final_pred": intent_labels[final_pred],
                "cofidence": final_conf,
                "file_name": file_name,
                "gender": gender,
                "transcript": transcript
            })
        
        # 记录潜伏期（从开始到决策的时间）
        latency = len(predictions) * STREAMING_STEP_SIZE * 0.01  # 秒
        latencies.append(latency)
        
        # 保存结果用于详细分析
        all_preds.append(final_pred)
        all_labels.append(true_label)
        confusion[true_label, final_pred] += 1
        target_intent_totals[true_label] += 1
        if final_pred == true_label:
            target_intent_corrects[true_label] += 1
        if true_label == intent_labels.index("OTHERS"):
            others_total += 1
            if final_pred != intent_labels.index("OTHERS"):
                others_to_non_others += 1 
                

    lines = []
    # 新增不记错的统计
    OTHERS_IDX = intent_labels.index("OTHERS")
    intent_label_names = test_dataset.intent_labels
    filtered_true = []
    filtered_pred = []
    intent_accs = []   # 目标意图准确率收集
    intent_conf_rates = []  # 指令混淆率收集

    for label, pred in zip(all_labels, all_preds):
        if label == OTHERS_IDX:
            filtered_true.append(label)
            filtered_pred.append(pred)
        elif pred != OTHERS_IDX:
            filtered_true.append(label)
            filtered_pred.append(pred)
    report_str_filtered = classification_report(filtered_true, filtered_pred, target_names=intent_label_names, digits=4)
    lines.append("\n====== 自定义分类到OTHERS不计错统计 ======")

    lines.append("目标意图准确率：")
    for idx, label in enumerate(intent_labels):
        if label == "OTHERS":
            continue
        num = sum(1 for t in filtered_true if t == idx)
        correct = sum(1 for t, p in zip(filtered_true, filtered_pred) if t == idx and p == idx)
        acc = correct / num if num > 0 else 0
        intent_accs.append(acc)
        lines.append(f"{label} - 目标意图准确率: {acc:.4f} ({correct}/{num})")
    lines.append("-"*30)
    lines.append("指令混淆率：")
    for idx, label in enumerate(intent_labels):
        if label == "OTHERS":
            continue
        num = sum(1 for t in filtered_true if t == idx)
        confused = sum(1 for t, p in zip(filtered_true, filtered_pred) if t == idx and p != idx)
        conf_rate = confused / num if num > 0 else 0
        intent_conf_rates.append(conf_rate)
        lines.append(f"{label} - 指令混淆率：{conf_rate:.4f} ({confused}/{num})")

    # 统计每类样本数量
    intent_nums = [sum(1 for t in filtered_true if t == idx) for idx, label in enumerate(intent_labels) if label != "OTHERS"]
    # 加权平均
    if len(intent_accs) > 0:
        weighted_avg_acc = sum(a*n for a, n in zip(intent_accs, intent_nums)) / sum(intent_nums)
        lines.append(f"目标意图准确率（加权平均）: {weighted_avg_acc:.4f}")
    if len(intent_conf_rates) > 0:
        weighted_avg_conf = sum(a*n for a, n in zip(intent_conf_rates, intent_nums)) / sum(intent_nums)
        lines.append(f"指令混淆率（加权平均）: {weighted_avg_conf:.4f}")


    lines.append("-"*30)
    others_total_f = sum(1 for t in filtered_true if t == OTHERS_IDX)
    others_to_non_others_f = sum(1 for t, p in zip(filtered_true, filtered_pred) if t == OTHERS_IDX and p != OTHERS_IDX)
    if others_total_f:
        lines.append(f'错误接受率（OTHERS类被识别为有效意图的比例）：{others_to_non_others_f / others_total_f:.4f}  ({others_to_non_others_f}/{others_total_f})')
    else:
        lines.append("OTHERS样本数为0")
    lines.append("="*30)
    lines.append("\n分类报告:")
    lines.append(report_str_filtered)

    lines.append("="*60)

    # 计算总体准确率
    accuracy = correct / total

    # 计算平均潜伏期和早停比例
    avg_latency = sum(latencies) / len(latencies) if latencies else 0
    early_stop_rate = early_stops / total
    
    # 打印评估结果
    lines.append("\n流式评估结果:")
    lines.append(f"准确率: {accuracy:.4f}")
    lines.append(f"平均潜伏期: {avg_latency:.4f} 秒")
    lines.append(f"早停比例: {early_stop_rate:.4f}")
    lines.append(f"平均预测变化次数: {sum(pred_changes) / len(pred_changes):.2f}")
    lines.append(f"平均最终置信度: {sum(final_confidences) / len(final_confidences):.4f}")
    lines.append("="*30)

    all_acc = []
    all_conf_rate = []
    all_num = []

    lines.append("目标意图准确率：")
    for idx, label in enumerate(intent_labels):
        if label == "OTHERS":
            continue
        num = target_intent_totals[idx]
        correct = target_intent_corrects[idx]
        acc = correct / num if num > 0 else 0
        all_acc.append(acc)
        all_num.append(num)
        lines.append(f"{label} - 目标意图准确率: {acc:.4f} ({correct}/{num})")
    lines.append("-"*30)
    lines.append("指令混淆率：")
    for idx, label in enumerate(intent_labels):
        if label == "OTHERS":
            continue
        num = target_intent_totals[idx]
        confused = num - target_intent_corrects[idx]
        conf_rate = confused / num if num > 0 else 0
        all_conf_rate.append(conf_rate)
        lines.append(f"{label} - 指令混淆率：{conf_rate:.4f} ({confused}/{num})")
    # 加权平均
    total_num = sum(all_num)
    if total_num > 0:
        weighted_acc = sum(a * n for a, n in zip(all_acc, all_num)) / total_num
        weighted_conf_rate = sum(c * n for c, n in zip(all_conf_rate, all_num)) / total_num
        lines.append(f"目标意图准确率（加权平均）: {weighted_acc:.4f}")
        lines.append(f"指令混淆率（加权平均）: {weighted_conf_rate:.4f}")

    sum_confused = np.sum(target_intent_totals[:-1] - target_intent_corrects[:-1])
    sum_totals = np.sum(target_intent_totals[:-1])
    lines.append("-"*30)
    lines.append(f'错误接受率（OTHERS类被识别为有效意图的比例）：{others_to_non_others / others_total:.4f}  ({others_to_non_others}/{others_total})' if others_total else "OTHERS样本数为0")
    lines.append("="*30)
    # 计算分类报告
    intent_label_names = test_dataset.intent_labels
    lines.append("\n分类报告:")
    report_str = classification_report(all_labels, all_preds, target_names=intent_label_names)
    report_dict = classification_report(all_labels, all_preds, target_names=intent_label_names, output_dict=True)
    lines.append(report_str)


    final_report_str = "\n".join(lines)
    print(final_report_str)
    
    # 绘制混淆矩阵
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", 
                xticklabels=intent_label_names, 
                yticklabels=intent_label_names)
    plt.xlabel('predicted')
    plt.ylabel('true')
    plt.title('confusion matrix')
    plt.tight_layout()
    plt.savefig(os.path.join(os.path.dirname(save_path), "streaming_confusion_matrix.png"))
    
    # 绘制预测稳定性分析
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.hist(pred_changes, bins=10, alpha=0.7)
    plt.xlabel('prediction changes')
    plt.ylabel('samples')
    plt.title('prediction stability distribution')
    
    plt.subplot(1, 2, 2)
    plt.hist(final_confidences, bins=10, alpha=0.7)
    plt.xlabel('final confidence')
    plt.ylabel('samples')
    plt.title('final confidence distribution')
    
    plt.tight_layout()
    plt.savefig(os.path.join(os.path.dirname(save_path), "prediction_stability.png"))
    df = pd.DataFrame(wrong_results)
    df.to_csv(os.path.join(os.path.dirname(save_path), "wrong_result.csv"), index=False)

    # 保存结果
    results = {
        'accuracy': accuracy,
        'avg_latency': avg_latency,
        'early_stop_rate': early_stop_rate,
        'pred_changes': pred_changes,
        'final_confidences': final_confidences,
        'predictions': all_preds,
        'true_labels': all_labels
    }
    
    # 确保目录存在
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # 保存结果
    import pickle
    with open(save_path, 'wb') as f:
        pickle.dump(results, f)

    return final_report_str, report_dict

def train_streaming_conformer(data_dir, annotation_file, model_save_path, 
                            valid_annotation_file=None, num_epochs=30, batch_size=32, seed=42,
                            num_layers= CONFORMER_LAYERS, learning_rate=LEARNING_RATE, weight_decay=0.01,
                            dropout=CONFORMER_DROPOUT, kernel_size=CONFORMER_CONV_KERNEL_SIZE,
                            use_mixup=USE_MIXUP,
                            loss_function=LOSS_FUNC_CROSS_ENTROPY,
                            label_smoothing=LABEL_SMOOTHING_VALUE,
                            focal_loss_alpha = FOCAL_LOSS_ALPHA_VALUE,
                            focal_loss_gamma = FOCAL_LOSS_GAMMA_VALUE,
                            progressive_training=PROGRESSIVE_TRAINING):
    """训练流式Conformer模型
    
    Args:
        data_dir: 数据目录
        annotation_file: 训练数据标注文件路径
        model_save_path: 模型保存路径
        valid_annotation_file: 独立验证集标注文件路径
        num_epochs: 训练轮数
        batch_size: 批大小
        seed: 随机种子
        learning_rate: 学习率
        weight_decay: 权重衰减
        use_mixup: 是否使用MixUp
        loss_function: 选择损失函数0(Cross Entrop),1(Label Smoothing), 2(Focal Loss)
        label_smoothing: 标签平滑参数 (LabelSmoothingCrossEntropy)
        focal_loss_alpha(list): Focal loss 正负样本比例调节参数维度与函数output一致
        focal_loss_gamma(list): Focal loss 聚焦参数gamma维度与函数output一致
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
        valid_annotation_file=valid_annotation_file,
        seed=seed
    )
    
    # 创建模型
    input_dim = N_MFCC * 3  # MFCC及其Delta特征
    hidden_dim = CONFORMER_HIDDEN_SIZE
    num_classes = len(intent_labels)
    num_heads = CONFORMER_ATTENTION_HEADS
    dropout = dropout
    kernel_size = kernel_size
    expansion_factor = CONFORMER_FF_EXPANSION_FACTOR
    
    model = StreamingConformer(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_classes=num_classes,
        num_layers=num_layers,
        num_heads=num_heads,
        dropout=dropout,
        kernel_size=kernel_size,
        expansion_factor=expansion_factor
    )
    if torch.cuda.device_count() > 1:
        print(f"使用{torch.cuda.device_count()}个GPU进行并行训练")
        model = nn.DataParallel(model)
    model = model.to(DEVICE)
    
    # 选择损失函数
    if loss_function == LOSS_FUNC_LABEL_SMOOTHING:
        criterion = LabelSmoothingCrossEntropy(smoothing=label_smoothing)
        print(f"使用标签平滑损失，平滑参数: {label_smoothing}")
    elif loss_function == LOSS_FUNC_FOCAL_LOSS:
        criterion = FocalLoss(alpha=focal_loss_alpha, gamma=focal_loss_gamma)
        print(f"使用Focal Loss 损失函数，正负样本比例调节参数alpha: {focal_loss_alpha}，聚焦参数gamma {focal_loss_gamma}")
    else:
        criterion = nn.CrossEntropyLoss()
        print("使用标准交叉熵损失")
    
    # 优化器
    optimizer = optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay
    )
    
    # 学习率调度器
    if USE_COSINE_SCHEDULER:
        # 余弦退火调度器
        T_max = num_epochs
        eta_min = learning_rate / 100
        
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=T_max, eta_min=eta_min
        )
        print(f"使用余弦退火学习率调度，T_max={T_max}，eta_min={eta_min}")
    else:
        # 使用固定学习率
        scheduler = None
        print(f"使用固定学习率: {learning_rate}")
    
    # 渐进式长度训练设置
    if progressive_training:
        # 渐进式长度训练 - 从短序列开始，增量增加序列长度
        # 根据模型输入特性调整序列长度范围
        progressive_lengths = [
            32, 64, 96, 128, 160, 192, None  # None表示使用完整序列
        ]
        print(f"启用渐进式长度训练，序列长度进度: {progressive_lengths}")
    else:
        progressive_lengths = [None] * num_epochs  # 所有epoch使用完整序列
        print("不使用渐进式长度训练，使用完整序列")
    
    # 训练历史
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }
    
    # 早停设置
    best_val_acc = 0
    patience = EARLY_STOPPING_PATIENCE
    patience_counter = 0
    
    # 训练循环
    print(f"\n开始训练流式Conformer模型...")
    print(f"模型配置: 隐藏层={CONFORMER_HIDDEN_SIZE}, 层数={CONFORMER_LAYERS}, 头数={CONFORMER_ATTENTION_HEADS}")
    print(f"特征维度: {N_MFCC * 3}, 卷积核大小: {CONFORMER_CONV_KERNEL_SIZE}")
    print(f"训练样本数: {len(train_loader.dataset)}, 验证样本数: {len(val_loader.dataset)}")
    print(f"总训练轮数: {num_epochs}")
    
    start_time = time.time()

    for epoch in range(num_epochs):
        # 确定当前使用的序列长度
        if progressive_training:
            current_seq_length = progressive_lengths[min(epoch, len(progressive_lengths) - 1)]
        else:
            current_seq_length = None
        
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print(f"当前序列长度: {'完整序列' if current_seq_length is None else current_seq_length}")
        print(f"当前学习率: {optimizer.param_groups[0]['lr']:.6f}")
        
        # 训练一个epoch
        train_loss, train_acc = train_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=DEVICE,
            seq_length=current_seq_length,
            use_mixup=use_mixup
        )
        
        # 验证
        val_loss, val_acc, val_preds, val_labels = validate(
            model=model,
            dataloader=val_loader,
            criterion=criterion,
            device=DEVICE,
            seq_length=current_seq_length
        )
        
        # 更新学习率
        if scheduler is not None:
            scheduler.step()
        
        # 记录历史
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        # 打印进度
        print(f"训练损失: {train_loss:.4f}, 训练准确率: {train_acc:.2f}%")
        print(f"验证损失: {val_loss:.4f}, 验证准确率: {val_acc:.2f}%")
        
        # 检查是否是最佳模型
        if val_acc > best_val_acc:
            print(f"验证准确率提升: {best_val_acc:.2f}% -> {val_acc:.2f}%")
            best_val_acc = val_acc
            patience_counter = 0
            
            # 保存最佳模型
            save_dict = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'intent_labels': intent_labels,
                'history': history,
                # 添加模型超参数
                'model_config': {
                    'input_dim': input_dim,
                    'hidden_dim': hidden_dim,
                    'num_classes': num_classes,
                    'num_layers': num_layers,
                    'num_heads': num_heads,
                    'dropout': dropout,
                    'kernel_size': kernel_size,
                    'expansion_factor': expansion_factor,
                    'model_type': 'streaming_conformer' # 添加模型类型标识
                }
            }
            torch.save(save_dict, model_save_path)
            print(f"保存最佳模型到: {model_save_path}")
        else:
            patience_counter += 1
            print(f"验证准确率未提升，当前耐心: {patience_counter}/{patience}")
            
            # 早停检查
            if USE_EARLY_STOPPING and patience_counter >= patience:
                print(f"早停: {patience}轮未提升，停止训练")
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

    return model, intent_labels, history

def plot_training_history(history, save_dir):
    """绘制训练历史
    
    Args:
        history: 训练历史字典
        save_dir: 保存目录
    """
    plt.figure(figsize=(15, 5))
    
    # 绘制损失
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='train loss')
    plt.plot(history['val_loss'], label='validation loss')
    plt.xlabel('Epoch')
    plt.ylabel('loss')
    plt.title('train and validation loss')
    plt.legend()
    plt.grid(True)
    
    # 绘制准确率
    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='train accuracy')
    plt.plot(history['val_acc'], label='validation accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('accuracy(%)')
    plt.title('train and validation accuracy')
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
    parser.add_argument('--valid_annotation_file', type=str, help='验证数据标注文件（使用不同说话者）')
    parser.add_argument('--test_annotation_file', type=str, help='测试数据标注文件（用于评估）')
    
    # 模型参数
    parser.add_argument('--model_type', type=str, default='streaming_conformer', help='模型类型')
    parser.add_argument('--model_save_path', type=str, required=True, help='模型保存路径')
    parser.add_argument('--hidden_dim', type=int, default=CONFORMER_HIDDEN_SIZE, help='隐藏层维度')
    parser.add_argument('--num_layers', nargs="+", type=int, default=CONFORMER_LAYERS, help='Conformer层数')
    parser.add_argument('--num_heads', type=int, default=CONFORMER_ATTENTION_HEADS, help='注意力头数')
    
    # 训练参数
    parser.add_argument('--num_epochs', type=int, default=NUM_EPOCHS, help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE, help='批大小')
    parser.add_argument('--learning_rate', nargs="+", type=float, default=LEARNING_RATE, help='学习率')
    parser.add_argument('--dropout', nargs="+", type=float, default=CONFORMER_DROPOUT, help='dropout')
    parser.add_argument('--kernel_size', nargs="+", type=int, default=CONFORMER_CONV_KERNEL_SIZE, help='卷积核大小')
    parser.add_argument('--weight_decay', nargs="+", type=float, default=0.01, help='权重衰减')
    parser.add_argument('--seed', type=int, default=41, help='随机种子')
    
    # 增强和训练策略
    parser.add_argument('--use_mixup', action='store_true', default=USE_MIXUP, help='是否使用MixUp增强')
    parser.add_argument('--loss_function', nargs="+", type=int, default=LOSS_FUNC_CROSS_ENTROPY, help='选择使用的loss function')
    parser.add_argument('--label_smoothing', nargs="+", type=float, default=LABEL_SMOOTHING_VALUE, help='标签平滑系数')
    parser.add_argument('--focal_loss_alpha', nargs="+", type=str, default="[]", help='Focal loss 正负样本比例调节参数')
    parser.add_argument('--focal_loss_gamma', nargs="+", type=str, default="[]", help='Focal loss 聚焦参数gamma')
    parser.add_argument('--progressive_training', action='store_true', default=PROGRESSIVE_TRAINING, help='是否使用渐进式训练')
    parser.add_argument('--eval_interval', type=int, default=1, help='每多少个epoch评估一次')
    
    # 评估参数
    parser.add_argument('--evaluate', action='store_true', help='是否评估模型')
    parser.add_argument('--confidence_threshold', type=float, default=0.85, help='流式评估的置信度阈值')
    
    return parser.parse_args()

def string_to_float_list(listStr):
    res = []
    listStr = listStr.strip().replace("[", "").replace("]", "").replace("\n", "")
    for ele in listStr.split(","):
        ele = ele.strip()
        if not ele:
            continue
        res.append(float(ele))
    return res

LOSS_FUNC_LABEL_SMOOTHING = 1
LOSS_FUNC_FOCAL_LOSS = 2
def prepare_grid_search(args):
    # 1. 定义所有超参数字典（每项都是一个list）
    search_space = {
        "learning_rate": args.learning_rate,
        "num_layers": args.num_layers,
        "dropout": args.dropout,
        "kernel_size": args.kernel_size,
        "weight_decay": args.weight_decay,
        "loss_function": args.loss_function,
        "loss_function_label_smoothing": args.label_smoothing,
        "loss_function_focal_loss_alpha": args.focal_loss_alpha,
        "loss_function_focal_loss_gamma": args.focal_loss_gamma,
    }
    # 2. 准备 Keys 列表 (便于顺次展开组合)
    keys = ["learning_rate", "num_layers", "dropout", "kernel_size", "weight_decay", "loss_function"]
    # 单独处理 loss 相关交叉（后面组合时动态加进去）
    grid_input_values = []
    # 3. 穷举前5个主超参（不含loss-related次级超参）
    for vals in itertools.product(*(search_space[k] for k in keys)):
        # 组合一个dict
        cur_hparam = dict(zip(keys, vals))
        # loss相关处理
        lf = cur_hparam["loss_function"]
        if lf == LOSS_FUNC_LABEL_SMOOTHING:
            # 仅label_smoothing变，focal的设为默认第0值
            for label_smoothing in search_space["loss_function_label_smoothing"]:
                hparam = copy.deepcopy(cur_hparam)
                hparam["loss_function_label_smoothing"] = label_smoothing
                hparam["loss_function_focal_loss_alpha"] = search_space["loss_function_focal_loss_alpha"][0]
                hparam["loss_function_focal_loss_gamma"] = search_space["loss_function_focal_loss_gamma"][0]
                grid_input_values.append(hparam)
        elif lf == LOSS_FUNC_FOCAL_LOSS:
            # 仅focal alpha/gamma变，label_smoothing用默认值
            for alpha in search_space["loss_function_focal_loss_alpha"]:
                for gamma in search_space["loss_function_focal_loss_gamma"]:
                    hparam = copy.deepcopy(cur_hparam)
                    hparam["loss_function_label_smoothing"] = search_space["loss_function_label_smoothing"][0]
                    hparam["loss_function_focal_loss_alpha"] = alpha
                    hparam["loss_function_focal_loss_gamma"] = gamma
                    grid_input_values.append(hparam)
        else:
            # 都用默认第0个值
            hparam = copy.deepcopy(cur_hparam)
            hparam["loss_function_label_smoothing"] = search_space["loss_function_label_smoothing"][0]
            hparam["loss_function_focal_loss_alpha"] = search_space["loss_function_focal_loss_alpha"][0]
            hparam["loss_function_focal_loss_gamma"] = search_space["loss_function_focal_loss_gamma"][0]
            grid_input_values.append(hparam)
    return grid_input_values

def print_hyperpara_setting(args):
    # 配置
    print("\n=== 流式Conformer训练配置 ===")
    print(f"数据目录: {args.data_dir}")
    print(f"训练标注文件: {args.annotation_file}")
    if args.valid_annotation_file:
        print(f"验证标注文件(独立): {args.valid_annotation_file}")
    if args.test_annotation_file:
        print(f"测试标注文件: {args.test_annotation_file}")
    print(f"模型保存路径: {args.model_save_path}")
    print(f"隐藏层维度: {args.hidden_dim}")
    print(f"Conformer层数: {args.num_layers}")
    print(f"注意力头数: {args.num_heads}")
    print(f"卷积核大小: {args.kernel_size}")
    print(f"训练轮数: {args.num_epochs}")
    print(f"批大小: {args.batch_size}")
    print(f"学习率: {args.learning_rate}")
    print(f"使用MixUp: {args.use_mixup}")
    print(f"损失函数: {args.loss_function}")
    if LOSS_FUNC_LABEL_SMOOTHING in args.loss_function:
        print(f"标签平滑: {args.label_smoothing}")
    elif LOSS_FUNC_FOCAL_LOSS in args.loss_function:
        print(f"Focal Loss Alpha: {args.focal_loss_alpha}")
        print(f"Focal Loss Gamma: {args.focal_loss_gamma}")
    print(f"渐进式训练: {args.progressive_training}")


def main():
    """主函数"""
    # 解析命令行参数
    args = parse_args()
    print_hyperpara_setting(args)
    grid_input_values = prepare_grid_search(args)

    # 保留总hyper parameter信息
    summary_data_path =  os.path.join(os.path.dirname(args.model_save_path), "training_summary_data")
    os.makedirs(summary_data_path,  exist_ok=True)
    with open(os.path.join(summary_data_path, "hyperparameters.txt"), "w") as f:
        f.write(f"hyper parameter name    : hyper parameter value")
        for hyper_para in args._get_kwargs():
            f.write(f"{hyper_para[0]:24}: {hyper_para[1]}\n")
    
    # 保留总训练信息
    summary_tarin_result = pd.DataFrame(columns=["module_name", "max_train_accuracy"])
    # 训练模型
    for grid_idx, grid_input_value in enumerate(grid_input_values):
        grid_input_value = grid_input_values[grid_idx]
        grid_train_time = time.strftime("%Y%m%d_%H%M%S", time.localtime())
        module_name = f"module_{grid_train_time}"
        try:
            summary_tarin_result.loc[grid_idx, "module_name"] = module_name
            model_save_path = os.path.join(os.path.dirname(args.model_save_path), module_name)
            os.makedirs(model_save_path, exist_ok=True)
            f_res = open(os.path.join(model_save_path, "module_data.txt"), "w")
            f_res.write("hyper parameters:\n")
            print("当前超参数:\n")
            for para_key in grid_input_value:
                f_res.write(f"    {para_key}: {grid_input_value[para_key]}\n")
                print(f"    {para_key}: {grid_input_value[para_key]}")

            local_model_path = os.path.join(model_save_path, os.path.basename(args.model_save_path))
            model, intent_labels, train_history = train_streaming_conformer(
                data_dir=args.data_dir,
                annotation_file=args.annotation_file,
                model_save_path=local_model_path,
                valid_annotation_file=args.valid_annotation_file,
                num_epochs=args.num_epochs,
                batch_size=args.batch_size,
                seed=args.seed,
                num_layers=grid_input_value["num_layers"],
                learning_rate=grid_input_value["learning_rate"],
                dropout=grid_input_value["dropout"],
                kernel_size=grid_input_value["kernel_size"],
                weight_decay=grid_input_value["weight_decay"],
                use_mixup=args.use_mixup,
                loss_function=grid_input_value["loss_function"],
                label_smoothing=grid_input_value["loss_function_label_smoothing"],
                focal_loss_alpha=string_to_float_list(grid_input_value["loss_function_focal_loss_alpha"]),
                focal_loss_gamma=string_to_float_list(grid_input_value["loss_function_focal_loss_gamma"]),
                progressive_training=args.progressive_training
            )
            summary_tarin_result.loc[grid_idx, "max_train_accuracy"] = max(train_history["train_acc"])
            # 评估模型
            if args.evaluate and args.test_annotation_file:
                print("\n=== 评估流式模型 ===")
                print(f"测试数据: {args.test_annotation_file}")
                print(f"置信度阈值: {args.confidence_threshold}")

                # 模型评估路径
                eval_save_path = os.path.join(model_save_path, "streaming_eval_results.pkl")

                # 流式评估
                report_str, report_dict = evaluate_streaming_model(
                    model=model,
                    test_annotation_file=args.test_annotation_file,
                    data_dir=args.data_dir,
                    save_path=eval_save_path,
                    device=DEVICE,
                    confidence_threshold=args.confidence_threshold
                )
                accuracy  = report_dict["accuracy"]
                f_res.write(f"accuracy: {accuracy:.4f}\n")
                f_res.write(f"report:\n")
                f_res.write(report_str)
                f_res.close()
                print(f"\n流式评估准确率: {accuracy:.4f}")
                for key in report_dict:
                    if type(report_dict[key]) == dict:
                        for subkey in report_dict[key]:
                            summary_tarin_result.loc[grid_idx, f"test_{key}_{subkey}"] = report_dict[key][subkey]
                    else:
                        summary_tarin_result.loc[grid_idx, f"test_{key}"] = report_dict[key]
            print(f"流式Conformer模型训练完成！模型保存在:{local_model_path}")
            print("导出ONNX模型...")
            local_model_bas_path, local_model_ext_path = os.path.splitext(local_model_path)
            cmd = ["python3", "export_onnx.py",
                "--model_path", local_model_path,
                "--model_type", "streaming",
                "--onnx_save_path", local_model_bas_path + ".onnx"]
            result = subprocess.run(cmd, capture_output=True, text=True)
            print(f"导出完成，模型保存在{local_model_bas_path}\n")
        except KeyboardInterrupt:
            # Ctrl+C 时会走到这里然后退出
            print("检测到用户中断，程序退出！")
            break  # 或用 exit()/sys.exit()
        except Exception as e:
            print(f"{module_name} fail \n {e}\n\n")
            print(traceback.format_exc())
            time.sleep(5)
    summary_tarin_result.to_csv(os.path.join(summary_data_path, "train_test_result.csv"), index=False)
    print("\n训练和评估完成!")

if __name__ == "__main__":
    main() 