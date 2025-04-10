# train.py  
import os  
import argparse  
import torch  
import torch.nn as nn  
import torch.optim as optim  
import numpy as np  
import matplotlib.pyplot as plt
from tqdm import tqdm  
from sklearn.metrics import classification_report, accuracy_score  
import time
import random
from config import *  
from augmented_dataset import prepare_augmented_dataloader, standardize_audio_length
from streaming_dataset import prepare_streaming_dataloader
from models.fast_classifier import FastIntentClassifier  
import librosa
import torch.onnx
import pandas as pd
import shutil
import pickle
import traceback
from models.streaming_conformer import StreamingConformer

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

def extract_features(audio, sr):
    """提取音频特征
    
    参数:
        audio: 音频数据
        sr: 采样率
        
    返回:
        提取的特征，形状为(time, features)
    """
    try:
        # 标准化音频长度
        audio = standardize_audio_length(audio, sr)
        
        # 提取MFCC特征
        # 可以根据需要修改特征提取方法
        mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=16)
        delta_mfccs = librosa.feature.delta(mfccs)
        delta2_mfccs = librosa.feature.delta(mfccs, order=2)
        
        # 合并特征
        features = np.concatenate([mfccs, delta_mfccs, delta2_mfccs], axis=0)
        
        # 转置为(time, features)格式
        features = features.T
        
        return features
    except Exception as e:
        print(f"特征提取错误: {e}")
        # 返回一个安全的默认特征
        return np.zeros((100, 48))  # 默认100帧，48个特征

def train_fast_model(data_dir, annotation_file, model_save_path, num_epochs=NUM_EPOCHS, 
                    augment=True, augment_prob=0.5, use_cache=True, seed=42):
    """训练快速分类器模型"""
    set_seed(seed)
    
    # 定义特征提取器
    def feature_extractor(audio, sr, **kwargs):
        audio = standardize_audio_length(audio, sr)
        # 提取MFCC特征和动态特征
        mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=16)
        delta_mfccs = librosa.feature.delta(mfccs)
        delta2_mfccs = librosa.feature.delta(mfccs, order=2)
        features = np.concatenate([mfccs, delta_mfccs, delta2_mfccs], axis=0)
        return features.T  # (time, features) 格式，不含上下文

    # 如果num_epochs为0，跳过训练过程，直接加载模型(如果存在)或创建新模型
    if num_epochs == 0:
        # 创建模型（MFCC+Delta+Delta2特征，共48维）
        input_size = N_MFCC * 3
        
        # 尝试加载已有模型
        if os.path.exists(model_save_path):
            print(f"加载已有模型: {model_save_path}")
            checkpoint = torch.load(model_save_path, map_location=torch.device('cpu'))
            
            # 检查是否是新的保存格式
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                model_state = checkpoint['model_state_dict']
                intent_labels = checkpoint['intent_labels']
                model = FastIntentClassifier(input_size=input_size, num_classes=len(intent_labels))
                model.load_state_dict(model_state)
                print(f"模型加载成功，意图标签: {intent_labels}")
                return model, intent_labels
            else:
                print("模型格式不兼容，需要先训练模型")
                # 继续正常训练流程
        else:
            print(f"模型文件不存在: {model_save_path}")
            print("无法跳过训练过程，请提供有效的模型文件或设置num_epochs > 0")
            # 继续正常训练流程，但需要获取意图标签
            train_loader, train_labels = prepare_augmented_dataloader(
                annotation_file=annotation_file, 
                data_dir=data_dir, 
                feature_extractor=feature_extractor,
                batch_size=1,  # 最小批次，仅用于获取标签
                augment=False,
                use_cache=False
            )
            # 创建模型
            model = FastIntentClassifier(input_size=input_size, num_classes=len(train_labels))
            # 保存模型
            save_dict = {
                'model_state_dict': model.state_dict(),
                'intent_labels': train_labels,
                'input_size': input_size
            }
            # 确保目录存在
            model_dir = os.path.dirname(model_save_path)
            if model_dir and not os.path.exists(model_dir):
                os.makedirs(model_dir)
            torch.save(save_dict, model_save_path)
            print(f"已创建新模型并保存到: {model_save_path}")
            return model, train_labels

    # 准备数据加载器
    train_loader, train_labels = prepare_augmented_dataloader(
        annotation_file=annotation_file, 
        data_dir=data_dir, 
        feature_extractor=feature_extractor,
        batch_size=BATCH_SIZE, 
        augment=augment,
        augment_prob=augment_prob,
        use_cache=use_cache,
        seed=seed
    )
    
    # 创建模型（MFCC+Delta+Delta2特征，共48维）
    input_size = N_MFCC * 3
    model = FastIntentClassifier(input_size=input_size, num_classes=len(train_labels))
    model = model.to(DEVICE)  
    
    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()  
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)  
    
    # 训练历史记录
    history = {
        'train_loss': [],
        'train_acc': []
    }
    
    # 开始训练
    print(f"开始训练快速分类器模型...")
    print(f"训练数据大小: {len(train_loader.dataset)}")
    print(f"意图标签: {train_labels}")
    print(f"数据增强: {'已启用' if augment else '已禁用'} (概率: {augment_prob})")
    
    start_time = time.time()
    
    for epoch in range(num_epochs):  
        epoch_start = time.time()
        model.train()  
        
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        # 训练循环
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            features = batch['features'].to(DEVICE)
            labels = batch['label'].to(DEVICE)
            
            # 前向传播  
            outputs = model(features)  
            loss = criterion(outputs, labels)  
            
            # 反向传播和优化  
            optimizer.zero_grad()
            loss.backward()  
            optimizer.step()  
            
            # 计算统计信息
            train_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
        
        # 计算epoch级别的指标
        epoch_loss = train_loss / len(train_loader)
        epoch_acc = train_correct / train_total
        epoch_time = time.time() - epoch_start
        
        # 记录训练历史
        history['train_loss'].append(epoch_loss)
        history['train_acc'].append(epoch_acc)
        
        # 显示进度
        print(f"Epoch {epoch+1}/{num_epochs} - "
              f"Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.4f}, "
              f"Time: {epoch_time:.2f}s")
    
    total_time = time.time() - start_time
    print(f"训练完成! 总时间: {total_time:.2f}s")
    
    # 保存模型和标签映射
    model_dir = os.path.dirname(model_save_path)
    if model_dir and not os.path.exists(model_dir):
        os.makedirs(model_dir)
        
    # 保存模型和标签映射
    # 创建包含模型状态和标签映射的字典
    save_dict = {
        'model_state_dict': model.state_dict(),
        'intent_labels': train_labels,
        'input_size': input_size
    }
    torch.save(save_dict, model_save_path)
    print(f"模型已保存到: {model_save_path}")
    
    # 保存训练历史
    if model_dir:
        plot_training_curves(history, model_dir, 'fast', augment)
    
    return model, train_labels

def train_streaming_model(data_dir, annotation_file, model_save_path, num_epochs=NUM_EPOCHS,
                         pretrain_epochs=10, finetune_epochs=10, use_random_crop=True,
                         use_cache=True, seed=42):
    """
    两阶段流式训练模型：预训练 + 流式微调
    
    Args:
        data_dir: 音频数据目录
        annotation_file: 标注文件路径
        model_save_path: 模型保存路径
        num_epochs: 总训练轮数
        pretrain_epochs: 预训练轮数
        finetune_epochs: 流式微调轮数
        use_random_crop: 是否使用随机裁剪增强
        use_cache: 是否使用特征缓存
        seed: 随机种子
    """
    set_seed(seed)
    
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 如果num_epochs为0，跳过训练过程，直接加载模型(如果存在)
    if num_epochs == 0:
        # 流式模型不需要特殊处理参数，直接获取意图标签
        input_size = N_MFCC * 3
        
        # 尝试加载已有模型
        if os.path.exists(model_save_path):
            print(f"加载已有模型: {model_save_path}")
            checkpoint = torch.load(model_save_path, map_location=torch.device('cpu'))
            
            # 检查是否是新的保存格式
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                model_state = checkpoint['model_state_dict']
                intent_labels = checkpoint['intent_labels']
                model = FastIntentClassifier(input_size=input_size, num_classes=len(intent_labels))
                model.load_state_dict(model_state)
                print(f"模型加载成功，意图标签: {intent_labels}")
                return model, intent_labels
            else:
                print("模型格式不兼容，需要先训练模型")
                # 继续正常训练流程
        else:
            print(f"模型文件不存在: {model_save_path}")
            print("无法跳过训练过程，请提供有效的模型文件或设置num_epochs > 0")
            # 继续训练流程，但先获取标签
            pretrain_loader, intent_labels = prepare_streaming_dataloader(
                annotation_file=annotation_file,
                data_dir=data_dir,
                batch_size=1,  # 最小批次，仅用于获取标签
                streaming_mode=False,
                cache_dir=None
            )
            # 创建模型
            model = FastIntentClassifier(input_size=input_size, num_classes=len(intent_labels))
            # 保存模型
            torch.save({
                'model_state_dict': model.state_dict(),
                'intent_labels': intent_labels
            }, model_save_path)
            print(f"已创建新模型并保存到: {model_save_path}")
            return model, intent_labels
    
    # 设置缓存目录
    cache_dir = os.path.join("tmp", "feature_cache") if use_cache else None
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
    
    # 阶段1: 完整音频预训练
    print("阶段1: 完整音频预训练...")
    # 准备完整音频数据加载器
    pretrain_loader, intent_labels = prepare_streaming_dataloader(
        annotation_file=annotation_file,
        data_dir=data_dir,
        batch_size=BATCH_SIZE,
        streaming_mode=False,  # 完整音频模式
        use_random_crop=use_random_crop,
        cache_dir=cache_dir,
        shuffle=True,
        seed=seed
    )
    
    # 创建模型（MFCC+Delta+Delta2特征，共48维）
    input_size = N_MFCC * 3
    model = FastIntentClassifier(input_size=input_size, num_classes=len(intent_labels))
    model = model.to(device)
    
    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # 预训练历史记录
    pretrain_history = {
        'train_loss': [],
        'train_acc': []
    }
    
    # 预训练循环
    for epoch in range(pretrain_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        # 使用tqdm进度条
        progress_bar = tqdm(pretrain_loader, desc=f"预训练 Epoch {epoch+1}/{pretrain_epochs}")
        
        for features, labels in progress_bar:
            features, labels = features.to(device), labels.to(device)
            
            # 清零梯度
            optimizer.zero_grad()
            
            # 前向传播 - 使用 torch.amp 自动混合精度加速
            with torch.amp.autocast(device_type='cuda' if torch.cuda.is_available() else 'cpu'):
                outputs = model(features)
                
                # 计算损失
                loss = criterion(outputs, labels)
            
            # 反向传播和优化
            loss.backward()
            optimizer.step()
            
            # 统计
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # 更新进度条
            progress_bar.set_postfix({
                'loss': running_loss / (progress_bar.n + 1),
                'acc': 100 * correct / total
            })
        
        # 计算训练指标
        epoch_loss = running_loss / len(pretrain_loader)
        epoch_acc = 100 * correct / total
        
        # 保存历史
        pretrain_history['train_loss'].append(epoch_loss)
        pretrain_history['train_acc'].append(epoch_acc)
        
        print(f"预训练 Epoch {epoch+1}/{pretrain_epochs} - 损失: {epoch_loss:.4f}, 准确率: {epoch_acc:.2f}%")
    
    # 保存预训练模型
    pretrain_save_path = f"{model_save_path}_pretrained.pt"
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'history': pretrain_history,
        'intent_labels': intent_labels
    }, pretrain_save_path)
    print(f"预训练模型已保存到: {pretrain_save_path}")
    
    # 阶段2: 流式微调
    print("\n阶段2: 流式微调...")
    
    # 创建流式特征的专用缓存目录
    stream_cache_dir = os.path.join(cache_dir, "streaming_features") if cache_dir else None
    if stream_cache_dir:
        os.makedirs(stream_cache_dir, exist_ok=True)

    # 重置优化器 - 使用更大的学习率来加速收敛
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE * 0.2)  # 调整学习率
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.7, patience=3, verbose=True, min_lr=1e-5)

    # 微调历史记录
    finetune_history = {
        'train_loss': [],
        'train_acc': []
    }

    # 使用预热策略防止崩溃
    warmup_epochs = 3
    warmup_factor = 0.1  # 初始学习率的比例
    
    # 设置特征抖动参数 - 随着训练进行逐渐减小抖动
    jitter_start = 0.04  # 初始抖动强度
    jitter_end = 0.01    # 最终抖动强度
    
    # 微调循环
    for epoch in range(finetune_epochs):
        # 计算当前epoch的抖动比例 - 线性衰减
        current_jitter = jitter_start - (jitter_start - jitter_end) * min(1.0, epoch / (finetune_epochs * 0.7))
        
        # 每个epoch清除流式特征缓存，确保重新生成流式特征
        if epoch > 0 and stream_cache_dir and os.path.exists(stream_cache_dir):
            print(f"清除流式特征缓存，确保重新生成特征...")
            # 清除流式特征缓存目录内容
            for cache_file in os.listdir(stream_cache_dir):
                os.remove(os.path.join(stream_cache_dir, cache_file))
        
        # 每个epoch重新准备流式数据加载器
        epoch_cache_dir = None if epoch > 0 else stream_cache_dir  # 只有第一轮使用缓存
        finetune_loader, _ = prepare_streaming_dataloader(
            annotation_file=annotation_file,
            data_dir=data_dir,
            batch_size=BATCH_SIZE,
            streaming_mode=True,  # 流式模式
            use_random_crop=False,  # 流式模式不需要随机裁剪
            cache_dir=epoch_cache_dir,
            shuffle=True,
            seed=seed + epoch,  # 每轮使用不同随机种子
            use_feature_augmentation=True,  # 启用特征增强
            jitter_ratio=current_jitter,  # 使用动态抖动比例
            mask_ratio=0.05  # 固定掩码比例
        )
        
        print(f"微调 Epoch {epoch+1}/{finetune_epochs} - 使用特征抖动 {current_jitter:.4f}")
        
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        # 在预热阶段使用更小的学习率
        if epoch < warmup_epochs:
            for g in optimizer.param_groups:
                g['lr'] = LEARNING_RATE * warmup_factor * (epoch + 1) / warmup_epochs
        
        # 使用tqdm进度条
        progress_bar = tqdm(finetune_loader, desc=f"微调 Epoch {epoch+1}/{finetune_epochs}")
        
        for features, labels in progress_bar:
            # 将数据移至设备
            features, labels = features.to(device), labels.to(device)
            
            # 清零梯度
            optimizer.zero_grad()
            
            # 前向传播 - 使用 torch.amp 自动混合精度加速
            with torch.amp.autocast(device_type='cuda' if torch.cuda.is_available() else 'cpu'):
                outputs = model(features)
                
                # 确保输出和标签形状匹配
                if outputs.shape[0] != labels.shape[0]:
                    print(f"警告: 输出形状 {outputs.shape} 与标签形状 {labels.shape} 不匹配")
                    continue
                
                # 计算损失
                loss = criterion(outputs, labels)
            
            # 反向传播和优化
            loss.backward()
            # 梯度裁剪，防止梯度爆炸
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            # 统计
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # 更新进度条
            progress_bar.set_postfix({
                'loss': running_loss / (progress_bar.n + 1),
                'acc': 100 * correct / total,
                'lr': optimizer.param_groups[0]['lr']
            })
        
        # 计算训练指标
        epoch_loss = running_loss / len(finetune_loader)
        epoch_acc = 100 * correct / total
        
        # 保存历史
        finetune_history['train_loss'].append(epoch_loss)
        finetune_history['train_acc'].append(epoch_acc)
        
        print(f"微调 Epoch {epoch+1}/{finetune_epochs} - 损失: {epoch_loss:.4f}, 准确率: {epoch_acc:.2f}%")
        
        # 学习率调度
        scheduler.step(epoch_loss)
    
    # 保存最终模型
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'pretrain_history': pretrain_history,
        'finetune_history': finetune_history,
        'intent_labels': intent_labels
    }, model_save_path)
    print(f"流式微调模型已保存到: {model_save_path}")
    
    # 绘制训练历史
    plt.figure(figsize=(15, 5))
    
    # 绘制损失
    plt.subplot(1, 2, 1)
    plt.plot(range(1, pretrain_epochs + 1), pretrain_history['train_loss'], 'b-', label='Pre-train Loss')
    plt.plot(range(pretrain_epochs + 1, pretrain_epochs + finetune_epochs + 1), 
             finetune_history['train_loss'], 'r-', label='Finetune Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.legend()
    
    # 绘制准确率
    plt.subplot(1, 2, 2)
    plt.plot(range(1, pretrain_epochs + 1), pretrain_history['train_acc'], 'b-', label='Pre-train Accuracy')
    plt.plot(range(pretrain_epochs + 1, pretrain_epochs + finetune_epochs + 1), 
             finetune_history['train_acc'], 'r-', label='Finetune Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Training Accuracy')
    plt.legend()
    
    # 保存图表
    plt.tight_layout()
    plt.savefig(f"{model_save_path.split('.')[0]}_history.png")
    plt.close()
    
    return model, intent_labels

def plot_training_curves(history, save_dir, model_type, augment=True):
    """绘制训练曲线并保存"""
    # 重置字体设置，使用系统默认字体
    plt.rcParams.update({'font.family': 'sans-serif'})
    plt.rcParams['axes.unicode_minus'] = False   # 正确显示负号
    
    # 创建图表
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
    
    # 绘制损失曲线
    ax1.plot(history['train_loss'])
    ax1.set_title('Model Loss')
    ax1.set_ylabel('Loss')
    ax1.set_xlabel('Epoch')
    ax1.grid(True)
    
    # 绘制准确率曲线
    ax2.plot(history['train_acc'])
    ax2.set_title('Model Accuracy')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_xlabel('Epoch')
    ax2.grid(True)
    
    plt.tight_layout()
    
    # 保存图表
    aug_str = "with_aug" if augment else "no_aug"
    plt.savefig(os.path.join(save_dir, f"{model_type}_training_curves_{aug_str}.png"))
    plt.close()

def export_model_to_onnx(model_path, model_type, onnx_save_path=None, dynamic_axes=True):
    """
    将PyTorch模型导出为ONNX格式
    
    Args:
        model_path: PyTorch模型路径
        model_type: 模型类型 ('fast')
        onnx_save_path: ONNX模型保存路径（如果为None则根据原模型路径生成）
        dynamic_axes: 是否使用动态轴（用于支持可变输入大小）
    
    Returns:
        onnx_save_path: 导出的ONNX模型路径
    """
    print(f"正在导出{model_type}模型到ONNX格式...")
    
    # 如果未指定ONNX保存路径，则根据PyTorch模型路径生成
    if onnx_save_path is None:
        onnx_save_path = os.path.splitext(model_path)[0] + '.onnx'
    
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 加载模型检查点
    checkpoint = torch.load(model_path, map_location=device)
    
    if isinstance(checkpoint, dict) and 'model_config' in checkpoint and checkpoint['model_config'].get('model_type') == 'streaming_conformer':
        # --- 处理 StreamingConformer 模型 --- 
        print("检测到 StreamingConformer 模型检查点...")
        model_config = checkpoint['model_config']
        model_state = checkpoint['model_state_dict']
        intent_labels = checkpoint['intent_labels']
        num_classes = model_config['num_classes']
        
        print(f"从配置加载模型参数: {model_config}")
        print(f"意图类别: {intent_labels}")
        
        # 使用保存的配置实例化 StreamingConformer
        model = StreamingConformer(
            input_dim=model_config['input_dim'],
            hidden_dim=model_config['hidden_dim'],
            num_classes=num_classes,
            num_layers=model_config['num_layers'],
            num_heads=model_config['num_heads'],
            dropout=model_config.get('dropout', 0.1), # 兼容旧模型
            kernel_size=model_config.get('kernel_size', 31), # 兼容旧模型
            expansion_factor=model_config.get('expansion_factor', 4) # 兼容旧模型
        )
        
        # 加载权重
        # 修正潜在的尺寸不匹配问题 (例如 pos_encoding.pe)
        # 在加载状态字典之前，检查并调整参数形状
        current_state_dict = model.state_dict()
        for name, param in model_state.items():
            if name in current_state_dict:
                if current_state_dict[name].shape != param.shape:
                    print(f"警告: 参数 '{name}' 形状不匹配。检查点: {param.shape}, 当前模型: {current_state_dict[name].shape}. 尝试调整...")
                    # 特别处理位置编码 (如果需要)
                    if name == 'pos_encoding.pe' and len(param.shape) == 3 and len(current_state_dict[name].shape) == 3:
                         # 假设 batch 和 dim 维度匹配，只调整长度维度
                         if param.shape[0] == current_state_dict[name].shape[0] and param.shape[2] == current_state_dict[name].shape[2]:
                             max_len_checkpoint = param.shape[1]
                             max_len_current = current_state_dict[name].shape[1]
                             # 从检查点复制尽可能多的位置编码
                             copy_len = min(max_len_checkpoint, max_len_current)
                             current_state_dict[name][:, :copy_len, :] = param[:, :copy_len, :]
                             print(f"已调整 '{name}' 的形状。")
                             model_state[name] = current_state_dict[name] # 使用调整后的参数
                         else:
                             print(f"无法自动调整 '{name}'，维度不匹配。跳过加载此参数。")
                             # 从 model_state 中移除，避免加载错误
                             model_state.pop(name)
                    else:
                        print(f"无法自动调整 '{name}'，跳过加载此参数。")
                        # 从 model_state 中移除
                        model_state.pop(name)
            else:
                print(f"警告: 在检查点中找到的参数 '{name}' 不在当前模型中。忽略。")

        # 加载可能已修改的状态字典
        model.load_state_dict(model_state, strict=False) # 使用 strict=False 忽略不匹配的键
        print("模型权重加载完成 (strict=False)")
        model.to(device)
        model.eval()
        
        # 创建流式模型的示例输入 (固定 batch_size=1, chunk_length=STREAMING_CHUNK_SIZE)
        dummy_input_chunk = torch.randn(1, STREAMING_CHUNK_SIZE, model_config['input_dim'], device=device)
        
        input_names = ["input_chunk"]
        output_names = ["output_logits"]
        
        # 导出核心模型逻辑（forward）
        print(f"导出 StreamingConformer (固定形状 [1, {STREAMING_CHUNK_SIZE}, {model_config['input_dim']}]) 到 {onnx_save_path}")
        torch.onnx.export(
            model, 
            dummy_input_chunk, 
            onnx_save_path,
            input_names=input_names,
            output_names=output_names,
            export_params=True,
            opset_version=13,
            do_constant_folding=True,
            verbose=False
        )

    elif model_type == 'fast' or (isinstance(checkpoint, dict) and checkpoint.get('model_config', {}).get('model_type') != 'streaming_conformer'):
        # --- 处理 FastIntentClassifier 模型 (保持原有逻辑) ---
        print("处理 FastIntentClassifier 模型...")
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            # ... (加载新格式 Fast模型的代码不变) ...
            model_state = checkpoint['model_state_dict']
            intent_labels = checkpoint['intent_labels']
            num_classes = len(intent_labels)
            input_size = checkpoint.get('input_size', N_MFCC * 3)
            if 'input_projection.weight' in model_state:
                _, actual_input_size = model_state['input_projection.weight'].shape
                if actual_input_size != input_size:
                    input_size = actual_input_size
        else:
            # ... (加载旧格式 Fast模型的代码不变) ...
            model_state = checkpoint
            num_classes = len(INTENT_CLASSES) # 假设旧模型使用 config 中的 INTENT_CLASSES
            if isinstance(model_state, dict) and 'input_projection.weight' in model_state:
                 _, input_size = model_state['input_projection.weight'].shape
            else:
                input_size = N_MFCC * 3
        
        print(f"使用input_size={input_size}创建模型")
        model = FastIntentClassifier(input_size=input_size, num_classes=num_classes)
        model.load_state_dict(model_state)
        model.to(device)
        model.eval()
        
        dummy_input = torch.randn(1, 500, input_size, device=device) # 保持 Fast 模型的输入
        input_names = ["input"]
        output_names = ["output"]
        dynamic_axes_dict = None
        if dynamic_axes:
            dynamic_axes_dict = {
                'input': {0: 'batch_size', 1: 'sequence_length'},
                'output': {0: 'batch_size'}
            }
            
        print(f"导出 FastIntentClassifier 到 {onnx_save_path}")
        torch.onnx.export(
            model,
            dummy_input,
            onnx_save_path,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes_dict,
            export_params=True,
            opset_version=13,
            do_constant_folding=True,
            verbose=False
        )
        
    else:
         raise ValueError(f"无法确定模型类型或不受支持的检查点格式: {model_path}")

    print(f"模型已导出至: {onnx_save_path}")
    return onnx_save_path

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="EdgeVoice语音命令分类模型训练")
    parser.add_argument("--model_type", type=str, default="fast", choices=["fast", "streaming", "transformer"], 
                     help="模型类型: fast, streaming, transformer")
    parser.add_argument("--data_dir", type=str, default=DATA_DIR, help="音频数据目录")
    parser.add_argument("--annotation_file", type=str, required=True, help="训练数据标注文件")
    parser.add_argument("--model_save_path", type=str, required=True, help="模型保存路径")
    parser.add_argument("--num_epochs", type=int, default=NUM_EPOCHS, help="训练轮数")
    parser.add_argument("--augment", action="store_true", help="是否使用数据增强")
    parser.add_argument("--augment_prob", type=float, default=0.5, help="增强概率")
    parser.add_argument("--pre_train_epochs", type=int, default=10, help="预训练轮数(仅用于streaming模型)")
    parser.add_argument("--fine_tune_epochs", type=int, default=10, help="微调轮数(仅用于streaming模型)")
    parser.add_argument("--export_onnx", action="store_true", help="是否导出为ONNX格式")
    parser.add_argument("--onnx_save_path", type=str, help="ONNX模型保存路径")
    parser.add_argument("--clear_cache", action="store_true", help="是否清理特征缓存")
    parser.add_argument("--use_cache", action="store_true", default=True, help="是否使用特征缓存")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--dynamic_axes", action="store_true", default=True, help="使用动态轴(支持可变输入大小)")
    
    return parser.parse_args()

def main():
    """主函数"""
    # 解析命令行参数
    args = parse_args()
    
    # 清理缓存目录
    if args.clear_cache:
        cache_dir = os.path.join("tmp", "feature_cache")
        if os.path.exists(cache_dir):
            print(f"清理特征缓存目录: {cache_dir}")
            shutil.rmtree(cache_dir)
            os.makedirs(cache_dir, exist_ok=True)
    
    # 训练模型
    print(f"训练模型类型: {args.model_type}")
    print(f"数据目录: {args.data_dir}")
    print(f"标注文件: {args.annotation_file}")
    print(f"模型保存路径: {args.model_save_path}")
    
    # 根据模型类型选择训练函数
    if args.model_type == "fast":
        model, intent_labels = train_fast_model(
            data_dir=args.data_dir,
            annotation_file=args.annotation_file,
            model_save_path=args.model_save_path,
            num_epochs=args.num_epochs,
            augment=args.augment,
            augment_prob=args.augment_prob,
            use_cache=args.use_cache,
            seed=args.seed
        )
    elif args.model_type == "streaming":
        model, intent_labels = train_streaming_model(
            data_dir=args.data_dir,
            annotation_file=args.annotation_file,
            model_save_path=args.model_save_path,
            num_epochs=args.num_epochs,
            pretrain_epochs=args.pre_train_epochs,
            finetune_epochs=args.fine_tune_epochs,
            use_random_crop=args.augment,
            use_cache=args.use_cache,
            seed=args.seed
        )
    else:
        raise ValueError(f"不支持的模型类型: {args.model_type}")
    
    # 导出ONNX模型
    if args.export_onnx:
        print("\n开始导出模型为ONNX格式...")
        export_model_to_onnx(
            model_path=args.model_save_path,
            model_type=args.model_type,
            onnx_save_path=args.onnx_save_path,
            dynamic_axes=args.dynamic_axes
        )
    
    print("训练完成！")

def evaluate_streaming_model(model, test_dataframe, device, save_path, 
                        early_stopping_conf=0.8, weighted_majority=True):
    """
    评估流式模型，并保存评估结果
    
    Args:
        model: 流式分类模型
        test_dataframe: 测试数据DataFrame
        device: 设备
        save_path: 保存评估结果的路径
        early_stopping_conf: 提前停止的置信度阈值
        weighted_majority: 是否使用加权多数投票
        
    Returns:
        accuracy: 准确率
    """
    try:
        # 设置模型为评估模式
        model.eval()
        
        # 初始化评估指标
        true_labels = []
        pred_labels = []
        latencies = []
        early_stopping_stats = []
        processing_errors = 0
        
        # 遍历测试数据
        for idx, row in test_dataframe.iterrows():
            file_path = row["file_path"]
            label = row["intent"]
            
            try:
                # 加载音频
                audio, sr = librosa.load(file_path, sr=TARGET_SAMPLE_RATE)
                
                # 提取chunk特征
                chunk_features, _ = streaming_feature_extractor(audio, sr, chunk_size, step_size)
                
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
                    features_tensor = torch.FloatTensor(features).unsqueeze(0).to(device)
                    
                    with torch.no_grad():
                        # 使用流式前向传播
                        pred, conf, cached_states = model.predict_streaming(features_tensor, cached_states)
                        
                        # 记录预测结果
                        pred_label = pred.item()
                        confidence = conf.item()
                        file_predictions.append((pred_label, confidence))
                        
                        # 计算当前时间点
                        current_time = (i * step_size * HOP_LENGTH) / sr
                        
                        # 检查是否满足早停条件
                        if confidence > early_stopping_conf:
                            decision_time = current_time
                            early_stopped = True
                            break
                
                # 如果没有预测，跳过
                if len(file_predictions) == 0:
                    print(f"警告: 文件没有生成预测: {file_path}, 跳过评估")
                    continue
                
                # 计算潜伏期（从开始到决策的时间）
                latency = decision_time
                latencies.append(latency)
                early_stopping_stats.append(1 if early_stopped else 0)
                
                # 确定最终预测结果
                if weighted_majority and len(file_predictions) > 1:
                    # 加权多数投票 (按置信度加权)
                    label_votes = {}
                    for pred_label, confidence in file_predictions:
                        if pred_label not in label_votes:
                            label_votes[pred_label] = 0
                        label_votes[pred_label] += confidence
                    
                    final_pred = max(label_votes.items(), key=lambda x: x[1])[0]
                else:
                    # 使用最后一个预测
                    final_pred = file_predictions[-1][0]
                
                # 记录真实标签和预测标签
                true_labels.append(label)
                pred_labels.append(final_pred)
                
            except Exception as e:
                print(f"处理文件时出错: {file_path}, 错误: {str(e)}")
                processing_errors += 1
                continue
        
        # 计算准确率
        if len(true_labels) == 0:
            print("警告: 没有成功处理任何测试文件，无法计算准确率")
            return 0.0
            
        accuracy = accuracy_score(true_labels, pred_labels)
        
        # 计算平均潜伏期
        avg_latency = np.mean(latencies) if latencies else float('nan')
        early_stopping_rate = np.mean(early_stopping_stats) if early_stopping_stats else float('nan')
        
        # 打印评估结果
        print(f"流式评估结果:")
        print(f"  准确率: {accuracy:.4f}")
        print(f"  平均潜伏期: {avg_latency:.4f}秒")
        print(f"  早停率: {early_stopping_rate:.4f}")
        print(f"  处理错误数: {processing_errors}")
        
        # 保存评估结果
        eval_results = {
            "accuracy": accuracy,
            "avg_latency": avg_latency,
            "early_stopping_rate": early_stopping_rate,
            "processing_errors": processing_errors,
            "true_labels": true_labels,
            "pred_labels": pred_labels,
            "latencies": latencies
        }
        
        # 确保目录存在
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # 保存评估结果
        with open(save_path, 'wb') as f:
            pickle.dump(eval_results, f)
        
        return accuracy
        
    except Exception as e:
        print(f"评估过程中出错: {str(e)}")
        traceback.print_exc()
        return 0.0

if __name__ == "__main__":
    main()