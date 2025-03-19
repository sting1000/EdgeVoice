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
            
            # 前向传播
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
    # 准备流式数据加载器
    finetune_loader, _ = prepare_streaming_dataloader(
        annotation_file=annotation_file,
        data_dir=data_dir,
        batch_size=BATCH_SIZE,
        streaming_mode=True,  # 流式模式
        use_random_crop=False,  # 流式模式不需要随机裁剪
        cache_dir=cache_dir,
        shuffle=True,
        seed=seed
    )
    
    # 重置优化器
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE * 0.1)  # 降低学习率
    
    # 微调历史记录
    finetune_history = {
        'train_loss': [],
        'train_acc': []
    }
    
    # 微调循环
    for epoch in range(finetune_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        # 使用tqdm进度条
        progress_bar = tqdm(finetune_loader, desc=f"微调 Epoch {epoch+1}/{finetune_epochs}")
        
        for features, labels in progress_bar:
            features, labels = features.to(device), labels.to(device)
            
            # 清零梯度
            optimizer.zero_grad()
            
            # 前向传播
            outputs = model(features)
            
            # 确保输出和标签形状匹配
            if outputs.shape[0] != labels.shape[0]:
                print(f"警告: 输出形状 {outputs.shape} 与标签形状 {labels.shape} 不匹配")
                continue
            
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
        epoch_loss = running_loss / len(finetune_loader)
        epoch_acc = 100 * correct / total
        
        # 保存历史
        finetune_history['train_loss'].append(epoch_loss)
        finetune_history['train_acc'].append(epoch_acc)
        
        print(f"微调 Epoch {epoch+1}/{finetune_epochs} - 损失: {epoch_loss:.4f}, 准确率: {epoch_acc:.2f}%")
    
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
    
    # 加载模型
    checkpoint = torch.load(model_path, map_location=device)
    
    # 检查是否是新的保存格式
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        print("检测到新的模型保存格式，包含标签映射和状态字典")
        model_state = checkpoint['model_state_dict']
        intent_labels = checkpoint['intent_labels']
        num_classes = len(intent_labels)
        print(f"模型包含 {num_classes} 个意图类别: {intent_labels}")
        
        # 获取input_size (对于fast模型)
        input_size = checkpoint.get('input_size', N_MFCC * 3)  # 使用N_MFCC * 3作为默认值，通常是48维
        
        # 检查模型状态字典中的参数形状，以确定真实的input_size
        if 'input_projection.weight' in model_state:
            # 从输入投影矩阵的形状推断input_size
            _, actual_input_size = model_state['input_projection.weight'].shape
            if actual_input_size != input_size:
                print(f"警告: 从保存的状态字典中检测到不同的input_size: {actual_input_size}，将使用此值")
                input_size = actual_input_size
    else:
        print("检测到旧的模型保存格式，仅包含状态字典")
        model_state = checkpoint
        num_classes = len(INTENT_CLASSES)
        # 从模型权重中推断input_size
        if isinstance(model_state, dict) and 'input_projection.weight' in model_state:
            _, input_size = model_state['input_projection.weight'].shape
            print(f"从模型权重中推断input_size: {input_size}")
        else:
            input_size = N_MFCC * 3  # 默认值
    
    print(f"使用input_size={input_size}创建模型")
    
    # 加载FastIntentClassifier模型
    print(f"创建新的FastIntentClassifier实例，输入维度: {input_size}，类别数: {num_classes}")
    model = FastIntentClassifier(input_size=input_size, num_classes=num_classes)
    
    # 加载保存的权重
    print("加载模型权重")
    model.load_state_dict(model_state)
    model.to(device)
    model.eval()
    
    # 创建示例输入（支持批量输入和单个输入）
    dummy_input = torch.randn(1, 500, input_size, device=device)  # 批量大小为1，序列长度为500
    
    # 定义ONNX导出的输入和输出名称
    input_names = ["input"]
    output_names = ["output"]
    
    # 定义动态轴（如果需要）
    dynamic_axes_dict = None
    if dynamic_axes:
        dynamic_axes_dict = {
            'input': {0: 'batch_size', 1: 'sequence_length'},  # 动态批量大小和序列长度
            'output': {0: 'batch_size'}
        }
    
    # 导出模型
    print(f"导出模型到 {onnx_save_path}")
    torch.onnx.export(
        model,
        dummy_input,
        onnx_save_path,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes_dict,
        export_params=True,
        opset_version=13,  # 使用更高版本的opset支持更多操作
        do_constant_folding=True,
        verbose=False
    )
    
    print(f"模型已导出至: {onnx_save_path}")
    return onnx_save_path

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="训练音频意图识别模型")
    parser.add_argument("--data_dir", type=str, default=DATA_DIR, help="数据目录")
    parser.add_argument("--annotation_file", type=str, required=True, help="标注文件")
    parser.add_argument("--model_type", type=str, choices=["fast", "streaming"], default="fast", help="模型类型")
    parser.add_argument("--model_save_path", type=str, required=True, help="模型保存路径")
    parser.add_argument("--num_epochs", type=int, default=NUM_EPOCHS, help="训练轮数")
    parser.add_argument("--pre_train_epochs", type=int, default=10, help="预训练轮数")
    parser.add_argument("--fine_tune_epochs", type=int, default=10, help="微调轮数")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--augment", action="store_true", help="是否使用数据增强")
    parser.add_argument("--augment_prob", type=float, default=0.5, help="数据增强概率")
    parser.add_argument("--use_cache", action="store_true", help="是否使用特征缓存")
    parser.add_argument("--export_onnx", action="store_true", help="训练完成后导出为ONNX格式")
    parser.add_argument("--onnx_save_path", type=str, default=None, help="ONNX模型保存路径")
    parser.add_argument("--dynamic_axes", action="store_true", default=True, help="使用动态轴(支持可变输入大小)")
    
    return parser.parse_args()

def main():
    """主函数，处理命令行参数并执行训练"""
    args = parse_args()
    
    print(f"训练模型类型: {args.model_type}")
    print(f"数据目录: {args.data_dir}")
    print(f"标注文件: {args.annotation_file}")
    print(f"模型保存路径: {args.model_save_path}")
    
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

if __name__ == "__main__":
    main()