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
from transformers import DistilBertTokenizer
import time
import random
from config import *  
# 使用增强版数据加载器替换原始版本
# from data_utils import prepare_dataloader  
from augmented_dataset import prepare_augmented_dataloader, standardize_audio_length
from models.fast_classifier import FastIntentClassifier  
from models.precise_classifier import PreciseIntentClassifier
import librosa
import torch.onnx

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
        mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
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
        return np.zeros((100, 39))  # 默认100帧，39个特征

def process_audio_for_precise(audio, sr, tokenizer, transcript=None):
    """处理音频数据，为精确分类器准备特征
    
    参数:
        audio: 音频数据
        sr: 采样率
        tokenizer: 分词器
        transcript: 文本转录(可选)
        
    返回:
        特征字典，包含input_ids和attention_mask
    """
    try:
        # 使用提供的文本转录或默认文本
        if transcript is None or transcript == "":
            # 在实际场景中，此处应调用ASR模型获取文本
            text = "未提供转录文本"
        else:
            text = transcript
        
        # 使用分词器处理文本
        encoding = tokenizer(
            text,
            padding='max_length',
            truncation=True,
            max_length=128,
            return_tensors='pt'
        )
        
        # 提取所需的特征并转换为numpy数组
        # 注意：这里我们需要将PyTorch tensor转换为numpy数组
        # 因为Dataset.__getitem__返回的数据会被collate_fn处理
        return {
            'input_ids': encoding['input_ids'].squeeze(0).numpy(),
            'attention_mask': encoding['attention_mask'].squeeze(0).numpy()
        }
    except Exception as e:
        print(f"文本特征提取错误: {e}")
        # 返回安全的默认值
        return {
            'input_ids': np.zeros((128,), dtype=np.int64),
            'attention_mask': np.zeros((128,), dtype=np.int64)
        }

def train_fast_model(data_dir, annotation_file, model_save_path, num_epochs=NUM_EPOCHS, 
                    augment=True, augment_prob=0.5, use_cache=True, seed=42):
    """训练一级快速分类器（支持数据增强）"""
    # 设置随机种子
    set_seed(seed)

    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    print("准备数据...")  
    # 使用增强版数据加载器
    train_loader, intent_labels = prepare_augmented_dataloader(
        annotation_file=annotation_file,
        data_dir=data_dir,
        batch_size=BATCH_SIZE,
        feature_extractor=lambda audio, sr, **kwargs: extract_features(audio, sr),  # 添加feature_extractor
        augment=augment, 
        augment_prob=augment_prob,
        use_cache=use_cache,
        shuffle=True
    )
    
    # 获取输入特征大小
    batch = next(iter(train_loader))
    features = batch['features']
    print(f"特征形状: {features.shape}")
    
    # 确保特征维度正确
    if features.dim() == 2:  # (batch_size, feature_dim)
        input_size = features.size(1)
    elif features.dim() == 3:  # (batch_size, seq_len, feature_dim)
        input_size = features.size(2)
    else:
        raise ValueError(f"特征维度错误: {features.dim()}, 形状: {features.shape}")
    
    num_classes = len(intent_labels)
    
    print(f"创建Conformer模型(输入大小: {input_size}, 隐藏大小: {FAST_MODEL_HIDDEN_SIZE}, 类别数: {num_classes})...")  
    print(f"  - 层数: {CONFORMER_LAYERS}")
    print(f"  - 注意力头数: {CONFORMER_ATTENTION_HEADS}")
    print(f"  - 卷积核大小: {CONFORMER_CONV_KERNEL_SIZE}")
    print(f"  - 前馈网络扩展因子: {CONFORMER_FF_EXPANSION_FACTOR}")
    model = FastIntentClassifier(input_size=input_size, hidden_size=FAST_MODEL_HIDDEN_SIZE, num_classes=num_classes)  
    model = model.to(device)  
    
    # 损失函数和优化器  
    criterion = nn.CrossEntropyLoss()  
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)  
    
    # 记录训练历史  
    history = {
        'train_loss': [],
        'train_acc': []
    }
    
    # 训练循环  
    print("开始训练...")  
    start_time = time.time()
    
    for epoch in range(num_epochs):  
        model.train()  
        running_loss = 0.0  
        correct = 0
        total = 0
        
        # 使用tqdm进度条
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        
        for batch in progress_bar:  
            # 准备数据（适配新的数据结构）
            inputs = batch['features'].to(device)
            labels = batch['label'].to(device)
            
            # 确保输入形状正确 - 如果是2D张量，转换为3D
            if inputs.dim() == 2:
                # 添加序列维度 (batch_size, feature_dim) -> (batch_size, 1, feature_dim)
                inputs = inputs.unsqueeze(1)
            
            # 梯度清零  
            optimizer.zero_grad()  
            
            # 前向传播  
            outputs = model(inputs)  
            loss = criterion(outputs, labels)  
            
            # 反向传播和优化  
            loss.backward()  
            optimizer.step()  
            
            # 统计  
            running_loss += loss.item()  
            _, preds = torch.max(outputs, 1)  
            total += labels.size(0)
            correct += (preds == labels).sum().item()
            
            # 更新进度条
            progress_bar.set_postfix({
                'loss': running_loss / (progress_bar.n + 1),
                'acc': 100 * correct / total
            })
        
        # 计算本轮平均损失和准确率
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100 * correct / total
        
        # 记录历史
        history['train_loss'].append(epoch_loss)
        history['train_acc'].append(epoch_acc)
        
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}')  
        
        # 每个epoch保存一次模型  
        torch.save(model.state_dict(), model_save_path)  
    
    # 计算总训练时间
    total_time = time.time() - start_time
    print(f"训练完成! 总时间: {total_time:.2f}秒")
    
    print("分类报告:")  
    print(classification_report(labels.cpu().numpy(), preds.cpu().numpy(), target_names=INTENT_CLASSES))  
    
    # 绘制训练曲线
    plot_training_curves(history, os.path.dirname(model_save_path), "fast", augment)
    
    return model  

def train_precise_model(data_dir, annotation_file, model_save_path, num_epochs=NUM_EPOCHS,
                       augment=True, augment_prob=0.5, use_cache=True, seed=42):
    """训练二级精确分类器（支持数据增强）"""
    # 设置随机种子
    set_seed(seed)

    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    print("准备数据...")  
    # 加载分词器
    try:
        tokenizer = DistilBertTokenizer.from_pretrained(DISTILBERT_MODEL_PATH)
        print(f"已从本地路径加载分词器: {DISTILBERT_MODEL_PATH}")
    except:
        tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        print("已从在线资源加载分词器")
        
    # 使用增强版数据加载器
    train_loader, intent_labels = prepare_augmented_dataloader(
        annotation_file=annotation_file,
        data_dir=data_dir,
        batch_size=BATCH_SIZE,
        feature_extractor=lambda audio, sr, transcript=None: process_audio_for_precise(audio, sr, tokenizer, transcript),
        augment=augment,
        augment_prob=augment_prob,
        use_cache=use_cache,
        shuffle=True
    )
    
    # 获取类别数
    num_classes = len(intent_labels)
    print(f"创建模型(类别数: {num_classes})...")
    
    # 初始化模型
    try:
        model = PreciseIntentClassifier(
            num_classes=num_classes,  # 使用从数据加载器获取的类别数
            pretrained_path=DISTILBERT_MODEL_PATH
        )
        print(f"已从本地路径初始化精确分类器: {DISTILBERT_MODEL_PATH}")
    except:
        model = PreciseIntentClassifier(num_classes=len(INTENT_CLASSES))
        print("已从在线资源初始化精确分类器")
    
    model = model.to(device)  
    
    # 损失函数和优化器  
    criterion = nn.CrossEntropyLoss()  
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)  
    
    # 记录训练历史  
    history = {
        'train_loss': [],
        'train_acc': []
    }
    
    # 训练循环  
    print("开始训练...")  
    start_time = time.time()
    
    for epoch in range(num_epochs):  
        model.train()  
        running_loss = 0.0  
        correct = 0
        total = 0
        
        # 使用tqdm进度条
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        
        for batch in progress_bar:
            # 准备数据（适配新的数据结构）
            features = batch['features']  # 这里features是一个字典
            input_ids = features['input_ids'].to(device)  
            attention_mask = features['attention_mask'].to(device)  
            labels = batch['label'].to(device)
            
            # 梯度清零  
            optimizer.zero_grad()  
            
            # 前向传播  
            outputs = model(input_ids, attention_mask)  
            loss = criterion(outputs, labels)  
            
            # 反向传播和优化  
            loss.backward()  
            optimizer.step()  
            
            # 统计  
            running_loss += loss.item()  
            _, preds = torch.max(outputs, 1)  
            total += labels.size(0)
            correct += (preds == labels).sum().item()
            
            # 更新进度条
            progress_bar.set_postfix({
                'loss': running_loss / (progress_bar.n + 1),
                'acc': 100 * correct / total
            })
        
        # 计算本轮平均损失和准确率
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100 * correct / total
        
        # 记录历史
        history['train_loss'].append(epoch_loss)
        history['train_acc'].append(epoch_acc)
        
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}')  
        
        # 每个epoch保存一次模型  
        torch.save(model.state_dict(), model_save_path)  
    
    # 计算总训练时间
    total_time = time.time() - start_time
    print(f"训练完成! 总时间: {total_time:.2f}秒")
    
    print("分类报告:")  
    print(classification_report(labels.cpu().numpy(), preds.cpu().numpy(), target_names=INTENT_CLASSES))  
    
    # 绘制训练曲线
    plot_training_curves(history, os.path.dirname(model_save_path), "precise", augment)
    
    return model 

def plot_training_curves(history, save_dir, model_type, augment=True):
    """绘制训练曲线并保存"""
    # 创建图表
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
    
    # 绘制损失曲线
    ax1.plot(history['train_loss'])
    ax1.set_title('模型损失')
    ax1.set_ylabel('损失')
    ax1.set_xlabel('Epoch')
    ax1.grid(True)
    
    # 绘制准确率曲线
    ax2.plot(history['train_acc'])
    ax2.set_title('模型准确率')
    ax2.set_ylabel('准确率 (%)')
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
        model_type: 模型类型 ('fast' 或 'precise')
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
    
    if model_type == 'fast':
        # 加载FastIntentClassifier模型
        # 注意: 由于缺少input_size参数，我们需要先创建一个示例模型
        # 我们使用MFCC特征加上上下文帧，所以特征大小为(N_MFCC * (2*CONTEXT_FRAMES + 1))
        input_size = N_MFCC * (2*CONTEXT_FRAMES + 1)
        model = FastIntentClassifier(input_size=input_size, num_classes=len(INTENT_CLASSES))
        
        # 加载保存的权重
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()
        
        # 创建示例输入
        dummy_input = torch.randn(1, 1, input_size, device=device)  # 批量大小为1，序列长度为1
        
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
        torch.onnx.export(
            model,
            dummy_input,
            onnx_save_path,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes_dict,
            export_params=True,
            opset_version=12,
            do_constant_folding=True,
            verbose=False
        )
    
    elif model_type == 'precise':
        # 加载PreciseIntentClassifier模型
        model = PreciseIntentClassifier(num_classes=len(INTENT_CLASSES))
        
        # 加载保存的权重
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()
        
        # 创建示例输入
        # 假设我们使用的最大序列长度为128
        max_length = 128
        dummy_input_ids = torch.randint(0, 30522, (1, max_length), device=device)  # 批量大小为1，序列长度为max_length
        dummy_attention_mask = torch.ones((1, max_length), device=device)
        
        # 定义ONNX导出的输入和输出名称
        input_names = ["input_ids", "attention_mask"]
        output_names = ["output"]
        
        # 定义动态轴（如果需要）
        dynamic_axes_dict = None
        if dynamic_axes:
            dynamic_axes_dict = {
                'input_ids': {0: 'batch_size', 1: 'sequence_length'},
                'attention_mask': {0: 'batch_size', 1: 'sequence_length'},
                'output': {0: 'batch_size'}
            }
        
        # 导出模型
        torch.onnx.export(
            model,
            (dummy_input_ids, dummy_attention_mask),
            onnx_save_path,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes_dict,
            export_params=True,
            opset_version=12,
            do_constant_folding=True,
            verbose=False
        )
    
    else:
        raise ValueError(f"不支持的模型类型: {model_type}")
    
    print(f"模型已导出至: {onnx_save_path}")
    return onnx_save_path

if __name__ == "__main__":  
    parser = argparse.ArgumentParser(description='训练语音意图识别模型')  
    parser.add_argument('--data_dir', type=str, default=DATA_DIR, help='数据目录')  
    parser.add_argument('--annotation_file', type=str, required=True, help='注释文件路径')  
    parser.add_argument('--model_type', type=str, choices=['fast', 'precise'], required=True, help='模型类型')  
    parser.add_argument('--epochs', type=int, default=NUM_EPOCHS, help='训练轮数')  
    parser.add_argument('--augment', action='store_true', default=True, help='是否启用数据增强')
    parser.add_argument('--augment_prob', type=float, default=0.5, help='数据增强概率')
    parser.add_argument('--use_cache', action='store_true', default=True, help='是否缓存音频数据')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    parser.add_argument('--export_onnx', action='store_true', help='是否导出为ONNX模型')
    parser.add_argument('--onnx_save_path', type=str, default=None, help='ONNX模型保存路径')
    args = parser.parse_args()  
    
    # 确保模型目录存在
    os.makedirs(MODEL_DIR, exist_ok=True)
    model_save_path = os.path.join(MODEL_DIR, f"{args.model_type}_intent_model.pth")  
    
    # 训练模型
    if args.model_type == 'fast':  
        model = train_fast_model(
            args.data_dir, 
            args.annotation_file, 
            model_save_path, 
            args.epochs,
            args.augment,
            args.augment_prob,
            args.use_cache,
            args.seed
        )  
    else:  
        model = train_precise_model(
            args.data_dir, 
            args.annotation_file, 
            model_save_path, 
            args.epochs,
            args.augment,
            args.augment_prob,
            args.use_cache,
            args.seed
        )
    
    # 导出ONNX模型（如果需要）
    if args.export_onnx:
        export_model_to_onnx(model_save_path, args.model_type, args.onnx_save_path)