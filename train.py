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
from augmented_dataset import prepare_augmented_dataloader
from models.fast_classifier import FastIntentClassifier  
from models.precise_classifier import PreciseIntentClassifier

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
    train_loader = prepare_augmented_dataloader(
        annotation_file=annotation_file,
        data_dir=data_dir,
        batch_size=BATCH_SIZE,
        augment=augment, 
        augment_prob=augment_prob,
        use_cache=use_cache,
        shuffle=True
    )
    
    # 获取输入特征大小
    batch = next(iter(train_loader))
    input_size = batch['audio_feature'].size(2)
    
    print(f"创建模型(输入大小: {input_size})...")  
    model = FastIntentClassifier(input_size=input_size)  
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
            inputs = batch['audio_feature'].to(device)
            labels = batch['intent_label'].to(device)
            
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
    train_loader = prepare_augmented_dataloader(
        annotation_file=annotation_file,
        data_dir=data_dir,
        batch_size=BATCH_SIZE,
        tokenizer=tokenizer,
        augment=augment,
        augment_prob=augment_prob,
        use_cache=use_cache,
        shuffle=True
    )
    
    print("创建模型...")  
    # 初始化模型
    try:
        model = PreciseIntentClassifier(
            num_classes=len(INTENT_CLASSES),
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
            input_ids = batch['input_ids'].to(device)  
            attention_mask = batch['attention_mask'].to(device)  
            labels = batch['intent_label'].to(device)
            
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
    args = parser.parse_args()  
    
    # 确保模型目录存在
    os.makedirs(MODEL_DIR, exist_ok=True)
    model_save_path = os.path.join(MODEL_DIR, f"{args.model_type}_intent_model.pth")  
    
    if args.model_type == 'fast':  
        train_fast_model(
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
        train_precise_model(
            args.data_dir, 
            args.annotation_file, 
            model_save_path, 
            args.epochs,
            args.augment,
            args.augment_prob,
            args.use_cache,
            args.seed
        )