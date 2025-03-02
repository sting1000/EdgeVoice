# train.py  
import os  
import argparse  
import torch  
import torch.nn as nn  
import torch.optim as optim  
import numpy as np  
from tqdm import tqdm  
from sklearn.metrics import classification_report, accuracy_score  
from config import *  
from data_utils import prepare_dataloader  
from models.fast_classifier import FastIntentClassifier  
from models.precise_classifier import PreciseIntentClassifier  

def train_fast_model(data_dir, annotation_file, model_save_path, num_epochs=NUM_EPOCHS):  
    """训练一级快速分类器"""  
    print("准备数据...")  
    train_loader = prepare_dataloader(data_dir, annotation_file, mode='fast')  
    
    # 样本数据批次用于确定输入大小  
    for features, _ in train_loader:  
        input_size = features.size(2)  
        break  
    
    print(f"创建模型(输入大小: {input_size})...")  
    model = FastIntentClassifier(input_size=input_size)  
    model = model.to(DEVICE)  
    
    # 损失函数和优化器  
    criterion = nn.CrossEntropyLoss()  
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)  
    
    # 训练循环  
    print("开始训练...")  
    for epoch in range(num_epochs):  
        model.train()  
        running_loss = 0.0  
        all_preds = []  
        all_labels = []  
        
        for features, labels in tqdm(train_loader):  
            features, labels = features.to(DEVICE), labels.to(DEVICE)  
            
            # 梯度清零  
            optimizer.zero_grad()  
            
            # 前向传播  
            outputs = model(features)  
            loss = criterion(outputs, labels)  
            
            # 反向传播和优化  
            loss.backward()  
            optimizer.step()  
            
            # 统计  
            running_loss += loss.item()  
            _, preds = torch.max(outputs, 1)  
            all_preds.extend(preds.cpu().numpy())  
            all_labels.extend(labels.cpu().numpy())  
        
        # 计算准确率  
        accuracy = accuracy_score(all_labels, all_preds)  
        
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_loader):.4f}, Accuracy: {accuracy:.4f}')  
        
        # 每个epoch保存一次模型  
        torch.save(model.state_dict(), model_save_path)  
        
    print("训练完成!")  
    print("分类报告:")  
    print(classification_report(all_labels, all_preds, target_names=INTENT_CLASSES))  
    
    return model  

def train_precise_model(data_dir, annotation_file, model_save_path, num_epochs=NUM_EPOCHS):  
    """训练二级精确分类器"""  
    print("准备数据...")  
    train_loader = prepare_dataloader(data_dir, annotation_file, mode='precise')  
    
    print("创建模型...")  
    model = PreciseIntentClassifier()  
    model = model.to(DEVICE)  
    
    # 损失函数和优化器  
    criterion = nn.CrossEntropyLoss()  
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)  
    
    # 训练循环  
    print("开始训练...")  
    for epoch in range(num_epochs):  
        model.train()  
        running_loss = 0.0  
        all_preds = []  
        all_labels = []  
        
        for batch in tqdm(train_loader):  
            input_ids = batch['input_ids'].to(DEVICE)  
            attention_mask = batch['attention_mask'].to(DEVICE)  
            labels = batch['label'].to(DEVICE)  
            
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
            all_preds.extend(preds.cpu().numpy())  
            all_labels.extend(labels.cpu().numpy())  
        
        # 计算准确率  
        accuracy = accuracy_score(all_labels, all_preds)  
        
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_loader):.4f}, Accuracy: {accuracy:.4f}')  
        
        # 每个epoch保存一次模型  
        torch.save(model.state_dict(), model_save_path)  
        
    print("训练完成!")  
    print("分类报告:")  
    print(classification_report(all_labels, all_preds, target_names=INTENT_CLASSES))  
    
    return model  

if __name__ == "__main__":  
    parser = argparse.ArgumentParser(description='训练语音意图识别模型')  
    parser.add_argument('--data_dir', type=str, default=DATA_DIR, help='数据目录')  
    parser.add_argument('--annotation_file', type=str, required=True, help='注释文件路径')  
    parser.add_argument('--model_type', type=str, choices=['fast', 'precise'], required=True, help='模型类型')  
    parser.add_argument('--epochs', type=int, default=NUM_EPOCHS, help='训练轮数')  
    args = parser.parse_args()  
    
    model_save_path = os.path.join(MODEL_DIR, f"{args.model_type}_intent_model.pth")  
    
    if args.model_type == 'fast':  
        train_fast_model(args.data_dir, args.annotation_file, model_save_path, args.epochs)  
    else:  
        train_precise_model(args.data_dir, args.annotation_file, model_save_path, args.epochs)