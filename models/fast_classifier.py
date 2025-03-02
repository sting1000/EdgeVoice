# models/fast_classifier.py  
import torch  
import torch.nn as nn  
import torch.nn.functional as F  
from config import *  

class FastIntentClassifier(nn.Module):  
    def __init__(self, input_size, hidden_size=FAST_MODEL_HIDDEN_SIZE,   
                 num_classes=len(INTENT_CLASSES)):  
        super(FastIntentClassifier, self).__init__()  
        
        # CNN部分  
        self.conv1 = nn.Conv1d(input_size, hidden_size, kernel_size=3, padding=1)  
        self.bn1 = nn.BatchNorm1d(hidden_size)  
        self.conv2 = nn.Conv1d(hidden_size, hidden_size, kernel_size=3, padding=1)  
        self.bn2 = nn.BatchNorm1d(hidden_size)  
        self.pool = nn.MaxPool1d(2)  
        self.dropout1 = nn.Dropout(0.2)  
        
        # LSTM部分  
        self.lstm = nn.GRU(hidden_size, hidden_size, batch_first=True, bidirectional=True)  
        self.dropout2 = nn.Dropout(0.2)  
        
        # 全连接层  
        self.fc = nn.Linear(hidden_size * 2, num_classes)  # 双向LSTM，所以是hidden_size * 2  
        
    def forward(self, x):  
        # 输入x形状: (batch_size, seq_len, input_size)  
        # 转换为CNN的输入形状: (batch_size, input_size, seq_len)  
        x = x.transpose(1, 2)  
        
        # CNN层  
        x = F.relu(self.bn1(self.conv1(x)))  
        x = self.pool(x)  
        x = F.relu(self.bn2(self.conv2(x)))  
        x = self.dropout1(x)  
        
        # 转换回序列形状(batch_size, seq_len, hidden_size)用于LSTM  
        x = x.transpose(1, 2)  
        
        # LSTM层  
        x, _ = self.lstm(x)  
        x = self.dropout2(x)  
        
        # 对序列进行全局平均池化  
        x = torch.mean(x, dim=1)  
        
        # 全连接层  
        x = self.fc(x)  
        
        return x  
    
    def predict(self, x):  
        """返回预测的类别和置信度"""  
        logits = self.forward(x)  
        probs = F.softmax(logits, dim=1)  
        confidences, predicted = torch.max(probs, dim=1)  
        
        return predicted, confidences