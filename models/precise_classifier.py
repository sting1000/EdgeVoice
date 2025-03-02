# models/precise_classifier.py  
import torch  
import torch.nn as nn  
import torch.nn.functional as F  
from transformers import DistilBertModel, DistilBertConfig  
from config import *  

class PreciseIntentClassifier(nn.Module):  
    def __init__(self, hidden_size=PRECISE_MODEL_HIDDEN_SIZE,   
                 num_classes=len(INTENT_CLASSES)):  
        super(PreciseIntentClassifier, self).__init__()  
        
        # 轻量级Transformer配置  
        # 使用DistilBert作为基础，但进一步减小  
        self.config = DistilBertConfig(  
            vocab_size=30522,  # 与原始DistilBERT相同  
            hidden_size=hidden_size,  # 减小隐藏大小  
            num_hidden_layers=3,  # 减少层数  
            num_attention_heads=4,  # 减少注意力头  
            intermediate_size=hidden_size * 2,  # 减小中间层大小  
            max_position_embeddings=128,  # 减小位置嵌入的最大长度  
        )  
        
        # 初始化轻量级Transformer模型  
        self.transformer = DistilBertModel(self.config)  
        
        # 分类头  
        self.dropout = nn.Dropout(0.1)  
        self.classifier = nn.Linear(hidden_size, num_classes)  
        
    def forward(self, input_ids, attention_mask=None):  
        """  
        input_ids: 分词后的输入ID  
        attention_mask: 注意力掩码  
        """  
        # Transformer编码  
        outputs = self.transformer(input_ids=input_ids, attention_mask=attention_mask)  
        
        # 取[CLS]位置的输出作为整个序列的表示  
        sequence_output = outputs.last_hidden_state[:, 0, :]  
        sequence_output = self.dropout(sequence_output)  
        
        # 分类  
        logits = self.classifier(sequence_output)  
        
        return logits  
    
    def predict(self, input_ids, attention_mask=None):  
        """返回预测的类别和置信度"""  
        logits = self.forward(input_ids, attention_mask)  
        probs = F.softmax(logits, dim=1)  
        confidences, predicted = torch.max(probs, dim=1)  
        
        return predicted, confidences