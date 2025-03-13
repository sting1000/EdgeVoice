# models/fast_classifier.py  
import torch  
import torch.nn as nn  
import torch.nn.functional as F  
from config import *  

class PositionalEncoding(nn.Module):
    """位置编码层"""
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        
        # 创建位置编码矩阵
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        
        # 注册为缓冲区，不作为模型参数
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        # x: [batch_size, seq_len, d_model]
        x = x + self.pe[:, :x.size(1)]
        return x

class MultiHeadSelfAttention(nn.Module):
    """多头自注意力层"""
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        
        assert self.head_dim * num_heads == d_model, "d_model必须能被num_heads整除"
        
        self.qkv_proj = nn.Linear(d_model, 3 * d_model)
        self.dropout = nn.Dropout(dropout)
        self.out_proj = nn.Linear(d_model, d_model)
        
    def forward(self, x, mask=None):
        # x: [batch_size, seq_len, d_model]
        batch_size, seq_len, _ = x.size()
        
        # 线性投影并分头
        qkv = self.qkv_proj(x)
        qkv = qkv.reshape(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, batch_size, num_heads, seq_len, head_dim]
        
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # 计算注意力分数
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)
            
        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_probs = self.dropout(attn_probs)
        
        # 应用注意力权重
        context = torch.matmul(attn_probs, v)
        context = context.permute(0, 2, 1, 3).contiguous()
        context = context.reshape(batch_size, seq_len, self.d_model)
        
        # 输出投影
        output = self.out_proj(context)
        
        return output

class ConvModule(nn.Module):
    """Conformer卷积模块"""
    def __init__(self, d_model, kernel_size=31, expansion_factor=2, dropout=0.1):
        super().__init__()
        
        inner_dim = d_model * expansion_factor
        
        self.layer_norm = nn.LayerNorm(d_model)
        self.pointwise_conv1 = nn.Conv1d(d_model, inner_dim, kernel_size=1)
        self.glu = nn.GLU(dim=1)
        self.depthwise_conv = nn.Conv1d(
            inner_dim // 2, 
            inner_dim // 2, 
            kernel_size=kernel_size, 
            padding=kernel_size // 2, 
            groups=inner_dim // 2
        )
        self.batch_norm = nn.BatchNorm1d(inner_dim // 2)
        self.activation = nn.SiLU()
        self.pointwise_conv2 = nn.Conv1d(inner_dim // 2, d_model, kernel_size=1)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # x: [batch_size, seq_len, d_model]
        residual = x
        x = self.layer_norm(x)
        
        # 调整维度为 [batch_size, d_model, seq_len]
        x = x.transpose(1, 2)
        x = self.pointwise_conv1(x)
        x = self.glu(x)
        
        # 深度可分离卷积
        x = self.depthwise_conv(x)
        x = self.batch_norm(x)
        x = self.activation(x)
        
        # 第二次逐点卷积
        x = self.pointwise_conv2(x)
        x = self.dropout(x)
        
        # 调整回 [batch_size, seq_len, d_model]
        x = x.transpose(1, 2)
        
        # 残差连接
        x = x + residual
        
        return x

class FeedForwardModule(nn.Module):
    """前馈网络模块"""
    def __init__(self, d_model, d_ff=None, dropout=0.1):
        super().__init__()
        
        if d_ff is None:
            d_ff = 4 * d_model
            
        self.layer_norm = nn.LayerNorm(d_model)
        self.fc1 = nn.Linear(d_model, d_ff)
        self.activation = nn.SiLU()
        self.dropout1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout2 = nn.Dropout(dropout)
        
    def forward(self, x):
        # x: [batch_size, seq_len, d_model]
        residual = x
        x = self.layer_norm(x)
        
        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout1(x)
        
        x = self.fc2(x)
        x = self.dropout2(x)
        
        x = x + residual
        
        return x

class ConformerBlock(nn.Module):
    """Conformer编码器块"""
    def __init__(self, d_model, num_heads, conv_kernel_size=31, ff_expansion_factor=4, dropout=0.1):
        super().__init__()
        
        self.ff_module1 = FeedForwardModule(d_model, d_ff=d_model*ff_expansion_factor, dropout=dropout)
        self.self_attn_module = nn.Sequential(
            nn.LayerNorm(d_model),
            MultiHeadSelfAttention(d_model, num_heads, dropout=dropout),
            nn.Dropout(dropout)
        )
        self.conv_module = ConvModule(d_model, kernel_size=conv_kernel_size, dropout=dropout)
        self.ff_module2 = FeedForwardModule(d_model, d_ff=d_model*ff_expansion_factor, dropout=dropout)
        self.layer_norm = nn.LayerNorm(d_model)
        
    def forward(self, x):
        # x: [batch_size, seq_len, d_model]
        
        # FFN模块1 (输出除以2用于缩放)
        x = x + 0.5 * self.ff_module1(x)
        
        # 自注意力模块
        x = x + self.self_attn_module(x)
        
        # 卷积模块
        x = x + self.conv_module(x)
        
        # FFN模块2 (输出除以2用于缩放)
        x = x + 0.5 * self.ff_module2(x)
        
        # 最终层归一化
        x = self.layer_norm(x)
        
        return x

class FastIntentClassifier(nn.Module):  
    def __init__(self, input_size, hidden_size=FAST_MODEL_HIDDEN_SIZE,   
                 num_classes=len(INTENT_CLASSES)):  
        super(FastIntentClassifier, self).__init__()  
        
        # 特征投影
        self.input_projection = nn.Linear(input_size, hidden_size)
        
        # 位置编码
        self.pos_encoding = PositionalEncoding(hidden_size)
        
        # Conformer编码器块
        self.conformer_blocks = nn.ModuleList([
            ConformerBlock(
                d_model=hidden_size,
                num_heads=CONFORMER_ATTENTION_HEADS,
                conv_kernel_size=CONFORMER_CONV_KERNEL_SIZE,
                ff_expansion_factor=CONFORMER_FF_EXPANSION_FACTOR,
                dropout=CONFORMER_DROPOUT
            ) for _ in range(CONFORMER_LAYERS)
        ])
        
        # 输出层
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(hidden_size, num_classes)
        
    def forward(self, x):  
        # 输入x形状: (batch_size, seq_len, input_size) 或 (batch_size, input_size)
        
        # 确保输入是3D张量，处理2D输入的情况
        if x.dim() == 2:
            # 如果输入是2D: [batch_size, input_size]，则添加seq_len维度
            x = x.unsqueeze(1)  # [batch_size, 1, input_size]
        
        # 特征投影
        x = self.input_projection(x)
        
        # 添加位置编码
        x = self.pos_encoding(x)
        
        # 通过Conformer层
        for block in self.conformer_blocks:
            x = block(x)
        
        # 对序列进行全局平均池化  
        x = torch.mean(x, dim=1)
        
        # 应用dropout并通过全连接层
        x = self.dropout(x)
        x = self.fc(x)
        
        return x
    
    def predict(self, x):  
        """返回预测的类别和置信度"""  
        logits = self.forward(x)  
        probs = F.softmax(logits, dim=1)  
        confidences, predicted = torch.max(probs, dim=1)  
        
        return predicted, confidences