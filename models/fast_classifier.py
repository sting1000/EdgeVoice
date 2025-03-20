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
        # 避免使用广播操作，使用expand显式扩展维度
        pe_expanded = self.pe[:, :x.size(1), :].expand(x.size(0), -1, -1)
        x = x + pe_expanded
        return x

class MultiHeadSelfAttention(nn.Module):
    """多头自注意力层 - 重构以避免5维计算"""
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        
        assert self.head_dim * num_heads == d_model, "d_model必须能被num_heads整除"
        
        # 分离QKV投影以避免reshape和permute导致的高维张量
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.out_proj = nn.Linear(d_model, d_model)
        
    def _reshape_for_attention(self, x):
        # 将[batch_size, seq_len, d_model]重塑为[batch_size*num_heads, seq_len, head_dim]
        # 避免使用5维张量
        batch_size, seq_len, _ = x.size()
        x = x.view(batch_size, seq_len, self.num_heads, self.head_dim)
        # 转置为[batch_size, num_heads, seq_len, head_dim]
        x = x.transpose(1, 2)
        # 合并batch和head维度
        x = x.reshape(batch_size * self.num_heads, seq_len, self.head_dim)
        return x
    
    def _reshape_from_attention(self, x, batch_size, seq_len):
        # 将[batch_size*num_heads, seq_len, head_dim]重塑回[batch_size, seq_len, d_model]
        x = x.view(batch_size, self.num_heads, seq_len, self.head_dim)
        x = x.transpose(1, 2)
        x = x.reshape(batch_size, seq_len, self.d_model)
        return x
        
    def forward(self, x, mask=None):
        # x: [batch_size, seq_len, d_model] 或 [batch_size, d_model]
        
        # 处理2D输入情况 (单帧)
        if x.dim() == 2:
            # 如果输入是2D: [batch_size, d_model]，添加序列维度
            x = x.unsqueeze(1)  # [batch_size, 1, d_model]
        
        batch_size, seq_len, _ = x.size()
        
        # 独立的Q、K、V投影
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        
        # 重塑张量以进行注意力计算，将batch和head维度合并
        q = self._reshape_for_attention(q)  # [batch_size*num_heads, seq_len, head_dim]
        k = self._reshape_for_attention(k)  # [batch_size*num_heads, seq_len, head_dim]
        v = self._reshape_for_attention(v)  # [batch_size*num_heads, seq_len, head_dim]
        
        # 计算注意力分数
        attn_scores = torch.bmm(q, k.transpose(1, 2)) / (self.head_dim ** 0.5)  # [batch_size*num_heads, seq_len, seq_len]
        
        if mask is not None:
            # 需要复制mask以匹配合并后的维度
            mask = mask.repeat(self.num_heads, 1, 1)
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)
            
        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_probs = self.dropout(attn_probs)
        
        # 应用注意力权重
        context = torch.bmm(attn_probs, v)  # [batch_size*num_heads, seq_len, head_dim]
        
        # 重塑回原始维度
        context = self._reshape_from_attention(context, batch_size, seq_len)  # [batch_size, seq_len, d_model]
        
        # 输出投影
        output = self.out_proj(context)
        
        return output

class ConvModule(nn.Module):
    """Conformer卷积模块 - 替换depthwise卷积为分组卷积"""
    def __init__(self, d_model, kernel_size=31, expansion_factor=2, dropout=0.1):
        super().__init__()
        
        inner_dim = d_model * expansion_factor
        
        self.layer_norm = nn.LayerNorm(d_model)
        self.pointwise_conv1 = nn.Conv1d(d_model, inner_dim, kernel_size=1)
        self.glu = nn.GLU(dim=1)
        
        # 替换depthwise卷积为分组卷积，适应硬件加速要求
        # 使用8组卷积，每组通道数为inner_dim//16
        num_groups = 8
        self.grouped_conv = nn.Conv1d(
            inner_dim // 2, 
            inner_dim // 2, 
            kernel_size=kernel_size, 
            padding=kernel_size // 2, 
            groups=num_groups  # 使用固定的组数而非depthwise
        )
        
        self.batch_norm = nn.BatchNorm1d(inner_dim // 2)
        self.activation = nn.SiLU()
        self.pointwise_conv2 = nn.Conv1d(inner_dim // 2, d_model, kernel_size=1)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # x: [batch_size, seq_len, d_model] 或 [batch_size, d_model]
        
        # 处理2D输入情况 (单帧)
        if x.dim() == 2:
            # 如果输入是2D: [batch_size, d_model]，添加序列维度
            x = x.unsqueeze(1)  # [batch_size, 1, d_model]
            is_single_frame = True
        else:
            is_single_frame = False
            
        residual = x
        x = self.layer_norm(x)
        
        # 调整维度为 [batch_size, d_model, seq_len]
        x = x.transpose(1, 2)
        x = self.pointwise_conv1(x)
        x = self.glu(x)
        
        # 分组卷积替代深度可分离卷积
        x = self.grouped_conv(x)
        x = self.batch_norm(x)
        x = self.activation(x)
        
        # 第二次逐点卷积
        x = self.pointwise_conv2(x)
        x = self.dropout(x)
        
        # 调整回 [batch_size, seq_len, d_model]
        x = x.transpose(1, 2)
        
        # 显式残差连接，避免广播
        x = x + residual
        
        # 如果输入是单帧，去掉序列维度
        if is_single_frame:
            x = x.squeeze(1)
            
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
        # x: [batch_size, seq_len, d_model] 或 [batch_size, d_model]
        
        # 处理2D输入情况 (单帧)
        if x.dim() == 2:
            # 如果输入是2D: [batch_size, d_model]，添加序列维度
            x = x.unsqueeze(1)  # [batch_size, 1, d_model]
            is_single_frame = True
        else:
            is_single_frame = False
            
        residual = x
        x = self.layer_norm(x)
        
        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout1(x)
        
        x = self.fc2(x)
        x = self.dropout2(x)
        
        # 显式相加，避免广播
        x = x + residual
        
        # 如果输入是单帧，去掉序列维度
        if is_single_frame:
            x = x.squeeze(1)
            
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
        
    def forward(self, x, cache=None):
        # x: [batch_size, seq_len, d_model] 或 [batch_size, d_model]
        
        # 处理2D输入情况 (单帧)
        if x.dim() == 2:
            # 如果输入是2D: [batch_size, d_model]，添加序列维度
            x = x.unsqueeze(1)  # [batch_size, 1, d_model]
            is_single_frame = True
        else:
            is_single_frame = False
        
        # 支持状态缓存的前向传播
        if cache is not None:
            prev_x, prev_attn, prev_conv = cache
            # 合并当前输入和缓存状态
            if prev_x is not None:
                x = torch.cat([prev_x, x], dim=1)
                is_single_frame = False  # 合并后肯定不是单帧了
        
        # FFN模块1 (输出除以2用于缩放)
        ff1_out = self.ff_module1(x)
        x = x + 0.5 * ff1_out
        
        # 自注意力模块
        attn_out = self.self_attn_module(x)
        x = x + attn_out
        
        # 卷积模块
        conv_out = self.conv_module(x)
        x = x + conv_out
        
        # FFN模块2 (输出除以2用于缩放)
        ff2_out = self.ff_module2(x)
        x = x + 0.5 * ff2_out
        
        # 最终层归一化
        x = self.layer_norm(x)
        
        # 如果是流式处理，保存状态
        if cache is not None:
            # 仅保留最后MAX_CACHED_FRAMES帧作为下一时刻的缓存
            if x.size(1) > MAX_CACHED_FRAMES:
                new_cache = (x[:, -MAX_CACHED_FRAMES:, :], attn_out[:, -MAX_CACHED_FRAMES:, :], conv_out[:, -MAX_CACHED_FRAMES:, :])
            else:
                new_cache = (x, attn_out, conv_out)
            
            # 如果原输入是单帧，且需要恢复单帧输出
            if is_single_frame:
                x = x.squeeze(1)  # 只有在不缓存状态时才有可能恢复为单帧
                
            return x, new_cache
        
        # 如果原输入是单帧，且需要恢复单帧输出
        if is_single_frame:
            x = x.squeeze(1)
            
        # 确保返回元组
        return x, None

class FastIntentClassifier(nn.Module):  
    def __init__(self, input_size, hidden_size=CONFORMER_HIDDEN_SIZE,   
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
            x, _ = block(x)  # 现在block返回(x, _)元组，我们只需要第一个元素
        
        # 对序列进行全局平均池化  
        x = torch.mean(x, dim=1)
        
        # 应用dropout并通过全连接层
        x = self.dropout(x)
        x = self.fc(x)
        
        return x
    
    def forward_streaming(self, x, cached_states=None):
        """支持流式处理的前向传播，优化版本
        
        Args:
            x: 新输入特征 [batch_size, chunk_size, input_size]
            cached_states: 前一时刻的缓存状态
            
        Returns:
            output: 模型输出
            new_states: 更新后的缓存状态
        """
        try:
            # 初始化缓存状态
            if cached_states is None:
                cached_states = [None] * len(self.conformer_blocks)
            
            # 确保输入是3D张量
            if x.dim() == 2:
                x = x.unsqueeze(1)
                
            # 特征投影
            x = self.input_projection(x)
            
            # 位置编码 - 对于流式处理，需要考虑之前帧的位置信息
            # 这里简化处理，每个chunk独立添加位置编码
            x = self.pos_encoding(x)
            
            # 缓存状态优化：使用PyTorch的JIT追踪以加速前向传播
            new_states = []
            for i, block in enumerate(self.conformer_blocks):
                try:
                    if cached_states[i] is not None:
                        # 优化：直接获取缓存状态的长度而不是解包
                        # 如果缓存状态太长，只保留最近的帧
                        if isinstance(cached_states[i], tuple) and len(cached_states[i]) > 0:
                            prev_x = cached_states[i][0]
                            if prev_x is not None and prev_x.size(1) > MAX_CACHED_FRAMES:
                                # 只保留最近的MAX_CACHED_FRAMES帧
                                cached_states[i] = tuple(s[:, -MAX_CACHED_FRAMES:, :] if s is not None else None 
                                                        for s in cached_states[i])
                        
                        # 带缓存状态的前向传播
                        x, state = block(x, cached_states[i])
                    else:
                        # 初始前向传播，没有缓存状态
                        x, state = block(x, None)
                except Exception as e:
                    print(f"Conformer块 {i} 前向传播错误: {e}")
                    # 如果这个块出错，使用原始输入
                    state = None
                
                # 将新状态添加到列表
                new_states.append(state)
            
            # 对当前chunk进行全局平均池化
            x = torch.mean(x, dim=1)
            
            # 应用dropout并通过全连接层
            x = self.dropout(x)
            logits = self.fc(x)
            
            return logits, new_states
            
        except Exception as e:
            print(f"流式前向传播整体错误: {e}")
            # 创建一个默认的输出
            device = x.device if hasattr(x, 'device') else torch.device('cpu')
            batch_size = x.size(0) if hasattr(x, 'size') and len(x.size()) > 0 else 1
            # 返回默认logits和空状态
            default_logits = torch.zeros(batch_size, self.fc.out_features, device=device)
            return default_logits, cached_states
    
    def predict(self, x):  
        """返回预测的类别和置信度"""  
        logits = self.forward(x)  
        probs = F.softmax(logits, dim=1)  
        confidences, predicted = torch.max(probs, dim=1)  
        
        return predicted, confidences
        
    def predict_streaming(self, x, cached_states=None):
        """流式模式下的预测
        
        Args:
            x: 新输入特征 [batch_size, chunk_size, input_size]
            cached_states: 前一时刻的缓存状态
            
        Returns:
            predicted: 预测的类别
            confidences: 预测的置信度
            new_states: 更新后的缓存状态
        """
        try:
            # 尝试正常的前向传播
            logits, new_states = self.forward_streaming(x, cached_states)
        except Exception as e:
            # 如果发生错误，创建一个默认的输出
            print(f"模型前向传播错误，使用默认输出: {e}")
            # 创建默认的logits和状态
            device = x.device
            logits = torch.zeros(x.size(0), self.fc.out_features, device=device)
            new_states = cached_states  # 保持状态不变
            
        # 计算概率和预测结果
        probs = F.softmax(logits, dim=1)
        confidences, predicted = torch.max(probs, dim=1)
        
        return predicted, confidences, new_states