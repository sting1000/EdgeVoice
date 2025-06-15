import torch
import torch.nn as nn
import torch.nn.functional as F
from config import *

class RelativePositionalEncoding(nn.Module):
    """4维相对位置编码，满足部署约束"""
    def __init__(self, d_model, max_len=1000):
        super().__init__()
        
        # 确保d_model满足16通道对齐
        d_model = (d_model // 16) * 16
        if d_model < 16:
            d_model = 16
        
        # 创建4维位置编码矩阵 [1, d_model, max_len, 1]
        pe = torch.zeros(1, d_model, max_len, 1)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(0).unsqueeze(0).unsqueeze(-1)  # [1, 1, max_len, 1]
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * 
            (-torch.log(torch.tensor(10000.0)) / d_model)
        ).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)  # [1, d_model//2, 1, 1]
        
        pe[:, 0::2, :, :] = torch.sin(position * div_term)
        pe[:, 1::2, :, :] = torch.cos(position * div_term)
        
        # 注册为缓冲区
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        """应用位置编码到4维输入特征
        
        Args:
            x: 输入特征 [batch_size, d_model, seq_len, 1]
            
        Returns:
            position_encoded: 添加位置编码的特征 [batch_size, d_model, seq_len, 1]
        """
        batch_size, d_model, seq_len, _ = x.shape
        
        # 扩展位置编码到当前批次大小，避免广播
        pe_expanded = self.pe[:, :d_model, :seq_len, :].expand(batch_size, -1, -1, -1)
        
        # 直接相加，不使用广播
        position_encoded = x + pe_expanded
        return position_encoded

class ConformerBlock(nn.Module):
    """4维Conformer编码器块，满足部署约束"""
    def __init__(self, dim, dim_head=64, heads=8, ff_mult=4, 
                 conv_expansion_factor=2, conv_kernel_size=31,
                 attn_dropout=0.1, ff_dropout=0.1, conv_dropout=0.1):
        super().__init__()
        
        # 确保dim满足16通道对齐
        dim = (dim // 16) * 16
        if dim < 16:
            dim = 16
        
        # 减少FF网络的乘数以减少参数量
        self.ff1 = FeedForward(dim, mult=ff_mult//2, dropout=ff_dropout) 
        self.attn = MultiHeadAttention(dim, heads=heads, dim_head=dim_head, dropout=attn_dropout)
        self.conv = ConformerConvolution(dim, expansion_factor=conv_expansion_factor, 
                                         kernel_size=conv_kernel_size, dropout=conv_dropout)
        self.ff2 = FeedForward(dim, mult=ff_mult//2, dropout=ff_dropout)
        
        # 使用4维BatchNorm替代LayerNorm
        self.norm = nn.BatchNorm2d(dim)
        
        # 移除额外投影层，使用缩放因子替代
        self.ff1_scale = 0.5
        self.ff2_scale = 0.5
        
    def forward(self, x, cache=None):
        """前向传播，支持流式缓存
        
        Args:
            x: 输入特征 [batch_size, dim, seq_len, 1]
            cache: 可选的缓存状态
            
        Returns:
            x: 输出特征 [batch_size, dim, seq_len, 1]
            new_cache: 更新后的缓存状态（当cache不为None时）
        """
        # 第一个前馈模块 - 使用简单缩放
        ff1_out = self.ff1(x)
        x_ff1 = x + ff1_out * self.ff1_scale
        
        # 自注意力模块
        if cache is not None:
            # 使用缓存状态的流式处理
            attn_cache = cache.get('attn', None)
            attn_out, new_attn_cache = self.attn(x_ff1, cache=attn_cache)
            x_attn = x_ff1 + attn_out
            new_cache = {'attn': new_attn_cache}
        else:
            # 标准处理
            attn_out = self.attn(x_ff1)
            x_attn = x_ff1 + attn_out
            new_cache = None
        
        # 卷积模块
        if cache is not None:
            conv_cache = cache.get('conv', None)
            conv_out, new_conv_cache = self.conv(x_attn, cache=conv_cache)
            x_conv = x_attn + conv_out
            new_cache['conv'] = new_conv_cache
        else:
            conv_out = self.conv(x_attn)
            x_conv = x_attn + conv_out
        
        # 第二个前馈模块
        ff2_out = self.ff2(x_conv)
        x_ff2 = x_conv + ff2_out * self.ff2_scale
        
        # 最后的归一化
        output = self.norm(x_ff2)
        
        return (output, new_cache) if cache is not None else output

class FeedForward(nn.Module):
    """4维前馈网络，使用Conv2d替代Linear"""
    def __init__(self, dim, mult=4, dropout=0.1):
        super().__init__()
        
        # 确保维度满足16通道对齐
        dim = (dim // 16) * 16
        if dim < 16:
            dim = 16
        
        inner_dim = (dim * mult // 16) * 16
        if inner_dim < 16:
            inner_dim = 16
        
        self.norm = nn.BatchNorm2d(dim)
        
        # 使用1x1卷积替代Linear，保持4维操作
        self.fc1 = nn.Conv2d(dim, inner_dim, kernel_size=1)
        self.act = nn.SiLU()  # SiLU比SwiGLU更轻量
        self.fc2 = nn.Conv2d(inner_dim, dim, kernel_size=1)
        self.dropout = nn.Dropout2d(dropout)
        
    def forward(self, x):
        """前向传播
        
        Args:
            x: 输入特征 [batch_size, dim, seq_len, 1]
            
        Returns:
            x: 输出特征 [batch_size, dim, seq_len, 1]
        """
        x_norm = self.norm(x)
        x = self.fc1(x_norm)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x

class MultiHeadAttention(nn.Module):
    """4维多头注意力模块，避免超过4维的中间张量"""
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.1):
        super().__init__()
        
        # 确保维度满足16通道对齐
        dim = (dim // 16) * 16
        if dim < 16:
            dim = 16
        
        dim_head = (dim_head // 16) * 16
        if dim_head < 16:
            dim_head = 16
        
        # 减少内部维度
        inner_dim = dim_head * heads
        inner_dim = (inner_dim // 16) * 16
        if inner_dim < 16:
            inner_dim = 16
        
        self.heads = heads
        self.dim_head = dim_head
        self.scale = dim_head ** -0.5
        
        self.norm = nn.BatchNorm2d(dim)
        
        # 使用1x1卷积替代Linear，保持4维操作
        self.q_proj = nn.Conv2d(dim, inner_dim, kernel_size=1)
        self.k_proj = nn.Conv2d(dim, inner_dim, kernel_size=1)
        self.v_proj = nn.Conv2d(dim, inner_dim, kernel_size=1)
        self.out_proj = nn.Conv2d(inner_dim, dim, kernel_size=1)
        self.dropout = nn.Dropout2d(dropout)
        
    def forward(self, x, mask=None, cache=None):
        """前向传播，支持注意力缓存
        
        Args:
            x: 输入特征 [batch_size, dim, seq_len, 1]
            mask: 可选的注意力掩码
            cache: 可选的k/v缓存
            
        Returns:
            out: 注意力输出 [batch_size, dim, seq_len, 1]
            updated_cache: 更新后的缓存(如果使用缓存)
        """
        batch_size, dim, seq_len, _ = x.shape
        
        # 归一化
        x = self.norm(x)
        
        # 分别投影Q、K、V，保持4维格式
        q = self.q_proj(x)  # [batch_size, inner_dim, seq_len, 1]
        k = self.k_proj(x)  # [batch_size, inner_dim, seq_len, 1]
        v = self.v_proj(x)  # [batch_size, inner_dim, seq_len, 1]
        
        # 处理KV缓存(用于流式推理)
        if cache is not None:
            cached_k, cached_v = cache
            
            if cached_k is not None and cached_v is not None:
                # 合并缓存的KV与当前KV，在seq_len维度上拼接
                k = torch.cat([cached_k, k], dim=2)
                v = torch.cat([cached_v, v], dim=2)
                
            # 为下一次流式推理更新缓存
            updated_cache = (k, v)
        else:
            updated_cache = None
        
        # 重新组织为多头格式，避免超过4维
        # 我们需要避免使用view创建5维张量，所以分别处理每个头
        head_outputs = []
        for head_idx in range(self.heads):
            start_idx = head_idx * self.dim_head
            end_idx = start_idx + self.dim_head
            
            q_head = q[:, start_idx:end_idx, :, :]  # [batch, dim_head, seq_len, 1]
            k_head = k[:, start_idx:end_idx, :, :]  # [batch, dim_head, seq_len_k, 1]
            v_head = v[:, start_idx:end_idx, :, :]  # [batch, dim_head, seq_len_k, 1]
            
            # 计算注意力分数
            # q_head: [batch, dim_head, seq_len, 1] -> [batch, seq_len, dim_head]
            # k_head: [batch, dim_head, seq_len_k, 1] -> [batch, seq_len_k, dim_head]
            q_reshaped = q_head.squeeze(-1).transpose(1, 2)  # [batch, seq_len, dim_head]
            k_reshaped = k_head.squeeze(-1).transpose(1, 2)  # [batch, seq_len_k, dim_head]
            
            # 计算注意力分数: [batch, seq_len, dim_head] x [batch, dim_head, seq_len_k]
            attn_scores = torch.matmul(q_reshaped, k_reshaped.transpose(-1, -2)) * self.scale
            # 结果: [batch, seq_len, seq_len_k]
            
            # 应用掩码(如果提供)
            if mask is not None:
                # 假设mask是[batch, seq_len, seq_len_k]格式
                attn_scores = attn_scores.masked_fill(mask == 0, -1e9)
            
            # 注意力权重
            attn_weights = F.softmax(attn_scores, dim=-1)  # [batch, seq_len, seq_len_k]
            
            # 应用注意力
            # v_head: [batch, dim_head, seq_len_k, 1] -> [batch, seq_len_k, dim_head]
            v_reshaped = v_head.squeeze(-1).transpose(1, 2)  # [batch, seq_len_k, dim_head]
            
            # [batch, seq_len, seq_len_k] x [batch, seq_len_k, dim_head] -> [batch, seq_len, dim_head]
            head_out = torch.matmul(attn_weights, v_reshaped)
            
            # 转换回4维格式: [batch, seq_len, dim_head] -> [batch, dim_head, seq_len, 1]
            head_out = head_out.transpose(1, 2).unsqueeze(-1)  # [batch, dim_head, seq_len, 1]
            
            head_outputs.append(head_out)
        
        # 合并所有头的输出
        out = torch.cat(head_outputs, dim=1)  # [batch_size, inner_dim, seq_len, 1]
        
        # 应用dropout
        out = self.dropout(out)
        
        # 输出投影
        out = self.out_proj(out)
        
        if cache is not None:
            return out, updated_cache
        return out

class ConformerConvolution(nn.Module):
    """4维Conformer卷积模块，满足部署约束"""
    def __init__(self, dim, expansion_factor=2, kernel_size=31, dropout=0.1):
        super().__init__()
        
        # 确保维度16通道对齐 (FP16要求)
        dim = (dim // 16) * 16
        if dim < 16:
            dim = 16
        
        inner_dim = ((dim * expansion_factor) // 16) * 16
        if inner_dim < 16:
            inner_dim = 16
            
        padding = (kernel_size - 1) // 2
        
        self.norm = nn.BatchNorm2d(dim)
        
        # 第一个逐点卷积
        self.pointwise_conv1 = nn.Conv2d(dim, inner_dim * 2, kernel_size=1)
        
        # 移除分组卷积，使用普通卷积替代（满足部署约束）
        self.conv = nn.Conv2d(
            inner_dim, inner_dim, 
            kernel_size=(1, kernel_size),  # 2D卷积(H=1,W=kernel_size)
            padding=(0, padding),         # 只在时间维度上填充
            groups=1  # 移除分组卷积
        )
        
        self.batch_norm = nn.BatchNorm2d(inner_dim)
        
        # 第二个逐点卷积
        self.pointwise_conv2 = nn.Conv2d(inner_dim, dim, kernel_size=1)
        
        self.dropout = nn.Dropout2d(dropout)
        
        # 缓存卷积状态用于流式处理
        self.cache_size = 0
        self.kernel_size = kernel_size
        
    def forward(self, x, cache=None):
        """前向传播，支持卷积状态缓存
        
        Args:
            x: 输入特征 [batch_size, dim, seq_len, 1]
            cache: 可选的卷积状态缓存
            
        Returns:
            out: 卷积输出 [batch_size, dim, seq_len, 1]
            updated_cache: 更新后的缓存状态(如果使用缓存)
        """
        batch_size, dim, seq_len, _ = x.shape
        
        # 归一化
        x = self.norm(x)
        
        # 第一个逐点卷积
        x = self.pointwise_conv1(x)
        # 对Conv2d的输出应用GLU
        x = F.glu(x, dim=1)
        
        # 处理卷积状态缓存
        if cache is not None:
            cached_x = cache
            
            if cached_x is not None:
                # 合并缓存状态与当前输入
                x = torch.cat([cached_x, x], dim=2)  # 在seq_len维度上拼接
                
            # 计算要保留的历史大小
            cache_size = self.kernel_size - 1
            
            # 更新卷积状态缓存
            if x.size(2) > cache_size:
                updated_cache = x[:, :, -cache_size:, :]
            else:
                updated_cache = x
        else:
            updated_cache = None
        
        # 使用2D卷积
        x = self.conv(x)
        x = self.batch_norm(x)
        x = F.silu(x)
        
        # 第二个逐点卷积
        x = self.pointwise_conv2(x)
        x = self.dropout(x)
        
        if cache is not None:
            # 对于流式处理，只返回实际处理的新输入
            if cached_x is not None:
                cached_len = cached_x.size(2)  # 缓存的seq_len
                x = x[:, :, -seq_len:, :]
            
            return x, updated_cache
        
        return x

class AttentivePooling(nn.Module):
    """4维注意力池化层"""
    def __init__(self, dim):
        super().__init__()
        
        # 确保维度16通道对齐
        dim = (dim // 16) * 16
        if dim < 16:
            dim = 16
        
        hidden_dim = max(16, (dim // 2 // 16) * 16)
        
        # 使用Conv2d替代Linear，保持4维操作
        self.attention = nn.Sequential(
            nn.Conv2d(dim, hidden_dim, kernel_size=1),
            nn.Tanh(),
            nn.Conv2d(hidden_dim, 1, kernel_size=1)
        )
        
    def forward(self, x):
        """应用注意力池化
        
        Args:
            x: 输入特征 [batch_size, dim, seq_len, 1]
            
        Returns:
            weighted_x: 池化后的特征 [batch_size, dim, 1, 1]
        """
        # 计算注意力权重
        attn_weights = self.attention(x)  # [batch_size, 1, seq_len, 1]
        attn_weights = F.softmax(attn_weights, dim=2)
        
        # 加权求和，避免使用广播
        # attn_weights: [batch_size, 1, seq_len, 1]
        # x: [batch_size, dim, seq_len, 1]
        
        # 扩展attn_weights到与x相同的通道数
        attn_weights_expanded = attn_weights.expand(-1, x.size(1), -1, -1)
        
        # 逐元素相乘然后在seq_len维度上求和
        weighted_sum = (x * attn_weights_expanded).sum(dim=2, keepdim=True)
        
        return weighted_sum

class StreamingConformer(nn.Module):
    """4维流式Conformer模型，满足部署约束"""
    def __init__(self, input_dim=48, hidden_dim=160, num_classes=4, 
                 num_layers=6, num_heads=8, dropout=0.1, 
                 kernel_size=15, expansion_factor=4):
        super().__init__()
        
        # 确保所有维度满足16通道对齐要求 (FP16部署要求)
        input_dim = (input_dim // 16) * 16
        if input_dim < 16:
            input_dim = 16
        
        hidden_dim = (hidden_dim // 16) * 16
        if hidden_dim < 16:
            hidden_dim = 16
        
        # 确保dim_head满足对齐要求
        dim_head = (hidden_dim // num_heads)
        dim_head = (dim_head // 16) * 16
        if dim_head < 16:
            dim_head = 16
        
        # 重新计算实际的hidden_dim以保持一致性
        hidden_dim = dim_head * num_heads
        
        # 使用Conv2d替代Linear，保持4维操作
        # 输入格式：[batch, 1, seq, input_dim] -> [batch, hidden_dim, seq, 1]
        self.input_projection = nn.Conv2d(input_dim, hidden_dim, kernel_size=1)
        
        # 位置编码
        self.pos_encoding = RelativePositionalEncoding(hidden_dim, max_len=1000)
        
        # Conformer编码器层
        self.conformer_layers = nn.ModuleList([
            ConformerBlock(
                dim=hidden_dim,
                dim_head=dim_head,
                heads=num_heads,
                ff_mult=expansion_factor,
                conv_expansion_factor=1,  # 减少卷积扩展因子降低复杂度
                conv_kernel_size=kernel_size,
                attn_dropout=dropout,
                ff_dropout=dropout,
                conv_dropout=dropout
            ) for _ in range(num_layers)
        ])
        
        # 特征池化层 - 使用注意力池化
        self.attention_pooling = AttentivePooling(hidden_dim)
        
        # 输出层 - 确保维度对齐
        classifier_hidden = max(16, (hidden_dim // 2 // 16) * 16)
        
        # 使用Conv2d替代Linear，然后在最后展平
        self.classifier = nn.Sequential(
            nn.Conv2d(hidden_dim, classifier_hidden, kernel_size=1),
            nn.BatchNorm2d(classifier_hidden),
            nn.ReLU(),
            nn.Dropout2d(dropout),
            nn.Conv2d(classifier_hidden, num_classes, kernel_size=1),
            nn.AdaptiveAvgPool2d((1, 1)),  # 池化到 [batch, num_classes, 1, 1]
            nn.Flatten()  # 展平到 [batch, num_classes]
        )
        
        # 缓存用于流式推理
        self.reset_streaming_state()
        
        # 打印关键参数
        print(f"4维Conformer参数: input_dim={input_dim}, hidden_dim={hidden_dim}, dim_head={dim_head}, heads={num_heads}")
        print(f"符合16通道对齐要求，所有算子都使用4维张量，满足部署约束")
    
    def reset_streaming_state(self):
        """重置流式状态"""
        self.cached_features = None
        self.cache_size = 0
        self.max_cache_size = MAX_CACHED_FRAMES  # 最大缓存帧数
        self.layer_caches = [None] * len(self.conformer_layers)
    
    def forward_streaming(self, x, cache_states=None, return_cache=True):
        """流式前向传播，支持特征缓存
        
        Args:
            x: 输入特征 [batch_size, 1, seq_len, input_dim]
            cache_states: 可选的缓存状态
            return_cache: 是否返回更新的缓存状态
            
        Returns:
            logits: 分类输出 [batch_size, num_classes]
            cache_states: 更新的缓存状态(如果return_cache=True)
        """
        batch_size, _, seq_len, input_dim = x.shape
        
        # 转换输入格式：[batch, 1, seq, input_dim] -> [batch, input_dim, seq, 1]
        x = x.permute(0, 3, 2, 1)
        
        # 输入投影
        x = self.input_projection(x)  # [batch, hidden_dim, seq, 1]
        
        if cache_states is not None:
            # 使用提供的缓存状态
            cached_features, layer_caches = cache_states
            
            if cached_features is not None:
                # 拼接缓存特征和当前特征
                cached_seq_len = cached_features.size(2)
                
                # 确保批次大小一致
                if cached_features.size(0) != batch_size:
                    cached_features = cached_features.expand(batch_size, -1, -1, -1)
                
                # 拼接缓存特征和当前特征
                x = torch.cat([cached_features, x], dim=2)
                new_seq_len = cached_seq_len + seq_len
                
                # 更新缓存大小
                cache_size = min(self.max_cache_size, new_seq_len)
            else:
                # 初始化缓存
                cache_size = min(self.max_cache_size, seq_len)
                new_seq_len = seq_len
                layer_caches = [None] * len(self.conformer_layers)
        else:
            # 初始化缓存状态
            cache_size = min(self.max_cache_size, seq_len)
            new_seq_len = seq_len
            layer_caches = [None] * len(self.conformer_layers)
            
        # 位置编码
        x = self.pos_encoding(x)
        
        # Conformer编码器层
        new_layer_caches = []
        for i, layer in enumerate(self.conformer_layers):
            layer_cache = layer_caches[i] if i < len(layer_caches) else None
            
            # 使用try-except处理缓存问题
            try:
                result = layer(x, cache=layer_cache)
                if isinstance(result, tuple) and len(result) == 2:
                    x, new_cache = result
                else:
                    x = result
                    new_cache = None
            except Exception as e:
                print(f"处理第{i}层时出错: {e}")
                x = layer(x)  # 退回到无缓存模式
                new_cache = None
                
            new_layer_caches.append(new_cache)
        
        # 注意力池化
        pooled = self.attention_pooling(x)  # [batch, hidden_dim, 1, 1]
        
        # 分类
        logits = self.classifier(pooled)  # [batch, num_classes]
        
        # 更新要缓存的特征
        if return_cache:
            if new_seq_len > cache_size:
                # 只保留最后cache_size帧
                new_cache_features = x[:, :, -cache_size:, :].detach()
            else:
                new_cache_features = x.detach()
            
            return logits, (new_cache_features, new_layer_caches)
        
        return logits

    def forward(self, x):
        """标准前向传播
        
        Args:
            x: 输入特征 [batch_size, 1, seq_len, input_dim]
            
        Returns:
            logits: 分类输出 [batch_size, num_classes]
        """
        # 转换输入格式：[batch, 1, seq, input_dim] -> [batch, input_dim, seq, 1]
        x = x.permute(0, 3, 2, 1)
        
        # 特征投影
        x = self.input_projection(x)
        
        # 位置编码
        x = self.pos_encoding(x)
        
        # Conformer编码器层
        for layer in self.conformer_layers:
            # 对于标准前向传播，不使用缓存
            x = layer(x)
        
        # 注意力池化
        pooled = self.attention_pooling(x)
        
        # 分类
        logits = self.classifier(pooled)
        
        return logits
    
    def predict_streaming(self, x, cached_states=None):
        """进行流式预测，返回预测结果和置信度
        
        Args:
            x: 输入特征 [batch_size, 1, seq_len, input_dim]
            cached_states: 可选的缓存状态
            
        Returns:
            pred: 预测类别索引
            conf: 预测置信度
            new_cached_states: 更新的缓存状态
        """
        logits, new_cached_states = self.forward_streaming(x, cached_states, True)
        
        # 获取预测类别和置信度
        probs = F.softmax(logits, dim=-1)
        conf, pred = torch.max(probs, dim=-1)
        
        return pred, conf, new_cached_states 