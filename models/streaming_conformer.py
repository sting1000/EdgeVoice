import torch
import torch.nn as nn
import torch.nn.functional as F
from config import *

class RelativePositionalEncoding(nn.Module):
    """相对位置编码，更适合流式处理"""
    def __init__(self, d_model, max_len=1000):
        super().__init__()
        
        # 创建相对位置编码矩阵
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * 
            (-torch.log(torch.tensor(10000.0)) / d_model)
        )
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        
        # 注册为缓冲区
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        """应用位置编码到输入特征
        
        Args:
            x: 输入特征 [batch_size, seq_len, d_model]
            
        Returns:
            position_encoded: 添加位置编码的特征
        """
        position_encoded = x + self.pe[:, :x.size(1), :]
        return position_encoded

class ConformerBlock(nn.Module):
    """改进的Conformer编码器块，针对流式处理优化"""
    def __init__(self, dim, dim_head=64, heads=8, ff_mult=4, 
                 conv_expansion_factor=2, conv_kernel_size=31,
                 attn_dropout=0.1, ff_dropout=0.1, conv_dropout=0.1):
        super().__init__()
        
        self.ff1 = FeedForward(dim, mult=ff_mult, dropout=ff_dropout) 
        self.attn = MultiHeadAttention(dim, heads=heads, dim_head=dim_head, dropout=attn_dropout)
        self.conv = ConformerConvolution(dim, expansion_factor=conv_expansion_factor, 
                                         kernel_size=conv_kernel_size, dropout=conv_dropout)
        self.ff2 = FeedForward(dim, mult=ff_mult, dropout=ff_dropout)
        self.norm = nn.LayerNorm(dim)
        
    def forward(self, x, cache=None):
        """前向传播，支持流式缓存
        
        Args:
            x: 输入特征 [batch_size, seq_len, dim]
            cache: 可选的缓存状态
            
        Returns:
            x: 输出特征
            new_cache: 更新后的缓存状态（当cache不为None时）
        """
        # 第一个前馈模块 - FFN的输出按照0.5的比例缩放
        x = x + 0.5 * self.ff1(x)
        
        # 自注意力模块
        if cache is not None:
            # 使用缓存状态的流式处理
            attn_cache = cache.get('attn', None)
            attn_out, new_attn_cache = self.attn(x, cache=attn_cache)
            x = x + attn_out
            new_cache = {'attn': new_attn_cache}
        else:
            # 标准处理
            x = x + self.attn(x)
            new_cache = None
        
        # 卷积模块
        if cache is not None:
            conv_cache = cache.get('conv', None)
            conv_out, new_conv_cache = self.conv(x, cache=conv_cache)
            x = x + conv_out
            new_cache['conv'] = new_conv_cache
        else:
            x = x + self.conv(x)
        
        # 第二个前馈模块
        x = x + 0.5 * self.ff2(x)
        
        # 最后的归一化
        x = self.norm(x)
        
        return (x, new_cache) if cache is not None else x

class FeedForward(nn.Module):
    """增强的前馈网络，使用SwiGLU激活"""
    def __init__(self, dim, mult=4, dropout=0.1):
        super().__init__()
        
        self.norm = nn.LayerNorm(dim)
        
        # SwiGLU激活提高表达能力
        self.proj_in = nn.Linear(dim, dim * mult * 2)
        self.proj_out = nn.Linear(dim * mult, dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        """前向传播
        
        Args:
            x: 输入特征 [batch_size, seq_len, dim]
            
        Returns:
            x: 输出特征
        """
        x = self.norm(x)
        
        # SwiGLU激活
        x1, x2 = self.proj_in(x).chunk(2, dim=-1)
        x = F.silu(x1) * x2
        
        # 输出投影
        x = self.dropout(self.proj_out(x))
        
        return x

class MultiHeadAttention(nn.Module):
    """优化的多头注意力模块"""
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.1):
        super().__init__()
        
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5
        
        self.norm = nn.LayerNorm(dim)
        self.qkv_proj = nn.Linear(dim, inner_dim * 3)
        self.out_proj = nn.Linear(inner_dim, dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None, cache=None):
        """前向传播，支持注意力缓存
        
        Args:
            x: 输入特征 [batch_size, seq_len, dim]
            mask: 可选的注意力掩码
            cache: 可选的k/v缓存
            
        Returns:
            out: 注意力输出
            updated_cache: 更新后的缓存(如果使用缓存)
        """
        batch_size, seq_len, _ = x.shape
        
        # 归一化
        x = self.norm(x)
        
        # 投影Q、K、V
        qkv = self.qkv_proj(x)
        qkv = qkv.reshape(batch_size, seq_len, 3, self.heads, -1)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, batch, heads, seq_len, dim_head]
        q, k, v = qkv[0], qkv[1], qkv[2]  # 各 [batch, heads, seq_len, dim_head]
        
        # 处理KV缓存(用于流式推理)
        if cache is not None:
            cached_k, cached_v = cache
            
            if cached_k is not None and cached_v is not None:
                # 合并缓存的KV与当前KV
                k = torch.cat([cached_k, k], dim=2)
                v = torch.cat([cached_v, v], dim=2)
                
            # 为下一次流式推理更新缓存
            # 保持缓存大小有限，避免过长序列
            if k.size(2) > 128:  # 限制缓存的最大长度
                k = k[:, :, -128:]
                v = v[:, :, -128:]
                
            updated_cache = (k, v)
        else:
            updated_cache = None
        
        # 计算注意力分数
        attn_scores = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        
        # 应用掩码(如果提供)
        if mask is not None:
            mask = mask.unsqueeze(1)  # [batch, 1, seq_len, seq_len]
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)
        
        # 注意力权重并应用dropout
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # 应用注意力
        out = torch.matmul(attn_weights, v)
        out = out.transpose(1, 2).reshape(batch_size, seq_len, -1)
        
        # 输出投影
        out = self.out_proj(out)
        
        if cache is not None:
            return out, updated_cache
        return out

class ConformerConvolution(nn.Module):
    """Conformer卷积模块，针对流式处理优化"""
    def __init__(self, dim, expansion_factor=2, kernel_size=31, dropout=0.1):
        super().__init__()
        
        inner_dim = dim * expansion_factor
        padding = (kernel_size - 1) // 2
        
        self.norm = nn.LayerNorm(dim)
        self.pointwise_conv1 = nn.Conv1d(dim, inner_dim * 2, kernel_size=1)
        
        # 优化：使用分组卷积而不是depthwise，对尺寸敏感的操作更友好
        self.depthwise_conv = nn.Conv1d(
            inner_dim, inner_dim, 
            kernel_size=kernel_size, 
            padding=padding, 
            groups=inner_dim // 8  # 8组卷积，增强并行性
        )
        
        self.batch_norm = nn.BatchNorm1d(inner_dim)
        self.pointwise_conv2 = nn.Conv1d(inner_dim, dim, kernel_size=1)
        self.dropout = nn.Dropout(dropout)
        
        # 缓存卷积状态用于流式处理
        self.cache_size = 0
        self.kernel_size = kernel_size
        
    def forward(self, x, cache=None):
        """前向传播，支持卷积状态缓存
        
        Args:
            x: 输入特征 [batch_size, seq_len, dim]
            cache: 可选的卷积状态缓存
            
        Returns:
            out: 卷积输出
            updated_cache: 更新后的缓存状态(如果使用缓存)
        """
        batch_size, seq_len, _ = x.shape
        
        # 归一化
        x = self.norm(x)
        
        # 转换为卷积格式 [batch, channels, seq_len]
        x = x.transpose(1, 2)
        
        # 第一个逐点卷积
        x = self.pointwise_conv1(x)
        x = F.glu(x, dim=1)  # 门控线性单元激活
        
        # 处理卷积状态缓存
        if cache is not None:
            cached_x = cache
            
            if cached_x is not None:
                # 合并缓存状态与当前输入
                x = torch.cat([cached_x, x], dim=2)
                
            # 计算要保留的历史大小
            cache_size = self.kernel_size - 1
            
            # 更新卷积状态缓存
            if x.size(2) > cache_size:
                updated_cache = x[:, :, -cache_size:]
            else:
                updated_cache = x
        else:
            updated_cache = None
        
        # 深度卷积
        x = self.depthwise_conv(x)
        x = self.batch_norm(x)
        x = F.silu(x)  # SiLU (Swish)激活
        
        # 第二个逐点卷积
        x = self.pointwise_conv2(x)
        x = self.dropout(x)
        
        # 转回序列格式 [batch, seq_len, dim]
        x = x.transpose(1, 2)
        
        if cache is not None:
            # 对于流式处理，只返回实际处理的新输入
            if cached_x is not None:
                cached_len = cached_x.size(2)
                x = x[:, -seq_len:, :]
            
            return x, updated_cache
        
        return x

class AttentivePooling(nn.Module):
    """增强的注意力池化层，更稳定的输出表示"""
    def __init__(self, dim):
        super().__init__()
        
        # 注意力计算模块
        self.attention = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Tanh(),
            nn.Linear(dim, 1)
        )
        
        # 新增: 位置偏好嵌入，偏向最近的帧
        self.position_bias = nn.Parameter(torch.zeros(1, 1000, 1))  # 最多1000帧
        nn.init.normal_(self.position_bias, std=0.02)
        
        # 新增: 输出稳定性增强层
        self.output_stabilizer = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Dropout(0.1)
        )
        
    def forward(self, x):
        """应用注意力池化
        
        Args:
            x: 输入特征 [batch_size, seq_len, dim]
            
        Returns:
            weighted_x: 池化后的特征 [batch_size, dim]
        """
        batch_size, seq_len, _ = x.shape
        
        # 计算注意力权重
        attn_weights = self.attention(x)  # [batch_size, seq_len, 1]
        
        # 应用位置偏好 - 偏向最近的帧
        position_weights = self.position_bias[:, :seq_len, :]
        # 指数增长权重，后面的帧权重更高
        position_factor = torch.exp(torch.linspace(0, 2, seq_len)).reshape(1, seq_len, 1).to(x.device)
        attn_weights = attn_weights + position_weights * position_factor
        
        # Softmax归一化
        attn_weights = F.softmax(attn_weights, dim=1)
        
        # 应用注意力加权
        weighted_x = torch.sum(x * attn_weights, dim=1)  # [batch_size, dim]
        
        # 应用输出稳定器
        weighted_x = self.output_stabilizer(weighted_x)
        
        return weighted_x

class StreamingConformer(nn.Module):
    """流式Conformer模型，针对语音命令识别优化"""
    def __init__(self, input_dim=48, hidden_dim=192, num_classes=4, 
                 num_layers=6, num_heads=8, dropout=0.1, 
                 kernel_size=15, expansion_factor=4):
        super().__init__()
        
        # 特征投影层
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        
        # 位置编码
        self.pos_encoding = RelativePositionalEncoding(hidden_dim, max_len=1000)
        
        # Conformer编码器层
        self.conformer_layers = nn.ModuleList([
            ConformerBlock(
                dim=hidden_dim,
                dim_head=hidden_dim // num_heads,
                heads=num_heads,
                ff_mult=expansion_factor,
                conv_expansion_factor=2,
                conv_kernel_size=kernel_size,
                attn_dropout=dropout,
                ff_dropout=dropout,
                conv_dropout=dropout
            ) for _ in range(num_layers)
        ])
        
        # 特征池化层 - 使用注意力池化替代简单平均
        self.attention_pooling = AttentivePooling(hidden_dim)
        
        # 输出层 - 添加一个额外的隐藏层增强表达能力
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim*2),
            nn.LayerNorm(hidden_dim*2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim*2, num_classes)
        )
        
        # 缓存用于流式推理
        self.reset_streaming_state()
    
    def reset_streaming_state(self):
        """重置流式状态"""
        self.cached_features = None
        self.cache_size = 0
        self.max_cache_size = MAX_CACHED_FRAMES  # 最大缓存帧数
        self.layer_caches = [None] * len(self.conformer_layers)
    
    def forward_streaming(self, x, cache_states=None, return_cache=True):
        """流式前向传播，支持特征缓存
        
        Args:
            x: 输入特征 [batch_size, seq_len, input_dim]
            cache_states: 可选的缓存状态
            return_cache: 是否返回更新的缓存状态
            
        Returns:
            logits: 分类输出
            cache_states: 更新的缓存状态(如果return_cache=True)
        """
        batch_size, seq_len, _ = x.shape
        
        # 先对当前输入进行投影
        x = self.input_projection(x)
        
        if cache_states is not None:
            # 使用提供的缓存状态
            cached_features, layer_caches = cache_states
            
            if cached_features is not None:
                # 拼接缓存特征和当前特征（已经投影过）
                cached_seq_len = cached_features.size(1)
                
                # 确保批次大小一致
                if cached_features.size(0) != batch_size:
                    cached_features = cached_features.repeat(batch_size, 1, 1)
                
                # 拼接缓存特征和已经投影的当前特征
                x = torch.cat([cached_features, x], dim=1)
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
        pooled = self.attention_pooling(x)
        
        # 分类
        logits = self.classifier(pooled)
        
        # 更新要缓存的特征
        if return_cache:
            if new_seq_len > cache_size:
                # 只保留最后cache_size帧
                new_cache_features = x[:, -cache_size:, :].detach()
            else:
                new_cache_features = x.detach()
            
            return logits, (new_cache_features, new_layer_caches)
        
        return logits

    def forward(self, x):
        """标准前向传播
        
        Args:
            x: 输入特征 [batch_size, seq_len, input_dim]
            
        Returns:
            logits: 分类输出
        """
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
            x: 输入特征 [batch_size, seq_len, input_dim]
            cached_states: 可选的缓存状态
            
        Returns:
            pred: 预测类别索引
            conf: 预测置信度
            new_cached_states: 更新的缓存状态
        """
        # 使用温度缩放提高置信度可靠性
        temperature = 0.8
        
        # 前向传播获取logits
        logits, new_cached_states = self.forward_streaming(x, cached_states, True)
        
        # 应用温度缩放
        scaled_logits = logits / temperature
        
        # 获取预测类别和置信度
        probs = F.softmax(scaled_logits, dim=-1)
        conf, pred = torch.max(probs, dim=-1)
        
        # 应用平滑处理以减少抖动
        if cached_states is not None:
            # 如果有之前的预测，尝试进行平滑
            prev_logits = getattr(self, '_prev_logits', None)
            if prev_logits is not None:
                # 指数移动平均平滑logits
                alpha = 0.7  # 平滑因子
                smoothed_logits = alpha * scaled_logits + (1 - alpha) * prev_logits
                
                # 重新计算概率和预测
                smooth_probs = F.softmax(smoothed_logits, dim=-1)
                smooth_conf, smooth_pred = torch.max(smooth_probs, dim=-1)
                
                # 如果平滑后的预测置信度足够高，则使用它
                if smooth_conf.item() > 0.4:  # 使用较低的阈值
                    pred = smooth_pred
                    conf = smooth_conf
        
        # 保存当前logits用于下次平滑
        self._prev_logits = scaled_logits.detach()
        
        return pred, conf, new_cached_states 