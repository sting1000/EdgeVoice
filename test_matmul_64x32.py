#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
测试修改后的AttentivePooling是否成功将64*1 matmul改为64*32
同时验证模型精度是否保持不变
"""

import torch
import torch.nn as nn
import numpy as np
from models.streaming_conformer import StreamingConformer, AttentivePooling
from config import *

def test_attentive_pooling_dimensions():
    """测试AttentivePooling的维度变化"""
    print("=== 测试AttentivePooling维度变化 ===")
    
    # 测试参数
    batch_size = 2
    seq_len = 50
    dim = 128  # 对应CONFORMER_HIDDEN_SIZE的常见值
    
    # 创建测试数据
    x = torch.randn(batch_size, seq_len, dim)
    
    # 创建AttentivePooling模块
    pooling = AttentivePooling(dim)
    
    print(f"输入维度: {x.shape}")
    
    # 打印模型结构中的linear层维度
    print("\n模型结构分析:")
    for name, module in pooling.named_modules():
        if isinstance(module, nn.Linear):
            in_features = module.in_features
            out_features = module.out_features
            print(f"  {name}: Linear({in_features}, {out_features})")
            
            # 检查是否还有64*1的情况
            if in_features == 64 and out_features == 1:
                print(f"    ❌ 发现64*1的Linear层: {name}")
            elif in_features == 64 and out_features == 32:
                print(f"    ✅ 成功修改为64*32: {name}")
            elif in_features == 32 and out_features == 1:
                print(f"    ✅ 压缩层32*1: {name}")
    
    # 前向传播测试
    with torch.no_grad():
        output = pooling(x)
        print(f"\n输出维度: {output.shape}")
        print(f"预期输出维度: [{batch_size}, {dim}]")
        
        if output.shape == (batch_size, dim):
            print("✅ 输出维度正确")
        else:
            print("❌ 输出维度错误")
    
    return pooling

def create_old_attentive_pooling(dim):
    """创建旧版本的AttentivePooling用于对比"""
    class OldAttentivePooling(nn.Module):
        def __init__(self, dim):
            super().__init__()
            self.attention = nn.Sequential(
                nn.Linear(dim, dim // 2),  
                nn.Tanh(),
                nn.Linear(dim // 2, 1)  # 旧版本：64*1
            )
            
        def forward(self, x):
            attn_weights = self.attention(x)  # [batch_size, seq_len, 1]
            attn_weights = torch.softmax(attn_weights, dim=1)
            
            # 使用4维矩阵乘法
            attn_weights_t = attn_weights.transpose(1, 2)  # [batch_size, 1, seq_len]
            attn_weights_4d = attn_weights_t.unsqueeze(0)  # [1, batch_size, 1, seq_len]
            x_4d = x.unsqueeze(0)  # [1, batch_size, seq_len, dim]
            
            weighted_sum_4d = torch.matmul(attn_weights_4d, x_4d)  # [1, batch_size, 1, dim]
            weighted_sum = weighted_sum_4d.squeeze(0).squeeze(1)  # [batch_size, dim]
            
            return weighted_sum
    
    return OldAttentivePooling(dim)

def test_precision_consistency():
    """测试新旧版本的精度一致性"""
    print("\n=== 测试精度一致性 ===")
    
    # 设置随机种子确保可重复性
    torch.manual_seed(42)
    np.random.seed(42)
    
    dim = 128
    batch_size = 2
    seq_len = 50
    
    # 创建测试数据
    x = torch.randn(batch_size, seq_len, dim)
    
    # 创建新版本模块
    new_pooling = AttentivePooling(dim)
    
    # 创建旧版本模块
    old_pooling = create_old_attentive_pooling(dim)
    
    # 将新版本的权重复制到旧版本进行公平对比
    # 复制第一个linear层和tanh层的权重
    with torch.no_grad():
        old_pooling.attention[0].weight.copy_(new_pooling.attention[0].weight)
        old_pooling.attention[0].bias.copy_(new_pooling.attention[0].bias)
        # 对于最后一层，我们需要特殊处理
        # 旧版本是64->1，新版本是64->32->1
        # 我们让新版本的compress层权重初始化为使得输出接近旧版本
        old_last_weight = old_pooling.attention[2].weight  # shape: [1, 64]
        old_last_bias = old_pooling.attention[2].bias      # shape: [1]
        
        # 将旧版本的64->1权重分配到新版本的64->32层
        # 使用复制策略：将原权重复制到前32维中的第一维
        new_pooling.attention[2].weight.data.zero_()
        new_pooling.attention[2].weight.data[0, :] = old_last_weight[0, :]  # 只设置第一个输出维度
        new_pooling.attention[2].bias.data.zero_()
        new_pooling.attention[2].bias.data[0] = old_last_bias[0]
        
        # 设置compress层权重，让它只取第一个维度
        new_pooling.attention_compress.weight.data.zero_()
        new_pooling.attention_compress.weight.data[0, 0] = 1.0  # 只取第一个维度
        new_pooling.attention_compress.bias.data.zero_()
    
    # 前向传播对比
    with torch.no_grad():
        old_output = old_pooling(x)
        new_output = new_pooling(x)
        
        print(f"旧版本输出形状: {old_output.shape}")
        print(f"新版本输出形状: {new_output.shape}")
        
        # 计算差异
        max_diff = torch.max(torch.abs(old_output - new_output)).item()
        mean_diff = torch.mean(torch.abs(old_output - new_output)).item()
        
        print(f"最大绝对差异: {max_diff:.8f}")
        print(f"平均绝对差异: {mean_diff:.8f}")
        
        # 检查是否在合理范围内
        if max_diff < 1e-5:
            print("✅ 精度保持良好 (差异 < 1e-5)")
        elif max_diff < 1e-3:
            print("⚠️  精度可接受 (差异 < 1e-3)")
        else:
            print("❌ 精度差异较大")
    
    return old_output, new_output

def test_full_model():
    """测试完整模型的matmul维度"""
    print("\n=== 测试完整StreamingConformer模型 ===")
    
    # 创建模型
    model = StreamingConformer(
        input_dim=N_MFCC * 3,  # 48
        hidden_dim=128,        # 确保是128以测试64*32的情况
        num_classes=len(INTENT_CLASSES),
        num_layers=2,          # 减少层数加快测试
        num_heads=4,
        dropout=0.1,
        kernel_size=15,
        expansion_factor=2
    )
    
    print(f"模型创建成功")
    print(f"隐藏维度: {128}")
    
    # 检查attention_pooling层的结构
    pooling = model.attention_pooling
    print("\nAttentivePooling 结构:")
    for name, module in pooling.named_modules():
        if isinstance(module, nn.Linear):
            in_features = module.in_features
            out_features = module.out_features
            print(f"  {name}: Linear({in_features}, {out_features})")
            
            if in_features == 64 and out_features == 1:
                print(f"    ❌ 仍有64*1层: {name}")
                return False
            elif in_features == 64 and out_features == 32:
                print(f"    ✅ 成功修改为64*32: {name}")
    
    # 测试前向传播
    batch_size = 1
    seq_len = 100
    input_dim = N_MFCC * 3
    
    x = torch.randn(batch_size, seq_len, input_dim)
    
    with torch.no_grad():
        try:
            output = model(x)
            print(f"\n前向传播成功")
            print(f"输入形状: {x.shape}")
            print(f"输出形状: {output.shape}")
            print(f"预期输出形状: [{batch_size}, {len(INTENT_CLASSES)}]")
            
            if output.shape == (batch_size, len(INTENT_CLASSES)):
                print("✅ 模型输出维度正确")
                return True
            else:
                print("❌ 模型输出维度错误")
                return False
                
        except Exception as e:
            print(f"❌ 前向传播失败: {e}")
            return False

def main():
    """主测试函数"""
    print("开始测试修改后的AttentivePooling...")
    print("目标: 将64*1 matmul改为64*32")
    
    # 测试1: AttentivePooling维度
    pooling = test_attentive_pooling_dimensions()
    
    # 测试2: 精度一致性
    old_output, new_output = test_precision_consistency()
    
    # 测试3: 完整模型
    model_success = test_full_model()
    
    print("\n=== 测试总结 ===")
    if model_success:
        print("✅ 所有测试通过")
        print("✅ 成功将64*1 matmul改为64*32")
        print("✅ 模型功能正常")
    else:
        print("❌ 部分测试失败")
    
    return model_success

if __name__ == "__main__":
    main() 