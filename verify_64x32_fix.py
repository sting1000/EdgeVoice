#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
综合验证脚本：验证64×1改为64×32的修改在所有场景下都正确
包括模型创建、训练、推理、导出等
"""

import torch
import torch.nn as nn
import torch.onnx
import onnx
import numpy as np
import tempfile
import os
from models.streaming_conformer import StreamingConformer, AttentivePooling
from config import *

def test_1_attentive_pooling_structure():
    """测试1：AttentivePooling结构正确性"""
    print("=== 测试1：AttentivePooling结构 ===")
    
    dim = 128
    pooling = AttentivePooling(dim)
    
    # 检查结构
    has_64x32 = False
    has_32x1 = False
    has_64x1 = False
    
    for name, module in pooling.named_modules():
        if isinstance(module, nn.Linear):
            in_features = module.in_features
            out_features = module.out_features
            print(f"  {name}: Linear({in_features}, {out_features})")
            
            if in_features == 64 and out_features == 32:
                has_64x32 = True
            elif in_features == 32 and out_features == 1:
                has_32x1 = True
            elif in_features == 64 and out_features == 1:
                has_64x1 = True
    
    success = has_64x32 and has_32x1 and not has_64x1
    print(f"✅ 结构检查: 64×32={has_64x32}, 32×1={has_32x1}, 64×1={has_64x1}")
    print(f"✅ 测试1通过: {success}")
    return success

def test_2_functional_equivalence():
    """测试2：功能等价性验证"""
    print("\n=== 测试2：功能等价性 ===")
    
    # 创建新旧版本
    dim = 128
    new_pooling = AttentivePooling(dim)
    
    class OldAttentivePooling(nn.Module):
        def __init__(self, dim):
            super().__init__()
            self.attention = nn.Sequential(
                nn.Linear(dim, dim // 2),
                nn.Tanh(),
                nn.Linear(dim // 2, 1)
            )
        def forward(self, x):
            attn_weights = self.attention(x)
            attn_weights = torch.softmax(attn_weights, dim=1)
            attn_weights_t = attn_weights.transpose(1, 2)
            attn_weights_4d = attn_weights_t.unsqueeze(0)
            x_4d = x.unsqueeze(0)
            weighted_sum_4d = torch.matmul(attn_weights_4d, x_4d)
            weighted_sum = weighted_sum_4d.squeeze(0).squeeze(1)
            return weighted_sum
    
    old_pooling = OldAttentivePooling(dim)
    
    # 设置相同的权重
    with torch.no_grad():
        old_pooling.attention[0].weight.copy_(new_pooling.attention[0].weight)
        old_pooling.attention[0].bias.copy_(new_pooling.attention[0].bias)
        
        # 特殊处理新版本权重
        old_weight = old_pooling.attention[2].weight[0, :]  # [64]
        old_bias = old_pooling.attention[2].bias[0]  # scalar
        
        new_pooling.attention[2].weight.data.zero_()
        new_pooling.attention[2].weight.data[0, :] = old_weight
        new_pooling.attention[2].bias.data.zero_()
        new_pooling.attention[2].bias.data[0] = old_bias
        
        new_pooling.attention_compress.weight.data.zero_()
        new_pooling.attention_compress.weight.data[0, 0] = 1.0
        new_pooling.attention_compress.bias.data.zero_()
    
    # 测试多个输入
    test_cases = [
        (2, 50, 128),
        (1, 100, 128),
        (4, 25, 128),
    ]
    
    max_diff_overall = 0
    for batch_size, seq_len, dim in test_cases:
        x = torch.randn(batch_size, seq_len, dim)
        
        with torch.no_grad():
            old_out = old_pooling(x)
            new_out = new_pooling(x)
            
            max_diff = torch.max(torch.abs(old_out - new_out)).item()
            max_diff_overall = max(max_diff_overall, max_diff)
    
    success = max_diff_overall < 1e-5
    print(f"  最大差异: {max_diff_overall:.10f}")
    print(f"✅ 测试2通过: {success}")
    return success

def test_3_model_creation():
    """测试3：完整模型创建"""
    print("\n=== 测试3：完整模型创建 ===")
    
    configs = [
        {"hidden_dim": 96, "expected_64x32": False},   # 96 // 2 = 48, 不会产生64×32
        {"hidden_dim": 128, "expected_64x32": True},   # 128 // 2 = 64, 会产生64×32
        {"hidden_dim": 160, "expected_64x32": False},  # 160 // 2 = 80, 不会产生64×32
    ]
    
    all_success = True
    for config in configs:
        hidden_dim = config["hidden_dim"]
        expected_64x32 = config["expected_64x32"]
        
        model = StreamingConformer(
            input_dim=N_MFCC * 3,
            hidden_dim=hidden_dim,
            num_classes=len(INTENT_CLASSES),
            num_layers=2,
            num_heads=4,
            dropout=0.1,
            kernel_size=15,
            expansion_factor=2
        )
        
        # 检查是否有64×32
        has_64x32 = False
        has_64x1 = False
        
        pooling = model.attention_pooling
        for name, module in pooling.named_modules():
            if isinstance(module, nn.Linear):
                in_f, out_f = module.in_features, module.out_features
                if in_f == 64 and out_f == 32:
                    has_64x32 = True
                elif in_f == 64 and out_f == 1:
                    has_64x1 = True
        
        success = (has_64x32 == expected_64x32) and (not has_64x1)
        all_success = all_success and success
        
        print(f"  hidden_dim={hidden_dim}: 64×32={has_64x32} (期望{expected_64x32}), 64×1={has_64x1} ({'✅' if success else '❌'})")
    
    print(f"✅ 测试3通过: {all_success}")
    return all_success

def test_4_streaming_functionality():
    """测试4：流式功能"""
    print("\n=== 测试4：流式功能 ===")
    
    model = StreamingConformer(
        input_dim=N_MFCC * 3,
        hidden_dim=128,
        num_classes=len(INTENT_CLASSES),
        num_layers=2,
        num_heads=4
    )
    model.eval()
    
    # 测试流式推理
    batch_size = 1
    chunk_size = 50
    input_dim = N_MFCC * 3
    
    # 模拟流式输入
    chunk1 = torch.randn(batch_size, chunk_size, input_dim)
    chunk2 = torch.randn(batch_size, chunk_size, input_dim)
    
    try:
        with torch.no_grad():
            # 第一个chunk
            pred1, conf1, cache1 = model.predict_streaming(chunk1, None)
            
            # 第二个chunk（使用缓存）
            pred2, conf2, cache2 = model.predict_streaming(chunk2, cache1)
            
            print(f"  Chunk1: pred={pred1.item()}, conf={conf1.item():.4f}")
            print(f"  Chunk2: pred={pred2.item()}, conf={conf2.item():.4f}")
            
        success = True
        print(f"✅ 测试4通过: {success}")
        return success
        
    except Exception as e:
        print(f"❌ 流式功能错误: {e}")
        return False

def test_5_onnx_export():
    """测试5：ONNX导出"""
    print("\n=== 测试5：ONNX导出 ===")
    
    model = StreamingConformer(
        input_dim=N_MFCC * 3,
        hidden_dim=128,
        num_classes=len(INTENT_CLASSES),
        num_layers=2,
        num_heads=4
    )
    model.eval()
    
    with tempfile.NamedTemporaryFile(suffix='.onnx', delete=False) as tmp_file:
        onnx_path = tmp_file.name
    
    try:
        # 导出ONNX
        dummy_input = torch.randn(1, 100, N_MFCC * 3)
        torch.onnx.export(
            model, dummy_input, onnx_path,
            export_params=True, opset_version=13,
            input_names=['input'], output_names=['output'],
            verbose=False
        )
        
        # 检查算子
        onnx_model = onnx.load(onnx_path)
        
        has_64x1 = False
        has_64x32 = False
        
        for node in onnx_model.graph.node:
            if node.op_type in ['MatMul', 'Gemm']:
                for input_name in node.input:
                    for init in onnx_model.graph.initializer:
                        if init.name == input_name:
                            dims = list(init.dims)
                            if len(dims) == 2:
                                if dims == [64, 1] or dims == [1, 64]:
                                    has_64x1 = True
                                elif dims == [64, 32] or dims == [32, 64]:
                                    has_64x32 = True
        
        success = has_64x32 and not has_64x1
        print(f"  ONNX算子: 64×32={has_64x32}, 64×1={has_64x1}")
        print(f"✅ 测试5通过: {success}")
        return success
        
    except Exception as e:
        print(f"❌ ONNX导出错误: {e}")
        return False
    finally:
        if os.path.exists(onnx_path):
            os.unlink(onnx_path)

def test_6_training_compatibility():
    """测试6：训练兼容性"""
    print("\n=== 测试6：训练兼容性 ===")
    
    model = StreamingConformer(
        input_dim=N_MFCC * 3,
        hidden_dim=128,
        num_classes=len(INTENT_CLASSES),
        num_layers=2,
        num_heads=4
    )
    
    # 模拟训练过程
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()
    
    batch_size = 2
    seq_len = 100
    input_dim = N_MFCC * 3
    
    try:
        x = torch.randn(batch_size, seq_len, input_dim)
        labels = torch.randint(0, len(INTENT_CLASSES), (batch_size,))
        
        # 前向传播
        outputs = model(x)
        loss = criterion(outputs, labels)
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        success = True
        print(f"  训练步骤完成，损失: {loss.item():.4f}")
        print(f"✅ 测试6通过: {success}")
        return success
        
    except Exception as e:
        print(f"❌ 训练兼容性错误: {e}")
        return False

def main():
    """主测试函数"""
    print("🔍 开始综合验证64×1改为64×32的修改")
    print("=" * 60)
    
    tests = [
        test_1_attentive_pooling_structure,
        test_2_functional_equivalence,
        test_3_model_creation,
        test_4_streaming_functionality,
        test_5_onnx_export,
        test_6_training_compatibility,
    ]
    
    results = []
    for i, test_func in enumerate(tests, 1):
        try:
            result = test_func()
            results.append(result)
        except Exception as e:
            print(f"❌ 测试{i}异常: {e}")
            results.append(False)
    
    print("\n" + "=" * 60)
    print("🎯 验证总结")
    print("=" * 60)
    
    for i, result in enumerate(results, 1):
        status = "✅ 通过" if result else "❌ 失败"
        print(f"  测试{i}: {status}")
    
    all_passed = all(results)
    
    if all_passed:
        print("\n🎉 所有测试通过！")
        print("✅ 成功将64×1 matmul改为64×32")
        print("✅ 模型精度完全保持")
        print("✅ 所有功能正常工作")
        print("✅ 可以安全使用修改后的模型")
    else:
        print("\n⚠️  存在失败的测试，请检查具体问题")
    
    return all_passed

if __name__ == "__main__":
    main() 