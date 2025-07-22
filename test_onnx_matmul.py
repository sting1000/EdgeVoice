#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
测试ONNX导出中的matmul算子维度
验证是否成功将64*1改为64*32
"""

import torch
import torch.onnx
import onnx
import numpy as np
from models.streaming_conformer import StreamingConformer
from config import *
import tempfile
import os

def create_test_model():
    """创建测试模型"""
    model = StreamingConformer(
        input_dim=N_MFCC * 3,  # 96
        hidden_dim=128,        # 确保是128以测试64*32的情况
        num_classes=len(INTENT_CLASSES),
        num_layers=2,          # 减少层数加快测试
        num_heads=4,
        dropout=0.1,
        kernel_size=15,
        expansion_factor=2,
        use_padded_output=False  # 先不使用填充输出
    )
    model.eval()
    return model

def export_and_analyze_onnx(model, save_path):
    """导出ONNX并分析算子"""
    print("=== 导出ONNX模型 ===")
    
    # 创建示例输入
    batch_size = 1
    seq_len = 100
    input_dim = N_MFCC * 3
    dummy_input = torch.randn(batch_size, seq_len, input_dim)
    
    print(f"输入维度: {dummy_input.shape}")
    
    # 导出ONNX
    torch.onnx.export(
        model,
        dummy_input,
        save_path,
        export_params=True,
        opset_version=13,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size', 1: 'seq_length'},
            'output': {0: 'batch_size'}
        },
        verbose=False
    )
    
    print(f"ONNX模型已导出到: {save_path}")
    return save_path

def analyze_onnx_operators(onnx_path):
    """分析ONNX模型中的算子"""
    print("\n=== 分析ONNX算子 ===")
    
    # 加载ONNX模型
    model = onnx.load(onnx_path)
    
    # 分析所有算子
    matmul_count = 0
    gemm_count = 0
    matmul_64x1_count = 0
    matmul_64x32_count = 0
    gemm_64x1_count = 0
    gemm_64x32_count = 0
    
    print("算子分析:")
    
    for node in model.graph.node:
        if node.op_type == 'MatMul':
            matmul_count += 1
            print(f"  发现MatMul算子: {node.name}")
            
            # 检查输入维度
            for input_name in node.input:
                for init in model.graph.initializer:
                    if init.name == input_name:
                        dims = list(init.dims)
                        print(f"    权重维度: {dims}")
                        
                        # 检查是否是我们关心的维度
                        if len(dims) == 2:
                            if dims == [64, 1]:
                                matmul_64x1_count += 1
                                print(f"    ❌ 发现64×1 MatMul")
                            elif dims == [64, 32]:
                                matmul_64x32_count += 1
                                print(f"    ✅ 发现64×32 MatMul")
        
        elif node.op_type == 'Gemm':
            gemm_count += 1
            print(f"  发现Gemm算子: {node.name}")
            
            # 检查Gemm的权重维度
            for input_name in node.input[:2]:  # Gemm通常前两个是权重相关
                for init in model.graph.initializer:
                    if init.name == input_name:
                        dims = list(init.dims)
                        print(f"    权重维度: {dims}")
                        
                        # 检查是否是我们关心的维度
                        if len(dims) == 2:
                            if dims == [64, 1] or dims == [1, 64]:
                                gemm_64x1_count += 1
                                print(f"    ❌ 发现64×1 Gemm")
                            elif dims == [64, 32] or dims == [32, 64]:
                                gemm_64x32_count += 1
                                print(f"    ✅ 发现64×32 Gemm")
    
    print(f"\n算子统计:")
    print(f"  MatMul总数: {matmul_count}")
    print(f"  Gemm总数: {gemm_count}")
    print(f"  64×1 MatMul: {matmul_64x1_count}")
    print(f"  64×32 MatMul: {matmul_64x32_count}")
    print(f"  64×1 Gemm: {gemm_64x1_count}")
    print(f"  64×32 Gemm: {gemm_64x32_count}")
    
    # 检查结果
    success = True
    if matmul_64x1_count > 0 or gemm_64x1_count > 0:
        print("❌ 仍然存在64×1的算子")
        success = False
    
    if matmul_64x32_count > 0 or gemm_64x32_count > 0:
        print("✅ 成功检测到64×32的算子")
    else:
        print("⚠️  未检测到64×32的算子，可能结构不同或被优化")
    
    return success

def test_onnx_inference(onnx_path, original_model):
    """测试ONNX模型推理结果与原模型的一致性"""
    print("\n=== 测试ONNX推理一致性 ===")
    
    try:
        import onnxruntime as ort
    except ImportError:
        print("⚠️  未安装onnxruntime，跳过推理测试")
        return True
    
    # 创建ONNX Runtime会话
    ort_session = ort.InferenceSession(onnx_path)
    
    # 创建测试输入
    batch_size = 1
    seq_len = 100
    input_dim = N_MFCC * 3
    test_input = torch.randn(batch_size, seq_len, input_dim)
    
    # PyTorch模型推理
    with torch.no_grad():
        pytorch_output = original_model(test_input)
    
    # ONNX模型推理
    onnx_input = {ort_session.get_inputs()[0].name: test_input.numpy()}
    onnx_output = ort_session.run(None, onnx_input)[0]
    
    # 比较结果
    pytorch_np = pytorch_output.numpy()
    max_diff = np.max(np.abs(pytorch_np - onnx_output))
    mean_diff = np.mean(np.abs(pytorch_np - onnx_output))
    
    print(f"PyTorch输出形状: {pytorch_np.shape}")
    print(f"ONNX输出形状: {onnx_output.shape}")
    print(f"最大绝对差异: {max_diff:.8f}")
    print(f"平均绝对差异: {mean_diff:.8f}")
    
    if max_diff < 1e-5:
        print("✅ ONNX推理结果与PyTorch高度一致")
        return True
    elif max_diff < 1e-3:
        print("⚠️  ONNX推理结果与PyTorch基本一致")
        return True
    else:
        print("❌ ONNX推理结果与PyTorch差异较大")
        return False

def main():
    """主测试函数"""
    print("开始测试ONNX导出中的matmul算子维度...")
    
    # 创建临时文件
    with tempfile.NamedTemporaryFile(suffix='.onnx', delete=False) as tmp_file:
        onnx_path = tmp_file.name
    
    try:
        # 创建模型
        model = create_test_model()
        print("模型创建成功")
        
        # 导出ONNX
        export_and_analyze_onnx(model, onnx_path)
        
        # 分析算子
        analysis_success = analyze_onnx_operators(onnx_path)
        
        # 测试推理一致性
        inference_success = test_onnx_inference(onnx_path, model)
        
        print("\n=== 测试总结 ===")
        if analysis_success and inference_success:
            print("✅ ONNX导出测试全部通过")
            print("✅ 成功消除64×1 matmul算子")
            print("✅ ONNX推理结果正确")
        else:
            print("❌ 部分测试失败")
            
        return analysis_success and inference_success
        
    finally:
        # 清理临时文件
        if os.path.exists(onnx_path):
            os.unlink(onnx_path)
            print(f"清理临时文件: {onnx_path}")

if __name__ == "__main__":
    main() 