#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
将训练好的EdgeVoice模型转换为ONNX格式
支持快速分类器和精确分类器的转换
"""

import os
import argparse
import torch
from config import *
from models.fast_classifier import FastIntentClassifier
from models.precise_classifier import PreciseIntentClassifier

def convert_fast_model_to_onnx(model_path, onnx_save_path=None, dynamic_axes=True):
    """
    将快速分类器模型转换为ONNX格式
    
    Args:
        model_path: PyTorch模型路径
        onnx_save_path: ONNX模型保存路径（如果为None则根据原模型路径生成）
        dynamic_axes: 是否使用动态轴（用于支持可变输入大小）
    
    Returns:
        onnx_save_path: 导出的ONNX模型路径
    """
    print(f"正在将快速分类器模型转换为ONNX格式...")
    
    # 如果未指定ONNX保存路径，则根据PyTorch模型路径生成
    if onnx_save_path is None:
        onnx_save_path = os.path.splitext(model_path)[0] + '.onnx'
    
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 加载FastIntentClassifier模型
    # 我们使用MFCC特征加上上下文帧，所以特征大小为(N_MFCC * (2*CONTEXT_FRAMES + 1))
    input_size = N_MFCC * (2*CONTEXT_FRAMES + 1)
    model = FastIntentClassifier(input_size=input_size, num_classes=len(INTENT_CLASSES))
    
    # 加载保存的权重
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
    # 创建示例输入
    dummy_input = torch.randn(1, 1, input_size, device=device)  # 批量大小为1，序列长度为1
    
    # 定义ONNX导出的输入和输出名称
    input_names = ["input"]
    output_names = ["output"]
    
    # 定义动态轴（如果需要）
    dynamic_axes_dict = None
    if dynamic_axes:
        dynamic_axes_dict = {
            'input': {0: 'batch_size', 1: 'sequence_length'},  # 动态批量大小和序列长度
            'output': {0: 'batch_size'}
        }
    
    # 导出模型
    torch.onnx.export(
        model,
        dummy_input,
        onnx_save_path,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes_dict,
        export_params=True,
        opset_version=12,
        do_constant_folding=True,
        verbose=False
    )
    
    print(f"快速分类器模型已转换为ONNX格式，保存在: {onnx_save_path}")
    return onnx_save_path

def convert_precise_model_to_onnx(model_path, onnx_save_path=None, dynamic_axes=True):
    """
    将精确分类器模型转换为ONNX格式
    
    Args:
        model_path: PyTorch模型路径
        onnx_save_path: ONNX模型保存路径（如果为None则根据原模型路径生成）
        dynamic_axes: 是否使用动态轴（用于支持可变输入大小）
    
    Returns:
        onnx_save_path: 导出的ONNX模型路径
    """
    print(f"正在将精确分类器模型转换为ONNX格式...")
    
    # 如果未指定ONNX保存路径，则根据PyTorch模型路径生成
    if onnx_save_path is None:
        onnx_save_path = os.path.splitext(model_path)[0] + '.onnx'
    
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 加载PreciseIntentClassifier模型
    model = PreciseIntentClassifier(num_classes=len(INTENT_CLASSES))
    
    # 加载保存的权重
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
    # 创建示例输入
    # 假设我们使用的最大序列长度为128
    max_length = 128
    dummy_input_ids = torch.randint(0, 30522, (1, max_length), device=device)  # 批量大小为1，序列长度为max_length
    dummy_attention_mask = torch.ones((1, max_length), device=device)
    
    # 定义ONNX导出的输入和输出名称
    input_names = ["input_ids", "attention_mask"]
    output_names = ["output"]
    
    # 定义动态轴（如果需要）
    dynamic_axes_dict = None
    if dynamic_axes:
        dynamic_axes_dict = {
            'input_ids': {0: 'batch_size', 1: 'sequence_length'},
            'attention_mask': {0: 'batch_size', 1: 'sequence_length'},
            'output': {0: 'batch_size'}
        }
    
    # 导出模型
    torch.onnx.export(
        model,
        (dummy_input_ids, dummy_attention_mask),
        onnx_save_path,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes_dict,
        export_params=True,
        opset_version=12,
        do_constant_folding=True,
        verbose=False
    )
    
    print(f"精确分类器模型已转换为ONNX格式，保存在: {onnx_save_path}")
    return onnx_save_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='将EdgeVoice模型转换为ONNX格式')
    parser.add_argument('--model_path', type=str, required=True, help='PyTorch模型路径')
    parser.add_argument('--model_type', type=str, choices=['fast', 'precise'], required=True, help='模型类型')
    parser.add_argument('--onnx_path', type=str, default=None, help='ONNX模型保存路径（默认为原模型路径加.onnx后缀）')
    parser.add_argument('--static', action='store_true', help='使用静态输入形状（默认为动态）')
    
    args = parser.parse_args()
    
    if args.model_type == 'fast':
        convert_fast_model_to_onnx(args.model_path, args.onnx_path, not args.static)
    else:
        convert_precise_model_to_onnx(args.model_path, args.onnx_path, not args.static) 