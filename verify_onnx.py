#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
验证ONNX模型中的卷积算子是否为4维
"""

import os
import argparse
import onnx
import numpy as np

def verify_conv_dims(onnx_path):
    """
    验证ONNX模型中的所有卷积算子是否为4维
    
    Args:
        onnx_path: ONNX模型路径
    
    Returns:
        success: 是否全部为4维
    """
    print(f"加载ONNX模型: {onnx_path}")
    model = onnx.load(onnx_path)
    
    # 检查模型是否合法
    try:
        onnx.checker.check_model(model)
        print("模型检查通过")
    except Exception as e:
        print(f"模型检查失败: {str(e)}")
        return False
    
    # 查找所有卷积节点
    conv_nodes = [node for node in model.graph.node if node.op_type == 'Conv']
    print(f"找到 {len(conv_nodes)} 个卷积节点")
    
    # 遍历每个卷积节点，检查其权重维度
    all_4d = True
    for i, node in enumerate(conv_nodes):
        weight_name = node.input[1]  # 卷积权重是第二个输入
        weight_tensor = None
        
        # 查找对应的权重张量
        for init in model.graph.initializer:
            if init.name == weight_name:
                weight_tensor = init
                break
        
        if weight_tensor is None:
            print(f"警告: 找不到卷积节点 {node.name} 的权重")
            continue
        
        dims = weight_tensor.dims
        is_4d = len(dims) == 4
        
        print(f"节点 {i+1}/{len(conv_nodes)}: {node.name}")
        print(f"  权重: {weight_name}")
        print(f"  维度: {dims} ({'4D' if is_4d else '非4D!!'})")
        
        if not is_4d:
            all_4d = False
    
    if all_4d:
        print("\n恭喜！所有卷积节点的权重都是4维的，符合部署要求")
    else:
        print("\n警告：存在非4维的卷积权重，可能不符合部署要求")
    
    return all_4d

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="验证ONNX模型中的卷积算子是否为4维")
    parser.add_argument("--model_path", type=str, required=True, help="ONNX模型路径")
    
    return parser.parse_args()

def main():
    """主函数"""
    # 解析命令行参数
    args = parse_args()
    
    # 检查模型文件是否存在
    if not os.path.exists(args.model_path):
        print(f"错误: 模型文件不存在: {args.model_path}")
        return 1
    
    # 验证卷积维度
    success = verify_conv_dims(args.model_path)
    
    return 0 if success else 1

if __name__ == "__main__":
    exit(main()) 