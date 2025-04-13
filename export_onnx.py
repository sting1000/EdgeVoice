#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
导出PyTorch模型为ONNX格式的独立脚本
"""

import os
import argparse
import torch
import onnx
from train import export_model_to_onnx

def check_and_fix_onnx_conv_dims(onnx_path):
    """
    检查并修复ONNX模型中的卷积算子维度，确保所有卷积操作都是4维的
    
    Args:
        onnx_path: ONNX模型路径
    
    Returns:
        success: 是否成功修复
    """
    try:
        print(f"正在检查ONNX模型的卷积算子维度: {onnx_path}")
        # 加载ONNX模型
        model = onnx.load(onnx_path)
        
        # 检查模型是否合法
        onnx.checker.check_model(model)
        
        # 遍历所有节点，检查卷积算子
        needs_fixing = False
        for node in model.graph.node:
            if node.op_type == 'Conv' and len(node.input) >= 3:  # Conv操作至少有3个输入
                # 检查卷积权重的维度，确保是4维
                for init in model.graph.initializer:
                    if init.name == node.input[1]:  # 输入1是卷积权重
                        if len(init.dims) != 4:
                            print(f"发现3维卷积: {node.name}, 输入: {node.input}")
                            needs_fixing = True
                            break
        
        if needs_fixing:
            print("ONNX模型中存在3维卷积，需要修复。请确保在PyTorch模型中已将所有Conv1d替换为Conv2d。")
            return False
        else:
            print("ONNX模型检查通过，所有卷积操作均为4维。")
            return True
    
    except Exception as e:
        print(f"检查ONNX模型时出错: {str(e)}")
        return False

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="将PyTorch模型导出为ONNX格式")
    parser.add_argument("--model_path", type=str, required=True, help="PyTorch模型路径")
    parser.add_argument("--model_type", type=str, choices=["fast", "streaming"], default="fast", help="模型类型")
    parser.add_argument("--onnx_save_path", type=str, default=None, help="ONNX模型保存路径（默认使用与PyTorch模型相同的文件名，但扩展名为.onnx）")
    parser.add_argument("--dynamic_axes", action="store_true", default=True, help="使用动态轴（支持可变输入大小）")
    parser.add_argument("--check_dims", action="store_true", default=True, help="检查并修复ONNX模型中的卷积算子维度")
    
    return parser.parse_args()

def main():
    """主函数"""
    # 解析命令行参数
    args = parse_args()
    
    # 检查模型文件是否存在
    if not os.path.exists(args.model_path):
        print(f"错误: 模型文件不存在: {args.model_path}")
        return 1
    
    # 如果未指定ONNX保存路径，则根据PyTorch模型路径生成
    if args.onnx_save_path is None:
        args.onnx_save_path = os.path.splitext(args.model_path)[0] + '.onnx'
    
    # 确保输出目录存在
    output_dir = os.path.dirname(args.onnx_save_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 导出模型
    print(f"开始导出模型: {args.model_path}")
    print(f"模型类型: {args.model_type}")
    print(f"ONNX保存路径: {args.onnx_save_path}")
    print(f"使用动态轴: {args.dynamic_axes}")
    
    try:
        onnx_path = export_model_to_onnx(
            model_path=args.model_path,
            model_type=args.model_type,
            onnx_save_path=args.onnx_save_path,
            dynamic_axes=args.dynamic_axes
        )
        
        # 检查并修复ONNX模型中的卷积算子维度
        if args.check_dims:
            check_result = check_and_fix_onnx_conv_dims(onnx_path)
            if not check_result:
                print("警告: ONNX模型中的卷积算子维度检查失败，可能需要手动修复。")
                print("请确保在PyTorch模型中已将所有Conv1d替换为Conv2d，然后重新导出。")
        
        print(f"导出成功! ONNX模型已保存到: {onnx_path}")
        return 0
    except Exception as e:
        print(f"导出失败: {str(e)}")
        return 1

if __name__ == "__main__":
    exit(main()) 