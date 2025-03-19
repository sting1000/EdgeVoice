#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
导出PyTorch模型为ONNX格式的独立脚本
"""

import os
import argparse
from train import export_model_to_onnx

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="将PyTorch模型导出为ONNX格式")
    parser.add_argument("--model_path", type=str, required=True, help="PyTorch模型路径")
    parser.add_argument("--model_type", type=str, choices=["fast", "streaming"], default="fast", help="模型类型")
    parser.add_argument("--onnx_save_path", type=str, default=None, help="ONNX模型保存路径（默认使用与PyTorch模型相同的文件名，但扩展名为.onnx）")
    parser.add_argument("--dynamic_axes", action="store_true", default=True, help="使用动态轴（支持可变输入大小）")
    
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
        print(f"导出成功! ONNX模型已保存到: {onnx_path}")
        return 0
    except Exception as e:
        print(f"导出失败: {str(e)}")
        return 1

if __name__ == "__main__":
    exit(main()) 