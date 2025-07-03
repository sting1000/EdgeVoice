#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
导出StreamingConformer模型为ONNX格式的独立脚本
"""

import os
import argparse
import torch
import onnx
from models.streaming_conformer import StreamingConformer
from config import STREAMING_CHUNK_SIZE

def export_model_to_onnx(model_path, onnx_save_path=None, dynamic_axes=False):
    """
    将StreamingConformer模型导出为ONNX格式
    
    Args:
        model_path: PyTorch模型路径
        onnx_save_path: ONNX模型保存路径（如果为None则根据原模型路径生成）
        dynamic_axes: 是否使用动态轴（用于支持可变输入大小），默认为False以固定batch_size=1
    
    Returns:
        onnx_save_path: 导出的ONNX模型路径
    """
    print(f"正在导出StreamingConformer模型到ONNX格式...")
    
    # 如果未指定ONNX保存路径，则根据PyTorch模型路径生成
    if onnx_save_path is None:
        onnx_save_path = os.path.splitext(model_path)[0] + '.onnx'
    
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 加载模型检查点
    checkpoint = torch.load(model_path, map_location=device)
    
    if isinstance(checkpoint, dict) and 'model_config' in checkpoint and checkpoint['model_config'].get('model_type') == 'streaming_conformer':
        # --- 处理 StreamingConformer 模型 --- 
        print("检测到 StreamingConformer 模型检查点...")
        model_config = checkpoint['model_config']
        model_state = checkpoint['model_state_dict']
        intent_labels = checkpoint['intent_labels']
        num_classes = model_config['num_classes']
        
        print(f"从配置加载模型参数: {model_config}")
        print(f"意图类别: {intent_labels}")
        
        # 使用保存的配置实例化 StreamingConformer
        model = StreamingConformer(
            input_dim=model_config['input_dim'],
            hidden_dim=model_config['hidden_dim'],
            num_classes=num_classes,
            num_layers=model_config['num_layers'],
            num_heads=model_config['num_heads'],
            dropout=model_config.get('dropout', 0.1), # 兼容旧模型
            kernel_size=model_config.get('kernel_size', 31), # 兼容旧模型
            expansion_factor=model_config.get('expansion_factor', 4), # 兼容旧模型
            use_padded_output=model_config.get('use_padded_output', False), # 兼容旧模型
            padded_output_dim=model_config.get('padded_output_dim', 16) # 兼容旧模型
        )
        
        # 创建新状态字典并调整参数
        new_state_dict = {}
        current_state_dict = model.state_dict()
        
        # 预处理卷积权重
        for name, param in model_state.items():
            # 特殊处理1D->2D卷积权重
            if 'pointwise_conv1.weight' in name and len(param.shape) == 3:
                # 获取对应的模型参数
                model_param = current_state_dict.get(name)
                if model_param is not None and len(model_param.shape) == 4:
                    # Conv1d -> Conv2d: [out_ch, in_ch, kernel] -> [out_ch, in_ch, 1, kernel]
                    out_ch, in_ch, kernel = param.shape
                    new_param = param.unsqueeze(2)  # 添加高度维度
                    # 如果kernel=1，我们认为是点卷积，需要调整为[out_ch, in_ch, 1, 1]
                    if kernel == 1:
                        new_param = new_param.squeeze(-1).unsqueeze(-1)
                    new_state_dict[name] = new_param
                    print(f"调整参数 '{name}': {param.shape} -> {new_param.shape}")
                else:
                    new_state_dict[name] = param
            elif 'pointwise_conv2.weight' in name and len(param.shape) == 3:
                # 同样处理第二个点卷积
                model_param = current_state_dict.get(name)
                if model_param is not None and len(model_param.shape) == 4:
                    out_ch, in_ch, kernel = param.shape
                    new_param = param.unsqueeze(2)  # 添加高度维度
                    if kernel == 1:
                        new_param = new_param.squeeze(-1).unsqueeze(-1)
                    new_state_dict[name] = new_param
                    print(f"调整参数 '{name}': {param.shape} -> {new_param.shape}")
                else:
                    new_state_dict[name] = param
            elif 'depthwise_conv.weight' in name:
                # 检查是否需要重命名为conv.weight
                conv_name = name.replace('depthwise_conv', 'conv')
                if conv_name in current_state_dict:
                    # 如果维度匹配，直接使用
                    model_param = current_state_dict.get(conv_name)
                    if model_param is not None and model_param.shape == param.shape:
                        new_state_dict[conv_name] = param
                        print(f"重命名参数 '{name}' -> '{conv_name}'")
                    else:
                        # 可能需要调整维度
                        # 传统depthwise从[out_ch, 1, kernel] -> [out_ch, in_ch//groups, 1, kernel]
                        print(f"跳过参数 '{name}'，无法自动调整为 '{conv_name}'")
                else:
                    new_state_dict[name] = param
            else:
                # 其他参数保持不变
                new_state_dict[name] = param
        
        # 从新状态字典加载可能已调整的参数
        try:
            model.load_state_dict(new_state_dict, strict=False)
            print("模型权重加载完成 (strict=False)")
        except Exception as e:
            print(f"加载模型权重时出错: {str(e)}")
            print("尝试使用原始状态字典...")
            model.load_state_dict(model_state, strict=False)
            
        model.to(device)
        model.eval()
        
        # 创建流式模型的示例输入 (固定 batch_size=1, chunk_length=STREAMING_CHUNK_SIZE)
        dummy_input_chunk = torch.randn(1, STREAMING_CHUNK_SIZE, model_config['input_dim'], device=device)
        
        input_names = ["input_chunk"]
        output_names = ["output_logits"]
        
        # 动态轴定义
        dynamic_axes_dict = None
        if dynamic_axes:
            dynamic_axes_dict = {
                'input_chunk': {0: 'batch_size', 1: 'seq_length'},
                'output_logits': {0: 'batch_size'}
            }
        
        # 确定实际输出维度
        use_padded_output = model_config.get('use_padded_output', False)
        padded_output_dim = model_config.get('padded_output_dim', 16)
        actual_output_dim = padded_output_dim if use_padded_output else num_classes
        
        # 导出核心模型逻辑（forward）
        if dynamic_axes:
            print(f"导出 StreamingConformer (动态形状 [batch_size, seq_length, {model_config['input_dim']}]) 到 {onnx_save_path}")
        else:
            print(f"导出 StreamingConformer (固定形状 [1, {STREAMING_CHUNK_SIZE}, {model_config['input_dim']}] -> [1, {actual_output_dim}]) 到 {onnx_save_path}")
            if use_padded_output:
                print(f"注意: 使用填充输出 {actual_output_dim}维，但只有前{num_classes}个维度有效")
        
        torch.onnx.export(
            model, 
            dummy_input_chunk, 
            onnx_save_path,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes_dict,
            export_params=True,
            opset_version=13,
            do_constant_folding=True,
            verbose=False
        )
    else:
         raise ValueError(f"无法确定模型类型或不受支持的检查点格式: {model_path}")

    print(f"模型已导出至: {onnx_save_path}")
    return onnx_save_path

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
    parser = argparse.ArgumentParser(description="将StreamingConformer模型导出为ONNX格式")
    parser.add_argument("--model_path", type=str, required=True, help="PyTorch模型路径")
    parser.add_argument("--model_type", type=str, default="streaming", help="模型类型(仅支持streaming)")
    parser.add_argument("--onnx_save_path", type=str, default=None, help="ONNX模型保存路径（默认使用与PyTorch模型相同的文件名，但扩展名为.onnx）")
    parser.add_argument("--dynamic_axes", action="store_true", default=False, help="使用动态轴（支持可变输入大小）")
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
    print(f"ONNX保存路径: {args.onnx_save_path}")
    if args.dynamic_axes:
        print(f"使用动态轴: 是 (可变batch_size和seq_length)")
    else:
        print(f"使用动态轴: 否 (固定batch_size=1, seq_length={STREAMING_CHUNK_SIZE})")
    
    try:
        onnx_path = export_model_to_onnx(
            model_path=args.model_path,
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
    main() 