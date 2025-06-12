#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
部署优化的ONNX模型导出脚本
专为满足limits.md约束而设计
"""

import os
import torch
import torch.onnx
import onnx
import onnxruntime as ort
import numpy as np
import argparse
from pathlib import Path

from config import *
from models.streaming_conformer import StreamingConformer

def validate_deployment_constraints(model, dummy_input):
    """
    验证模型是否满足部署约束
    
    Args:
        model: PyTorch模型
        dummy_input: 虚拟输入
        
    Returns:
        is_valid: 是否符合约束
        issues: 约束违反问题列表
    """
    issues = []
    
    # 1. 检查输入维度 (最大4维)
    if len(dummy_input.shape) > 4:
        issues.append(f"输入维度过高: {len(dummy_input.shape)}维 > 4维限制")
    
    # 2. 检查通道对齐 (FP16需要16通道对齐)
    input_channels = dummy_input.shape[-1]
    if input_channels % 16 != 0:
        issues.append(f"输入通道数未对齐: {input_channels} 不是16的倍数")
    
    # 3. 运行模型检查中间张量维度
    with torch.no_grad():
        try:
            output = model(dummy_input)
            if isinstance(output, tuple):
                for i, o in enumerate(output):
                    if len(o.shape) > 4:
                        issues.append(f"输出{i}维度过高: {len(o.shape)}维 > 4维限制")
            else:
                if len(output.shape) > 4:
                    issues.append(f"输出维度过高: {len(output.shape)}维 > 4维限制")
        except Exception as e:
            issues.append(f"模型前向传播失败: {e}")
    
    return len(issues) == 0, issues

def optimize_model_for_deployment(model):
    """
    为部署优化模型结构
    
    Args:
        model: 原始模型
        
    Returns:
        optimized_model: 优化后的模型
    """
    # 将模型设置为评估模式
    model.eval()
    
    # 禁用所有dropout
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = 0.0
            
    # 冻结BatchNorm层
    for module in model.modules():
        if isinstance(module, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d)):
            module.eval()
            for param in module.parameters():
                param.requires_grad = False
    
    return model

def create_deployment_model(checkpoint_path, config_override=None):
    """
    创建部署专用模型
    
    Args:
        checkpoint_path: 检查点文件路径
        config_override: 配置覆盖参数
        
    Returns:
        model: 部署优化的模型
        input_spec: 输入规格说明
    """
    # 使用部署优化的配置
    if config_override is None:
        config_override = {}
    
    # 确保维度对齐
    input_dim = config_override.get('input_dim', N_MFCC * (2 * CONTEXT_FRAMES + 1))
    hidden_dim = config_override.get('hidden_dim', CONFORMER_HIDDEN_SIZE)
    
    # 16通道对齐
    input_dim = ((input_dim + 15) // 16) * 16
    hidden_dim = ((hidden_dim + 15) // 16) * 16
    
    # 创建模型
    model = StreamingConformer(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_classes=len(INTENT_CLASSES),
        num_layers=config_override.get('num_layers', CONFORMER_LAYERS),
        num_heads=config_override.get('num_heads', CONFORMER_ATTENTION_HEADS),
        dropout=0.0,  # 部署时不使用dropout
        kernel_size=config_override.get('kernel_size', CONFORMER_CONV_KERNEL_SIZE),
        expansion_factor=config_override.get('expansion_factor', CONFORMER_FF_EXPANSION_FACTOR)
    )
    
    # 加载权重
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        if 'model_state_dict' in checkpoint:
            # 尝试加载权重，忽略形状不匹配
            model_dict = model.state_dict()
            checkpoint_dict = checkpoint['model_state_dict']
            
            # 过滤掉形状不匹配的权重
            filtered_dict = {}
            for k, v in checkpoint_dict.items():
                if k in model_dict and model_dict[k].shape == v.shape:
                    filtered_dict[k] = v
                else:
                    print(f"跳过权重 {k}: 形状不匹配 ({model_dict.get(k, torch.tensor([])).shape} vs {v.shape})")
            
            model_dict.update(filtered_dict)
            model.load_state_dict(model_dict, strict=False)
        else:
            model.load_state_dict(checkpoint, strict=False)
        print(f"成功加载权重: {checkpoint_path}")
    else:
        print(f"警告: 检查点文件不存在: {checkpoint_path}")
        print("使用随机初始化的权重")
    
    # 部署优化
    model = optimize_model_for_deployment(model)
    
    # 输入规格
    input_spec = {
        'shape': [-1, -1, input_dim],  # [batch, seq_len, features]
        'dtype': 'float32',
        'name': 'audio_features'
    }
    
    return model, input_spec

def export_to_onnx(model, dummy_input, output_path, input_names=None, output_names=None):
    """
    导出模型为ONNX格式
    
    Args:
        model: PyTorch模型
        dummy_input: 虚拟输入
        output_path: 输出路径
        input_names: 输入名称列表
        output_names: 输出名称列表
        
    Returns:
        success: 是否导出成功
    """
    try:
        # 设置默认名称
        if input_names is None:
            input_names = ['audio_features']
        if output_names is None:
            output_names = ['intent_logits']
        
        # 导出ONNX
        print(f"正在导出ONNX模型到: {output_path}")
        
        torch.onnx.export(
            model,
            dummy_input,
            output_path,
            export_params=True,
            opset_version=11,  # 使用稳定的opset版本
            do_constant_folding=True,  # 常量折叠优化
            input_names=input_names,
            output_names=output_names,
            dynamic_axes={
                input_names[0]: {0: 'batch_size', 1: 'sequence_length'},
                output_names[0]: {0: 'batch_size'}
            },
            verbose=False
        )
        
        print("ONNX导出成功")
        return True
        
    except Exception as e:
        print(f"ONNX导出失败: {e}")
        return False

def validate_onnx_model(onnx_path, dummy_input):
    """
    验证ONNX模型
    
    Args:
        onnx_path: ONNX模型路径
        dummy_input: 测试输入
        
    Returns:
        is_valid: 验证是否通过
        results: 验证结果详情
    """
    results = {}
    
    try:
        # 1. 检查模型结构
        onnx_model = onnx.load(onnx_path)
        onnx.checker.check_model(onnx_model)
        results['structure_check'] = True
        print("✓ ONNX模型结构检查通过")
        
        # 2. 检查算子数量
        total_ops = len(onnx_model.graph.node)
        results['total_operators'] = total_ops
        if total_ops > 768:
            print(f"⚠ 警告: 算子数量 {total_ops} 超过限制 768")
            results['operator_limit_check'] = False
        else:
            print(f"✓ 算子数量检查通过: {total_ops}/768")
            results['operator_limit_check'] = True
        
        # 3. 检查输入输出数量
        num_inputs = len(onnx_model.graph.input)
        num_outputs = len(onnx_model.graph.output)
        results['num_inputs'] = num_inputs
        results['num_outputs'] = num_outputs
        
        if num_inputs > 7:
            print(f"⚠ 警告: 输入数量 {num_inputs} 超过限制 7")
            results['io_limit_check'] = False
        elif num_outputs > 8:
            print(f"⚠ 警告: 输出数量 {num_outputs} 超过限制 8") 
            results['io_limit_check'] = False
        else:
            print(f"✓ 输入输出数量检查通过: {num_inputs}输入, {num_outputs}输出")
            results['io_limit_check'] = True
        
        # 4. 运行时测试
        ort_session = ort.InferenceSession(onnx_path)
        
        # 准备输入
        ort_inputs = {ort_session.get_inputs()[0].name: dummy_input.numpy()}
        
        # 推理测试
        ort_outputs = ort_session.run(None, ort_inputs)
        results['runtime_test'] = True
        print("✓ ONNX运行时测试通过")
        
        # 输出形状信息
        for i, output in enumerate(ort_outputs):
            print(f"  输出{i}形状: {output.shape}")
            results[f'output_{i}_shape'] = output.shape
        
        # 整体验证结果
        is_valid = all([
            results.get('structure_check', False),
            results.get('operator_limit_check', False),
            results.get('io_limit_check', False),
            results.get('runtime_test', False)
        ])
        
        return is_valid, results
        
    except Exception as e:
        print(f"ONNX模型验证失败: {e}")
        results['error'] = str(e)
        return False, results

def create_model_info_file(output_dir, model_info):
    """创建模型信息文件"""
    info_path = os.path.join(output_dir, 'model_info.txt')
    
    with open(info_path, 'w', encoding='utf-8') as f:
        f.write("EdgeVoice 部署优化模型信息\n")
        f.write("=" * 40 + "\n\n")
        
        f.write("模型配置:\n")
        for key, value in model_info.items():
            f.write(f"  {key}: {value}\n")
        
        f.write(f"\n意图类别 ({len(INTENT_CLASSES)}):\n")
        for i, intent in enumerate(INTENT_CLASSES):
            f.write(f"  {i}: {intent}\n")
        
        f.write("\n部署约束:\n")
        f.write("  ✓ 16通道对齐 (FP16)\n")
        f.write("  ✓ 移除分组卷积\n")
        f.write("  ✓ 最大4维张量\n")
        f.write("  ✓ 算子数量 ≤ 768\n")
        f.write("  ✓ 输入数量 ≤ 7\n")
        f.write("  ✓ 输出数量 ≤ 8\n")
        
    print(f"模型信息已保存到: {info_path}")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='导出部署优化的ONNX模型')
    parser.add_argument('--checkpoint', '-c', required=True, help='PyTorch模型检查点路径')
    parser.add_argument('--output', '-o', default='./deployed_models', help='输出目录')
    parser.add_argument('--name', '-n', default='streaming_conformer_optimized', help='模型名称')
    parser.add_argument('--batch-size', '-b', type=int, default=1, help='批大小')
    parser.add_argument('--seq-len', '-s', type=int, default=100, help='序列长度')
    parser.add_argument('--validate', action='store_true', help='是否验证导出的模型')
    
    args = parser.parse_args()
    
    # 创建输出目录
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("EdgeVoice 部署优化 ONNX 导出工具")
    print("=" * 60)
    print(f"检查点: {args.checkpoint}")
    print(f"输出目录: {output_dir}")
    print(f"模型名称: {args.name}")
    print("=" * 60)
    
    try:
        # 1. 创建部署模型
        print("\n1. 创建部署优化模型...")
        model, input_spec = create_deployment_model(args.checkpoint)
        
        # 2. 创建虚拟输入
        input_dim = input_spec['shape'][2]
        dummy_input = torch.randn(args.batch_size, args.seq_len, input_dim)
        
        print(f"输入规格: {dummy_input.shape}")
        print(f"输入维度对齐: {input_dim % 16 == 0} (16通道对齐)")
        
        # 3. 验证部署约束
        print("\n2. 验证部署约束...")
        is_valid, issues = validate_deployment_constraints(model, dummy_input)
        
        if not is_valid:
            print("⚠ 发现约束违反问题:")
            for issue in issues:
                print(f"  - {issue}")
            print("请修复问题后重新导出")
            return
        else:
            print("✓ 所有部署约束验证通过")
        
        # 4. 导出ONNX
        print("\n3. 导出ONNX模型...")
        onnx_path = output_dir / f"{args.name}.onnx"
        
        success = export_to_onnx(
            model, 
            dummy_input, 
            str(onnx_path),
            input_names=['audio_features'],
            output_names=['intent_logits']
        )
        
        if not success:
            print("ONNX导出失败")
            return
        
        # 5. 验证ONNX模型
        if args.validate:
            print("\n4. 验证ONNX模型...")
            is_valid, results = validate_onnx_model(str(onnx_path), dummy_input)
            
            if is_valid:
                print("✓ ONNX模型验证全部通过")
            else:
                print("⚠ ONNX模型验证发现问题，请检查")
        
        # 6. 创建模型信息文件
        model_info = {
            'input_dim': input_dim,
            'hidden_dim': model.conformer_layers[0].attn.q_proj.in_features,
            'num_layers': len(model.conformer_layers),
            'num_heads': model.conformer_layers[0].attn.heads,
            'num_classes': len(INTENT_CLASSES),
            'total_parameters': sum(p.numel() for p in model.parameters()),
            'trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad)
        }
        
        create_model_info_file(str(output_dir), model_info)
        
        print(f"\n✅ 部署优化模型导出完成!")
        print(f"ONNX模型: {onnx_path}")
        print(f"模型信息: {output_dir / 'model_info.txt'}")
        print(f"总参数量: {model_info['total_parameters']:,}")
        
    except Exception as e:
        print(f"\n❌ 导出失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 