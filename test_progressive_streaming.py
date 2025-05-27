#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
渐进式流式训练测试脚本
验证新实现的渐进式流式训练功能
"""

import torch
import numpy as np
from utils.progressive_streaming_trainer import (
    ProgressiveStreamingTrainer, 
    FinalPredictionLoss, 
    EdgeVoiceMetrics
)
from models.streaming_conformer import StreamingConformer
from config import *

def test_progressive_streaming_trainer():
    """测试渐进式流式训练器"""
    print("=== 测试渐进式流式训练器 ===")
    
    # 初始化训练器
    trainer = ProgressiveStreamingTrainer()
    
    # 测试流式比例调度
    print("\n1. 测试流式比例调度:")
    for epoch in [1, 5, 10, 15, 20, 25, 30]:
        ratio = trainer.get_streaming_ratio(epoch)
        should_use = trainer.should_use_streaming(epoch)
        print(f"  Epoch {epoch:2d}: 流式比例={ratio:.1f}, 应该使用流式={should_use}")
    
    # 测试序列分割
    print("\n2. 测试序列分割:")
    batch_size, seq_len, feature_dim = 2, 300, 96
    features = torch.randn(batch_size, seq_len, feature_dim)
    
    chunks = trainer.split_sequence_to_chunks(features)
    print(f"  输入序列: {features.shape}")
    print(f"  分割后chunks数量: {len(chunks)}")
    for i, chunk in enumerate(chunks):
        print(f"    Chunk {i}: {chunk.shape}")
    
    return trainer

def test_final_prediction_loss():
    """测试最终预测损失函数"""
    print("\n=== 测试最终预测损失函数 ===")
    
    # 创建损失函数
    criterion = FinalPredictionLoss()
    
    # 模拟数据
    batch_size, num_classes = 4, 8
    final_output = torch.randn(batch_size, num_classes)
    labels = torch.randint(0, num_classes, (batch_size,))
    
    # 模拟多个chunk的输出
    all_outputs = [
        torch.randn(batch_size, num_classes),
        torch.randn(batch_size, num_classes),
        final_output
    ]
    
    # 计算损失
    loss_without_stability = criterion(final_output, labels)
    loss_with_stability = criterion(final_output, labels, all_outputs)
    
    print(f"  不含稳定性损失: {loss_without_stability:.4f}")
    print(f"  包含稳定性损失: {loss_with_stability:.4f}")
    print(f"  稳定性损失权重: {STABILITY_LOSS_WEIGHT}")
    
    return criterion

def test_edgevoice_metrics():
    """测试EdgeVoice评估指标"""
    print("\n=== 测试EdgeVoice评估指标 ===")
    
    # 创建评估器
    metrics = EdgeVoiceMetrics()
    
    # 模拟预测结果
    intent_labels = ['TAKE_PHOTO', 'START_RECORDING', 'STOP_RECORDING', 
                    'CAPTURE_AND_DESCRIBE', 'OTHERS']
    
    predictions = [0, 1, 2, 0, 4, 1, 2, 3, 0, 1]  # 模拟预测
    labels = [0, 1, 1, 0, 4, 1, 2, 3, 0, 2]       # 模拟真实标签
    
    # 计算准确率
    accuracy_metrics = metrics.calculate_top1_accuracy(predictions, labels, intent_labels)
    print(f"  总体准确率: {accuracy_metrics['total_accuracy']:.2%}")
    print(f"  核心指令准确率: {accuracy_metrics['core_accuracy']:.2%}")
    print(f"  核心指令样本数: {accuracy_metrics['core_samples']}")
    
    # 模拟预测序列（用于稳定性评估）
    prediction_sequences = [
        [0, 0, 0, 0],      # 稳定预测
        [1, 2, 1, 1],      # 有变化的预测
        [3, 3, 4, 4],      # 中途变化
        [2, 2, 2, 2]       # 完全稳定
    ]
    
    stability_metrics = metrics.calculate_stability_score(prediction_sequences)
    print(f"  稳定性评分: {stability_metrics['stability_score']:.2%}")
    print(f"  平均变化次数: {stability_metrics['avg_changes']:.1f}")
    
    # 计算误识别率
    misid_metrics = metrics.calculate_misidentification_rate(predictions, labels, intent_labels)
    print(f"  总体误识别率: {misid_metrics['total_misidentification_rate']:.2%}")
    print(f"  核心指令误识别率: {misid_metrics['core_misidentification_rate']:.2%}")
    
    return metrics

def test_streaming_forward_pass():
    """测试流式前向传播"""
    print("\n=== 测试流式前向传播 ===")
    
    # 创建模型
    model = StreamingConformer(
        input_dim=N_MFCC * 3,
        hidden_dim=64,  # 使用较小的隐藏层以加快测试
        num_classes=8,
        num_layers=2,   # 使用较少层数以加快测试
        num_heads=4,
        dropout=0.1,
        kernel_size=9,
        expansion_factor=2
    )
    model.eval()
    
    # 创建训练器
    trainer = ProgressiveStreamingTrainer()
    
    # 模拟输入
    batch_size, seq_len, feature_dim = 2, 250, N_MFCC * 3
    features = torch.randn(batch_size, seq_len, feature_dim)
    device = torch.device('cpu')
    
    print(f"  输入特征形状: {features.shape}")
    
    try:
        # 执行流式前向传播
        final_output, all_outputs = trainer.streaming_forward_pass(model, features, device)
        
        print(f"  最终输出形状: {final_output.shape}")
        print(f"  总chunk数量: {len(all_outputs)}")
        print(f"  各chunk输出形状: {[output.shape for output in all_outputs]}")
        
        # 验证输出合理性
        assert final_output.shape == (batch_size, 8), f"最终输出形状错误: {final_output.shape}"
        assert len(all_outputs) > 0, "应该至少有一个chunk输出"
        
        print("  ✅ 流式前向传播测试通过")
        
    except Exception as e:
        print(f"  ❌ 流式前向传播测试失败: {e}")
        return False
    
    return True

def main():
    """主测试函数"""
    print("开始测试渐进式流式训练功能...\n")
    
    # 设置随机种子
    torch.manual_seed(42)
    np.random.seed(42)
    
    try:
        # 测试各个组件
        trainer = test_progressive_streaming_trainer()
        criterion = test_final_prediction_loss()
        metrics = test_edgevoice_metrics()
        
        # 测试流式前向传播
        streaming_success = test_streaming_forward_pass()
        
        if streaming_success:
            print("\n🎉 所有测试通过！渐进式流式训练功能正常工作。")
            print("\n📋 功能总结:")
            print("  ✅ 渐进式训练调度器")
            print("  ✅ 序列分割和chunk处理")
            print("  ✅ 最终预测损失函数")
            print("  ✅ EdgeVoice评估指标")
            print("  ✅ 流式前向传播")
            
            print("\n🚀 可以开始使用渐进式流式训练了！")
            print("使用方法:")
            print("  python train_streaming.py --annotation_file data/split/train_annotations.csv \\")
            print("                           --model_save_path saved_models/streaming_conformer_progressive.pt \\")
            print("                           --progressive_streaming \\")
            print("                           --num_epochs 30")
        else:
            print("\n❌ 部分测试失败，请检查实现。")
            
    except Exception as e:
        print(f"\n❌ 测试过程中出现错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 