#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
渐进式流式训练使用示例
展示如何使用新实现的渐进式流式训练功能
"""

import os
import argparse
from train_streaming import train_streaming_conformer

def main():
    """主函数 - 演示渐进式流式训练的使用"""
    
    print("=== EdgeVoice 渐进式流式训练示例 ===\n")
    
    # 检查数据文件是否存在
    train_file = "data/split/train_annotations.csv"
    val_file = "data/split/val_annotations.csv"
    
    if not os.path.exists(train_file):
        print(f"❌ 训练数据文件不存在: {train_file}")
        print("请先运行数据收集和分割脚本")
        return
    
    print(f"✅ 找到训练数据: {train_file}")
    
    if os.path.exists(val_file):
        print(f"✅ 找到验证数据: {val_file}")
    else:
        print(f"⚠️  验证数据文件不存在: {val_file}")
        print("将从训练集中分割验证集")
        val_file = None
    
    # 设置保存路径
    save_dir = "saved_models"
    os.makedirs(save_dir, exist_ok=True)
    model_save_path = os.path.join(save_dir, "streaming_conformer_progressive.pt")
    
    print(f"\n📁 模型将保存到: {model_save_path}")
    
    # 训练配置
    config = {
        'data_dir': 'data',
        'annotation_file': train_file,
        'valid_annotation_file': val_file,
        'model_save_path': model_save_path,
        'num_epochs': 15,  # 使用较少的epoch进行演示
        'batch_size': 16,  # 使用较小的batch size
        'learning_rate': 2e-4,
        'progressive_streaming': True,  # 启用渐进式流式训练
        'progressive_training': True,   # 同时启用渐进式长度训练
        'use_mixup': True,
        'use_label_smoothing': True,
        'label_smoothing': 0.1
    }
    
    print("\n🔧 训练配置:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    print(f"\n📊 渐进式流式训练调度:")
    print(f"  Epoch 1-10:  0% 流式训练 (纯完整序列训练)")
    print(f"  Epoch 11-15: 30% 流式训练 (混合训练)")
    print(f"  注: 完整30个epoch的训练中，Epoch 21-30将使用70%流式训练")
    
    print(f"\n🎯 EdgeVoice特定优化:")
    print(f"  - 针对核心指令的重点评估")
    print(f"  - 预测稳定性监控")
    print(f"  - 最终预测损失优化")
    
    # 开始训练
    print(f"\n🚀 开始渐进式流式训练...")
    print(f"=" * 60)
    
    try:
        model, intent_labels = train_streaming_conformer(**config)
        
        print(f"\n" + "=" * 60)
        print(f"🎉 训练完成!")
        print(f"✅ 模型已保存到: {model_save_path}")
        print(f"📋 识别的意图类别: {intent_labels}")
        
        print(f"\n📈 训练结果文件:")
        result_dir = os.path.dirname(model_save_path)
        print(f"  - 训练历史图表: {result_dir}/streaming_conformer_history_with_streaming.png")
        print(f"  - 模型权重: {model_save_path}")
        
        print(f"\n🔍 下一步:")
        print(f"  1. 查看训练历史图表，分析流式训练效果")
        print(f"  2. 使用evaluate_streaming_model评估模型性能")
        print(f"  3. 对比渐进式流式训练与传统训练的效果")
        
    except Exception as e:
        print(f"\n❌ 训练过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
        
        print(f"\n🔧 故障排除建议:")
        print(f"  1. 检查数据文件格式是否正确")
        print(f"  2. 确认GPU内存是否足够")
        print(f"  3. 尝试减小batch_size")

if __name__ == "__main__":
    main() 