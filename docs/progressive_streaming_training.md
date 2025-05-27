# EdgeVoice 渐进式流式训练

## 概述

渐进式流式训练是为EdgeVoice项目专门设计的训练策略，旨在让模型在训练过程中逐步适应流式推理模式，从而提高实际部署时的性能和稳定性。

## 核心特性

### 1. 渐进式训练调度
- **Phase 1 (Epoch 1-10)**: 100% 完整序列训练，建立基础特征表示
- **Phase 2 (Epoch 11-20)**: 70% 完整序列 + 30% 流式训练，逐步适应
- **Phase 3 (Epoch 21-30)**: 30% 完整序列 + 70% 流式训练，重点流式优化

### 2. 最终预测损失优化
- 只对流式处理的最终预测计算损失
- 忽略中间chunk的预测，专注于最终决策准确性
- 可选的稳定性损失，减少预测跳变

### 3. EdgeVoice特定评估
- 核心指令Top1准确率监控
- 预测稳定性评分
- 误识别率分析
- 针对EdgeVoice眼镜应用场景的指标

## 实现架构

### 核心组件

#### 1. ProgressiveStreamingTrainer
```python
class ProgressiveStreamingTrainer:
    def __init__(self, chunk_size=STREAMING_CHUNK_SIZE, step_size=STREAMING_STEP_SIZE):
        # 管理渐进式训练策略
        # 处理chunk分割和状态传递
    
    def get_streaming_ratio(self, epoch):
        # 根据epoch返回流式训练比例
    
    def streaming_forward_pass(self, model, features, device):
        # 执行流式前向传播
        # 分chunk处理，维护流式状态
```

#### 2. FinalPredictionLoss
```python
class FinalPredictionLoss(nn.Module):
    def forward(self, final_output, labels, all_outputs=None):
        # 主要损失：最终预测损失
        # 可选：稳定性损失（减少预测跳变）
```

#### 3. EdgeVoiceMetrics
```python
class EdgeVoiceMetrics:
    def calculate_top1_accuracy(self, predictions, labels, intent_labels):
        # 计算总体和核心指令准确率
    
    def calculate_stability_score(self, prediction_sequences):
        # 计算预测稳定性评分
    
    def calculate_misidentification_rate(self, predictions, labels, intent_labels):
        # 计算误识别率
```

## 使用方法

### 1. 基本使用

```bash
python train_streaming.py \
    --annotation_file data/split/train_annotations.csv \
    --model_save_path saved_models/streaming_conformer_progressive.pt \
    --progressive_streaming \
    --num_epochs 30
```

### 2. 完整配置

```bash
python train_streaming.py \
    --annotation_file data/split/train_annotations.csv \
    --valid_annotation_file data/split/val_annotations.csv \
    --model_save_path saved_models/streaming_conformer_progressive.pt \
    --progressive_streaming \
    --progressive_training \
    --use_mixup \
    --use_label_smoothing \
    --label_smoothing 0.1 \
    --num_epochs 30 \
    --batch_size 32 \
    --learning_rate 2e-4
```

### 3. 使用示例脚本

```bash
python example_progressive_streaming_training.py
```

## 配置参数

### config.py 中的新参数

```python
# 渐进式流式训练参数
PROGRESSIVE_STREAMING_TRAINING = True
STREAMING_TRAINING_SCHEDULE = {
    'phase1': {'epochs': (1, 10), 'streaming_ratio': 0.0},
    'phase2': {'epochs': (11, 20), 'streaming_ratio': 0.3},
    'phase3': {'epochs': (21, 30), 'streaming_ratio': 0.7}
}

# EdgeVoice验证参数
EDGEVOICE_VALIDATION = True
TARGET_ACCURACY_QUIET = 0.95  # 安静环境目标准确率
TARGET_ACCURACY_NOISY = 0.90  # 噪声环境目标准确率
TARGET_STABILITY_SCORE = 0.85  # 目标稳定性评分

# 核心指令定义
CORE_COMMANDS = ['TAKE_PHOTO', 'START_RECORDING', 'STOP_RECORDING']

# 流式训练损失权重
FINAL_PREDICTION_WEIGHT = 1.0
STABILITY_LOSS_WEIGHT = 0.1
```

## 训练输出

### 1. 训练过程监控
```
Epoch 15/30
当前序列长度: 完整序列
当前学习率: 0.000200
流式训练比例: 30.0%

训练中 (Epoch 15): 100%|████████| 50/50 [00:30<00:00, 1.65it/s, loss=0.234, acc=92.3, streaming_ratio=30.0%, stream_batches=15, regular_batches=35]

训练损失: 0.2340, 训练准确率: 92.30%
验证损失: 0.1890, 验证准确率: 94.50%
流式训练批次: 15, 常规训练批次: 35
流式训练损失: 0.2456, 常规训练损失: 0.2298
核心指令准确率: 96.00% (25 样本)
```

### 2. 训练历史可视化
生成的图表包含：
- 训练/验证损失和准确率
- 流式训练比例变化
- 流式vs常规训练批次分布
- 流式训练损失对比
- 实际流式训练百分比

### 3. 模型保存信息
```python
{
    'model_state_dict': ...,
    'streaming_config': {
        'progressive_streaming': True,
        'chunk_size': 200,
        'step_size': 100,
        'schedule': {...}
    }
}
```

## 性能优势

### 1. 训练-推理一致性
- 训练时模拟真实流式推理过程
- 减少训练和部署之间的性能差异
- 提高流式推理的准确性和稳定性

### 2. EdgeVoice特定优化
- 针对核心指令的重点优化
- 预测稳定性提升，减少用户体验问题
- 适应EdgeVoice眼镜的实际使用场景

### 3. 渐进式策略
- 早期建立稳定的特征表示
- 后期专注于流式适应
- 平衡训练稳定性和流式性能

## 测试和验证

### 1. 功能测试
```bash
python test_progressive_streaming.py
```

### 2. 性能对比
建议对比以下训练策略：
- 纯完整序列训练
- 纯流式训练
- 渐进式流式训练

### 3. EdgeVoice评估
使用EdgeVoiceMetrics评估：
- 核心指令准确率
- 预测稳定性
- 误识别率

## 故障排除

### 常见问题

1. **内存不足**
   - 减小batch_size
   - 减小chunk_size
   - 使用梯度累积

2. **训练不稳定**
   - 调整streaming_ratio
   - 增加stability_loss_weight
   - 检查学习率设置

3. **流式性能差**
   - 增加流式训练比例
   - 延长训练时间
   - 检查chunk_size设置

### 调试技巧

1. **监控流式训练统计**
   ```python
   # 查看训练历史中的streaming_stats
   history['streaming_stats']
   ```

2. **可视化训练过程**
   ```python
   # 查看生成的训练历史图表
   streaming_conformer_history_with_streaming.png
   ```

3. **分析预测稳定性**
   ```python
   # 使用EdgeVoiceMetrics分析
   metrics.calculate_stability_score(prediction_sequences)
   ```

## 未来扩展

### 1. 自适应chunk size
- 根据音频内容动态调整chunk大小
- 基于置信度的早停机制

### 2. 多场景训练
- 不同噪声环境的专门训练
- 音量级别适应训练

### 3. 在线学习
- 部署后的持续学习
- 用户个性化适应

## 总结

渐进式流式训练为EdgeVoice项目提供了一个强大的训练策略，通过模拟真实的流式推理过程，显著提高了模型在实际部署中的性能和稳定性。这种方法特别适合EdgeVoice眼镜这样的实时语音交互设备。 