# 流式Conformer语音命令识别模型

本项目实现了优化的流式Conformer模型，用于实时语音命令识别。

## 主要优化点

### 1. 模型架构优化
- **高效的Conformer模型**：使用多层Conformer结构，提高模型复杂度和表达能力
- **合理的隐藏层维度**：根据设备能力可调整的隐藏层大小
- **注意力池化层**：替代简单平均池化，更好地关注关键特征
- **卷积核大小优化**：降低卷积核大小，更适合短语音命令特征提取
- **高效的前馈网络**：使用优化的激活函数，提高非线性表达能力

### 2. 特征提取优化
- **MFCC及Delta特征**：提取MFCC及其Delta特征，捕捉动态信息
- **流式特征处理**：优化的流式特征提取和处理逻辑
- **特征缓存机制**：支持特征缓存加速训练

### 3. 训练策略改进
- **两阶段训练**：完整音频预训练 + 流式微调的两阶段策略
- **动态特征抖动**：随训练进程减小的特征抖动强度
- **学习率调度**：支持学习率预热和衰减策略
- **梯度裁剪**：防止梯度爆炸，稳定训练过程

## 运行方法

### 环境准备
确保安装所需的依赖：
```bash
pip install -r requirements.txt
```

### 训练流式Conformer模型
使用以下命令训练模型：
```bash
./run_streaming_conformer.sh
```

也可以使用精简版训练脚本直接训练：
```bash
python train.py \
  --data_dir data \
  --annotation_file data/split/train_annotations.csv \
  --model_save_path saved_models/streaming_conformer.pt \
  --num_epochs 20 \
  --pre_train_epochs 10 \
  --fine_tune_epochs 10 \
  --augment \
  --use_cache
```

### 导出ONNX模型
训练完成后，可以将模型导出为ONNX格式部署到边缘设备：
```bash
python export_onnx.py \
  --model_path saved_models/streaming_conformer.pt \
  --onnx_save_path saved_models/streaming_conformer.onnx \
  --dynamic_axes \
  --check_dims
```

## 文件结构
- `train.py`: 精简版的StreamingConformer模型训练脚本
- `models/streaming_conformer.py`: 流式Conformer模型实现
- `streaming_dataset.py`: 流式数据集实现
- `export_onnx.py`: 模型导出ONNX工具
- `run_streaming_conformer.sh`: 训练运行脚本
- `config.py`: 模型配置参数

## 代码说明

1. **训练流程**
   - 预训练阶段：使用完整音频进行传统训练
   - 微调阶段：使用流式特征进行微调，模拟实际推理场景

2. **特征处理**
   - 支持动态特征增强
   - 特征抖动随训练进程衰减
   - 支持特征缓存加速训练

3. **模型保存**
   - 保存模型配置信息，方便ONNX导出
   - 同时保存预训练和微调的历史记录

## 调参建议

1. **模型大小调整**：
   - 小型设备：`CONFORMER_LAYERS=4, CONFORMER_HIDDEN_SIZE=128`
   - 中型设备：`CONFORMER_LAYERS=6, CONFORMER_HIDDEN_SIZE=192`
   - 大型设备：`CONFORMER_LAYERS=8, CONFORMER_HIDDEN_SIZE=256`

2. **训练参数调整**：
   - 增大`pre_train_epochs`可提高模型基础性能
   - 增大`fine_tune_epochs`可提高流式推理性能
   - 使用`--augment`选项启用数据增强，提高泛化能力

3. **特征提取参数**：
   - 调整`N_MFCC`可以改变特征维度
   - 修改特征抖动参数可控制训练稳定性

4. **导出参数**：
   - 启用`--dynamic_axes`使ONNX模型支持动态输入大小
   - 使用`--check_dims`检查卷积操作维度，确保部署兼容性

## 注意事项
- 确保数据目录结构正确
- 训练前可使用`--clear_cache`清理特征缓存
- 支持设置随机种子`--seed`保证可重复性
