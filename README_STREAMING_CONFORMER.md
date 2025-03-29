# 流式Conformer语音命令识别模型

本项目实现了优化的流式Conformer模型，用于实时语音命令识别。针对之前流式训练loss不下降和准确率不提升的问题，我们进行了全面优化。

## 主要优化点

### 1. 模型架构优化
- **更深的Conformer模型**：从4层增加到6层，提高模型复杂度和表达能力
- **扩大隐藏层维度**：从128增加到192，增强特征表示能力
- **注意力池化层**：替代简单平均池化，更好地关注关键特征
- **减小卷积核大小**：从31减少到15，更适合短语音命令特征提取
- **改进的前馈网络**：使用SwiGLU激活函数，提高非线性表达能力

### 2. 特征提取优化
- **增加MFCC系数**：从16增加到20，提供更丰富的频域信息
- **增大FFT窗口**：设置为512，获取更精细的频域分辨率
- **增加上下文帧**：从2增加到4，增强时间依赖性捕捉能力
- **优化流式参数**：调整块大小(15)和历史缓存(60)，更适合短命令

### 3. 训练策略改进
- **标签平滑**：引入标签平滑(0.1)，减轻过拟合风险
- **AdamW优化器**：使用AdamW替代Adam，更好的权重正则化
- **渐进式长度训练**：从短序列开始，逐步增加序列长度
- **余弦学习率调度**：使用余弦退火学习率，提高收敛质量
- **早停机制**：当验证集性能不再提升时停止训练

### 4. 数据增强策略
- **特征级增强**：频谱掩码、时间掩码、高斯噪声
- **时域扭曲**：随机延长或缩短某些帧，增加时间变形不变性
- **MixUp技术**：线性混合不同样本，提高泛化能力

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

也可以自定义参数：
```bash
python train_streaming.py \
  --data_dir data \
  --annotation_file data/split/train_annotations.csv \
  --model_save_path saved_models/streaming_conformer.pt \
  --num_epochs 30 \
  --batch_size 32 \
  --learning_rate 0.0002 \
  --weight_decay 0.01 \
  --use_mixup \
  --progressive_training \
  --evaluate \
  --test_annotation_file data/split/test_annotations.csv
```

### 评估流式性能
```bash
python train_streaming.py \
  --evaluate \
  --model_save_path saved_models/streaming_conformer.pt \
  --test_annotation_file data/split/test_annotations.csv \
  --confidence_threshold 0.85
```

### 实时测试
使用音频文件测试：
```bash
python real_time_streaming_demo.py \
  --model_path saved_models/streaming_conformer.pt \
  --chunk_size 15 \
  --audio_file data/test_samples/test_audio.wav
```

使用麦克风实时测试：
```bash
python real_time_streaming_demo.py \
  --model_path saved_models/streaming_conformer.pt \
  --use_mic \
  --chunk_size 15
```

## 文件结构
- `models/streaming_conformer.py`: 流式Conformer模型实现
- `train_streaming.py`: 流式模型训练、评估脚本
- `utils/feature_augmentation.py`: 特征增强工具
- `run_streaming_conformer.sh`: 训练运行脚本

## 调参建议

1. **流式参数调整**：
   - 对于短命令，可以尝试`STREAMING_CHUNK_SIZE=10-15`
   - 对于长命令，可以尝试`STREAMING_CHUNK_SIZE=20-30`
   - 设置`STREAMING_STEP_SIZE=STREAMING_CHUNK_SIZE/3`更新较为平滑

2. **模型大小调整**：
   - 小型设备：`CONFORMER_LAYERS=4, CONFORMER_HIDDEN_SIZE=128`
   - 中型设备：`CONFORMER_LAYERS=6, CONFORMER_HIDDEN_SIZE=192`(当前配置)
   - 大型设备：`CONFORMER_LAYERS=8, CONFORMER_HIDDEN_SIZE=256`

3. **特征提取参数**：
   - 增大`N_MFCC`可以提高识别率，但会增加计算量
   - 增大`CONTEXT_FRAMES`可以提高模型时间依赖性捕捉，但需要更多内存

4. **训练策略**：
   - 渐进式训练对短命令效果显著，可尝试不同的长度序列
   - 对于小数据集，增加数据增强和标签平滑可有效提高泛化能力 