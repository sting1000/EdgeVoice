# 音频数据增强功能说明

本文档说明如何使用EdgeVoice项目的音频数据增强功能来扩充小型训练数据集。

## 1. 数据增强方法

EdgeVoice实现了多种音频数据增强技术，特别适用于语音意图识别任务：

### 1.1 基础增强方法

- **音高变化 (Pitch Shifting)**：改变音频的音调，模拟不同用户的声音特征
- **时间伸缩 (Time Stretching)**：调整音频的语速，保持音高不变
- **音量调整 (Volume Adjustment)**：改变音频的响度，模拟不同距离和环境条件
- **噪声添加 (Noise Addition)**：增加白噪声，提高模型对噪声的鲁棒性
- **组合增强 (Combined Augmentation)**：同时应用多种增强方法，创造更多样化的变体

### 1.2 增强实现方式

我们提供了两种使用数据增强的方式：

1. **离线增强**：预先生成增强数据并保存到磁盘，适合小型数据集和复杂增强
2. **实时增强**：在训练过程中即时生成增强样本，更灵活且节省存储空间

## 2. 离线数据增强

离线方式通过预先生成增强变体来扩充数据集：

### 2.1 使用方法

```bash
# 基本用法
python audio_augmentation.py --annotation_file data/annotations.csv --data_dir data --output_dir data_augmented

# 高级选项
python audio_augmentation.py --annotation_file data/annotations.csv \
                            --data_dir data \
                            --output_dir data_augmented \
                            --augment_factor 5 \
                            --visualize 10
```

### 2.2 参数说明

- `--annotation_file`：原始数据集的注释文件路径（必需）
- `--data_dir`：原始音频文件目录，默认为"data"
- `--output_dir`：增强数据的输出目录，默认为"data_augmented"
- `--augment_factor`：每个原始样本生成的增强版本数量，默认为5
- `--visualize`：生成可视化的样本数量，默认为5

### 2.3 输出结果

离线增强将生成：
- 增强后的音频文件，保持原始目录结构
- 新的注释文件（`augmented_annotations.csv`），包含原始和增强样本
- 可视化图表，展示原始与增强波形对比（位于"visualizations"子目录）

## 3. 实时数据增强

实时增强在训练过程中动态生成增强样本，无需额外存储空间：

### 3.1 使用增强训练

```bash
# 基本用法（默认启用增强）
python train_with_augmentation.py --annotation_file data/annotations.csv --model_type fast

# 高级选项
python train_with_augmentation.py --annotation_file data/annotations.csv \
                                 --model_type fast \
                                 --epochs 30 \
                                 --augment_prob 0.7 \
                                 --batch_size 16
```

### 3.2 参数说明

- `--annotation_file`：数据集的注释文件路径（必需）
- `--data_dir`：音频文件目录，默认为配置中的DATA_DIR
- `--model_type`：模型类型，'fast'或'precise'，默认为'fast'
- `--epochs`：训练轮数，默认为20
- `--batch_size`：批处理大小，默认为配置中的BATCH_SIZE
- `--learning_rate`：学习率，默认为配置中的LEARNING_RATE
- `--save_dir`：模型保存目录，默认为"saved_models"
- `--seed`：随机种子，确保可重复性，默认为42
- `--no_augment`：禁用数据增强（默认启用）
- `--augment_prob`：数据增强概率，即每个样本被增强的概率，默认为0.5
- `--no_cache`：禁用音频缓存（默认启用缓存以加速训练）

## 4. 自定义数据增强

你也可以在自己的代码中直接使用增强器：

```python
from audio_augmentation import AudioAugmenter

# 初始化增强器
augmenter = AudioAugmenter(sample_rate=16000)

# 加载音频
audio, sr = librosa.load("sample.wav", sr=16000)

# 应用特定增强
pitched_audio, pitch_rate = augmenter.pitch_shift(audio)
stretched_audio, stretch_rate = augmenter.time_stretch(audio)
louder_audio, gain = augmenter.adjust_volume(audio)

# 随机增强
augmented_audios, descriptions = augmenter.random_augment(audio)
```

## 5. 增强参数调优建议

针对语音意图识别任务，以下是推荐的增强参数设置：

- **音高变化**：`n_steps_range=(-3, 3)`，避免过大变化导致不自然音色
- **时间伸缩**：`rate_range=(0.8, 1.2)`，保持语音内容可理解
- **音量调整**：`gain_range=(0.5, 1.5)`，模拟合理的音量变化
- **噪声添加**：`noise_level_range=(0.001, 0.01)`，低强度噪声更接近真实场景

对于不同类型的命令，可能需要特定调整：
- 简短命令（如"拍照"）：使用较小的时间伸缩范围
- 复杂指令：可以应用更强的音高变化和噪声增强

## 6. 数据增强效果监控

通过以下方式监控数据增强的效果：

1. 比较有无增强的训练曲线（使用train_with_augmentation.py生成）
2. 观察各意图类别的准确率变化
3. 分析在不同环境条件下的泛化性能

如果发现某类增强导致性能下降，可以调整对应参数或减少其使用频率。

## 7. 结合其他增强技术

数据增强可以与其他技术结合使用以获得更好效果：

- **迁移学习**：使用预训练模型，结合增强数据进行微调
- **交叉验证**：结合增强在不同折上评估模型性能
- **混合精度训练**：启用FP16加速增强数据的训练过程 