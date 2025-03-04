### 1. 环境准备

首先需要安装必要的Python依赖：

```bash
pip install -r requirements.txt
```

### 2. 准备DistilBERT模型

由于系统使用了DistilBERT模型进行文本处理，需要提前下载模型文件到本地：

```bash
# 创建模型目录
mkdir -p models/distilbert-base-uncased

# 从Hugging Face下载模型文件（需要互联网连接）
# 方法1：使用transformers库下载后复制
python -c "from transformers import DistilBertTokenizer, DistilBertModel; tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased', cache_dir='./tmp'); model = DistilBertModel.from_pretrained('distilbert-base-uncased', cache_dir='./tmp')"

# 复制下载的模型文件到项目目录
cp -r ./tmp/models--distilbert-base-uncased/* ./models/distilbert-base-uncased/

# 或者方法2：手动从Hugging Face网站下载
# 访问 https://huggingface.co/distilbert-base-uncased/tree/main
# 下载所有必要文件（config.json, vocab.txt, pytorch_model.bin等）
# 保存到 models/distilbert-base-uncased/ 目录
```

系统会优先使用本地模型文件，如果本地文件不存在，则会尝试从在线资源获取。

### 3. 准备数据集

创建一个CSV格式的注释文件，包含以下列：
- `file_path`: 音频文件相对路径
- `intent`: 意图标签（来自8个预定义类别）
- `transcript`: (可选，用于精确分类器) 语音内容文本

将音频文件放置在`data`目录下，按照注释文件中的路径组织。

### 4. 训练模型

#### 训练一级快速分类器:

```bash
python train.py --annotation_file data/annotations.csv --model_type fast --epochs 20
```

#### 训练二级精确分类器:

```bash
python train.py --annotation_file data/annotations.csv --model_type precise --epochs 20
```

这将在`saved_models`目录中保存训练好的模型。

### 5. 执行演示

使用训练好的模型运行演示程序:

#### 5.1 实时麦克风模式

使用麦克风进行实时语音识别（需要麦克风设备）:

```bash
python demo.py --fast_model saved_models/fast_intent_model.pth --precise_model saved_models/precise_intent_model.pth
```

#### 5.2 音频文件模式

在没有麦克风或服务器环境下，可以使用音频文件进行演示:

```bash
python demo.py --fast_model saved_models/fast_intent_model.pth --precise_model saved_models/precise_intent_model.pth --use_file --file_path data/samples/take_photo.wav
```

#### 5.3 批处理模式

批量处理目录中的所有音频文件:

```bash
python demo.py --fast_model saved_models/fast_intent_model.pth --precise_model saved_models/precise_intent_model.pth --batch_mode --file_path data/samples/
```

命令行参数说明:
- `--fast_model`: 指定一级快速分类器模型路径（必需）
- `--precise_model`: 指定二级精确分类器模型路径（可选）
- `--use_file`: 使用音频文件代替麦克风输入
- `--file_path`: 指定音频文件或目录的路径
- `--batch_mode`: 启用批处理模式，处理指定目录中的所有音频文件

演示程序会:
1. **实时模式**：启动音频流，持续捕获麦克风输入，等待用户输入"wake"来模拟唤醒词检测
2. **文件模式**：直接处理指定的音频文件并显示识别结果
3. **批处理模式**：处理指定目录中的所有音频文件（支持.wav, .mp3, .flac, .ogg格式）

识别结果包括:
- 意图类型
- 置信度
- 处理路径（快速/精确分类器）
- 处理时间

### 6. 模型评估

系统提供了评估脚本，可以对训练好的模型进行性能评估，并生成详细的指标报告：

```bash
python evaluate.py --annotation_file data/test_annotations.csv --fast_model saved_models/fast_intent_model.pth --precise_model saved_models/precise_intent_model.pth
```

评估脚本会：
1. 分别评估fast模型和precise模型的性能
2. 评估完整推理流程的性能（两个模型协同工作）
3. 生成包含以下内容的Excel报告：
   - 总体性能指标（准确率、精确率、召回率、F1分数、推理时间）
   - 每个意图类别的详细指标
   - 推理时间分布数据
4. 生成可视化图表：
   - 模型准确率比较图
   - 平均推理时间比较图
   - 各模型推理时间分布直方图

#### 参数说明

- `--annotation_file`: 测试数据集的注释文件路径（必需）
- `--fast_model`: 一级快速分类器模型路径（必需）
- `--precise_model`: 二级精确分类器模型路径（可选）
- `--data_dir`: 数据目录，默认为config.py中的DATA_DIR
- `--output_dir`: 评估结果输出目录，默认为"evaluation_results"
- `--analyze_length`: 开启音频长度对模型性能的影响分析

#### 仅评估快速模型

如果只想评估快速模型的性能，可以省略`--precise_model`参数：

```bash
python evaluate.py --annotation_file data/test_annotations.csv --fast_model saved_models/fast_intent_model.pth
```

#### 评估结果

评估结果将保存在`output_dir`指定的目录中，包括：
- Excel报告文件（包含多个工作表）
- 可视化图表（PNG格式）

### 7. 音频长度处理

系统对不同长度的音频文件进行了专门处理，确保在训练和评估阶段都能正确处理各种长度的音频：

#### 7.1 音频长度标准化

系统实现了音频长度标准化，以解决不同长度音频文件在处理过程中可能出现的问题：

1. **过长音频处理**：
   - 使用语音活动检测(VAD)找到最关键的语音段
   - 优先保留语音段的中间部分
   - 避免简单截断导致的信息丢失

2. **过短音频处理**：
   - 对过短音频进行居中填充
   - 确保音频有足够的上下文信息

3. **标准化方法**：
   ```bash
   # 音频长度标准化函数
   standardize_audio_length(audio, sample_rate, target_length=5.0, min_length=0.5)
   ```

#### 7.2 音频长度对性能的影响分析

使用 `--analyze_length` 参数可以分析音频长度对模型性能的影响：

```bash
python evaluate.py --annotation_file data/test_annotations.csv --fast_model saved_models/fast_intent_model.pth --analyze_length
```

分析结果将包括：
- 不同长度组（短、中、长、超长）的性能指标
- 长度对准确率的影响图表
- 长度对推理时间的影响图表
- 详细的Excel分析报告

这些分析有助于理解模型在处理不同长度音频时的行为，并可以指导模型优化和数据收集策略。

#### 7.3 训练阶段处理

在训练阶段，系统会自动:
- 统计数据集中的音频长度分布情况
- 标准化音频长度，确保数据质量
- 处理异常情况，如过长或过短的样本
- 记录警告信息，帮助识别潜在问题

### 8. 数据增强

为了改善小型数据集的训练效果，系统提供了全面的音频数据增强功能：

#### 8.1 支持的增强方法

系统实现了多种音频增强技术，特别适用于语音意图识别任务：

- **音高变化**：改变音频的音调，模拟不同用户的声音特征
- **时间伸缩**：调整音频的语速，保持音高不变
- **音量调整**：改变音频的响度，模拟不同距离和环境条件
- **噪声添加**：增加白噪声，提高模型对噪声的鲁棒性
- **组合增强**：同时应用多种增强方法，创造更多样化的变体

#### 8.2 离线数据增强

离线方式可以预先生成增强数据并保存到磁盘：

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

#### 8.3 使用增强数据集训练

使用`train_with_augmentation.py`脚本可以在训练过程中应用实时数据增强：

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

参数说明：
- `--no_augment`：禁用数据增强（默认启用）
- `--augment_prob`：数据增强概率，即每个样本被增强的概率，默认为0.5
- `--no_cache`：禁用音频缓存（默认启用缓存以加速训练）

其他参数与标准训练脚本相同。

#### 8.4 增强参数调优建议

针对语音意图识别任务，以下是推荐的增强参数设置：

- **音高变化**：`n_steps_range=(-3, 3)`，避免过大变化导致不自然音色
- **时间伸缩**：`rate_range=(0.8, 1.2)`，保持语音内容可理解
- **音量调整**：`gain_range=(0.5, 1.5)`，模拟合理的音量变化
- **噪声添加**：`noise_level_range=(0.001, 0.01)`，低强度噪声更接近真实场景

对于不同类型的命令，可能需要特定调整：
- 简短命令（如"拍照"）：使用较小的时间伸缩范围
- 复杂指令：可以应用更强的音高变化和噪声增强

#### 8.5 增强效果监控

通过以下方式监控数据增强的效果：

1. 比较有无增强的训练曲线（使用train_with_augmentation.py生成）
2. 观察各意图类别的准确率变化
3. 分析在不同环境条件下的泛化性能

详细的数据增强功能说明请参考[README_augmentation.md](README_augmentation.md)文档。