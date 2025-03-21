# EdgeVoice: 智能眼镜语音意图识别系统

EdgeVoice是一个为智能眼镜设计的轻量级语音意图识别系统，通过两级分类器架构，在保证高准确率的同时实现低延迟响应。系统特别针对相机相关功能（如拍照、录像等）进行了优化，提供自然流畅的语音交互体验。

## 1. 环境准备

首先需要安装必要的Python依赖：

```bash
pip install -r requirements.txt
```

## 2. 准备DistilBERT模型

系统使用了DistilBERT模型进行文本处理，需要提前下载模型文件到本地：

```bash
# 创建模型目录
mkdir -p models/distilbert-base-uncased

# 从Hugging Face下载模型文件（需要互联网连接）
python -c "from transformers import DistilBertTokenizer, DistilBertModel; tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased', cache_dir='./tmp'); model = DistilBertModel.from_pretrained('distilbert-base-uncased', cache_dir='./tmp')"

# 复制下载的模型文件到项目目录
cp -r ./tmp/models--distilbert-base-uncased/* ./models/distilbert-base-uncased/
```

系统会优先使用本地模型文件，如果本地文件不存在，则会尝试从在线资源获取。

## 3. 准备数据集

创建一个CSV格式的注释文件，包含以下列：
- `file_path`: 音频文件相对路径
- `intent`: 意图标签（来自8个预定义类别）
- `transcript`: (可选，用于精确分类器) 语音内容文本

将音频文件放置在`data`目录下，按照注释文件中的路径组织。

## 4. 训练模型

系统支持两种类型的模型训练，并集成了实时音频数据增强功能以提高模型性能。

### 4.1 基本训练命令

#### 训练一级快速分类器:

```bash
python train.py --annotation_file data/annotations.csv --model_type fast --epochs 20
```

#### 训练二级精确分类器:

```bash
python train.py --annotation_file data/annotations.csv --model_type precise --epochs 20
```

### 4.2 启用数据增强训练

数据增强可以显著提高模型泛化能力，特别适用于小规模数据集。EdgeVoice系统内置了实时数据增强功能，可在训练过程中即时生成增强样本。

```bash
# 启用数据增强训练快速分类器
python train.py --annotation_file data/annotations.csv --model_type fast --augment --augment_prob 0.6

# 启用数据增强训练精确分类器 
python train.py --annotation_file data/annotations.csv --model_type precise --augment --augment_prob 0.6
```

### 4.3 导出ONNX模型

EdgeVoice支持将训练好的PyTorch模型导出为ONNX格式，便于在更多平台上部署和优化推理速度。

#### 在训练过程中导出ONNX模型

您可以在训练模型的同时导出ONNX格式：

```bash
# 训练并导出快速分类器
python train.py --annotation_file data/annotations.csv --model_type fast --export_onnx --model_save_path saved_models/fast_intent_model.pth

# 训练并导出流式模型
python train.py --annotation_file data/annotations.csv --model_type streaming --export_onnx --model_save_path saved_models/streaming_model.pth

# 指定ONNX模型保存路径
python train.py --annotation_file data/annotations.csv --model_type fast --export_onnx --model_save_path saved_models/fast_intent_model.pth --onnx_save_path saved_models/fast_model_optimized.onnx
```

#### 转换已训练好的模型为ONNX格式

对于已经训练好的模型，可以直接使用train.py的导出功能：

```bash
# 转换快速分类器
python train.py --annotation_file data/annotations.csv --model_type fast --export_onnx --model_save_path saved_models/fast_intent_model.pth --num_epochs 0

# 转换流式模型
python train.py --annotation_file data/annotations.csv --model_type streaming --export_onnx --model_save_path saved_models/streaming_model.pth --num_epochs 0

# 指定ONNX模型保存路径
python train.py --annotation_file data/annotations.csv --model_type fast --export_onnx --model_save_path saved_models/fast_intent_model.pth --onnx_save_path saved_models/fast_optimized.onnx --num_epochs 0

# 使用静态输入形状（如果目标平台不支持动态形状）
python train.py --annotation_file data/annotations.csv --model_type fast --export_onnx --model_save_path saved_models/fast_intent_model.pth --dynamic_axes False --num_epochs 0
```

> 注意：使用`--num_epochs 0`参数可以跳过训练过程，直接进行模型导出。

#### ONNX模型的优势

- **跨平台兼容性**：可在多种推理引擎和硬件上运行
- **运行时优化**：许多平台可对ONNX模型进行进一步优化
- **更低的部署复杂度**：不依赖于PyTorch运行环境

### 4.4 训练参数说明

基本参数:
- `--annotation_file`: 训练数据集的注释CSV文件路径（必需）
- `--data_dir`: 音频文件目录，默认为`DATA_DIR`
- `--model_type`: 模型类型，'fast'或'precise'（必需）
- `--epochs`: 训练轮数，默认为配置中的`NUM_EPOCHS`

数据增强参数:
- `--augment`: 启用数据增强（默认启用）
- `--augment_prob`: 应用增强的概率，默认为0.5
- `--use_cache`: 启用音频缓存以加速训练（默认启用）
- `--seed`: 随机种子，用于确保实验可重复性，默认为42

ONNX导出参数:
- `--export_onnx`: 训练后导出模型为ONNX格式
- `--onnx_save_path`: ONNX模型保存路径（默认使用与PyTorch模型相同的文件名，但扩展名为.onnx）

### 4.5 支持的数据增强方法

系统内置以下音频增强技术，可实时增强训练样本：

- **音高变化**：改变音频的音调，模拟不同用户的声音特征（范围：-3到3个半音）
- **时间伸缩**：调整音频的语速，保持音高不变（范围：0.8到1.2倍）
- **音量调整**：改变音频的响度，模拟不同距离和环境条件（范围：0.5到1.5倍）
- **噪声添加**：增加白噪声，提高模型对噪声的鲁棒性（强度：0.001到0.01）
- **组合增强**：同时应用多种增强方法，创造更多样化的变体

增强在训练过程中实时进行，不需要预先生成增强数据，节省磁盘空间并提高训练灵活性。每个训练周期结束后，系统会自动生成训练曲线图，帮助可视化训练进度和增强效果。

## 5. 模型评估

系统提供了评估脚本，可以对训练好的模型进行性能评估，并生成详细的指标报告：

```bash
python evaluate.py --annotation_file data/test_annotations.csv --fast_model saved_models/fast_intent_model.pth --precise_model saved_models/precise_intent_model.pth
```

评估脚本会：
1. 分别评估fast模型和precise模型的性能
2. 评估完整推理流程的性能（两个模型协同工作）
3. 生成包含以下内容的评估报告：
   - 总体性能指标（准确率、精确率、召回率、F1分数、推理时间）
   - 每个意图类别的详细指标
   - 可视化图表（准确率比较、推理时间分布等）

### 5.1 评估参数说明

- `--annotation_file`: 测试数据集的注释文件路径（必需）
- `--fast_model`: 一级快速分类器模型路径（必需）
- `--precise_model`: 二级精确分类器模型路径（可选）
- `--data_dir`: 数据目录，默认为config.py中的DATA_DIR
- `--output_dir`: 评估结果输出目录，默认为"evaluation_results"
- `--analyze_length`: 开启音频长度对模型性能的影响分析

### 5.2 仅评估快速模型

如果只想评估快速模型的性能，可以省略`--precise_model`参数：

```bash
python evaluate.py --annotation_file data/test_annotations.csv --fast_model saved_models/fast_intent_model.pth
```

## 6. 执行演示

使用训练好的模型运行演示程序:

### 6.1 实时麦克风模式

使用麦克风进行实时语音识别（需要麦克风设备）:

```bash
python demo.py --fast_model saved_models/fast_intent_model.pth --precise_model saved_models/precise_intent_model.pth
```

### 6.2 音频文件模式

在没有麦克风或服务器环境下，可以使用音频文件进行演示:

```bash
python demo.py --fast_model saved_models/fast_intent_model.pth --precise_model saved_models/precise_intent_model.pth --use_file --file_path data/samples/take_photo.wav
```

### 6.3 批处理模式

批量处理目录中的所有音频文件:

```bash
python demo.py --fast_model saved_models/fast_intent_model.pth --precise_model saved_models/precise_intent_model.pth --batch_mode --file_path data/samples/
```

### 6.4 命令行参数说明

- `--fast_model`: 指定一级快速分类器模型路径（必需）
- `--precise_model`: 指定二级精确分类器模型路径（可选）
- `--use_file`: 使用音频文件代替麦克风输入
- `--file_path`: 指定音频文件或目录的路径
- `--batch_mode`: 启用批处理模式，处理指定目录中的所有音频文件

## 7. 音频长度处理

系统对不同长度的音频文件进行了专门处理，确保在训练和评估阶段都能正确处理各种长度的音频：

### 7.1 音频长度标准化

系统实现了音频长度标准化，以解决不同长度音频文件在处理过程中可能出现的问题：

- **过长音频处理**：对超过目标长度的音频进行中心裁剪
- **过短音频处理**：对过短音频进行居中填充，确保音频有足够的上下文信息

```python
# 音频长度标准化函数
standardize_audio_length(audio, sample_rate, target_length=5.0, min_length=0.5)
```

### 7.2 音频长度对性能的影响分析

使用 `--analyze_length` 参数可以分析音频长度对模型性能的影响：

```bash
python evaluate.py --annotation_file data/test_annotations.csv --fast_model saved_models/fast_intent_model.pth --analyze_length
```

分析结果将包括：
- 不同长度组（短、中、长、超长）的性能指标
- 长度对准确率的影响图表
- 长度对推理时间的影响图表

## 8. 流式处理功能

EdgeVoice系统实现了流式处理技术，支持音频数据的增量处理，特别适合智能眼镜等实时交互场景。

### 8.1 流式处理原理

系统将连续音频流分割成重叠的小块(chunks)进行处理，保留前一块的状态信息，实现低延迟实时识别：

- **块大小**：默认每块10帧(100ms)，可通过`STREAMING_CHUNK_SIZE`参数调整
- **步长**：默认5帧(50ms)，可通过`STREAMING_STEP_SIZE`参数调整
- **状态缓存**：保留模型的隐藏状态，确保连续识别的一致性
- **早停机制**：当置信度达到预设阈值(默认0.9)时提前结束识别

### 8.2 流式处理演示

使用流式处理演示脚本测试模型的流式识别能力：

```bash
# 使用音频文件测试流式处理
python streaming_demo.py --model_path saved_models/fast_intent_model.pth --audio_file data/samples/take_photo.wav

# 自定义块大小和步长
python streaming_demo.py --model_path saved_models/fast_intent_model.pth --audio_file data/samples/take_photo.wav --chunk_size 15 --step_size 8
```

### 8.3 流式模型评估

专门的流式评估脚本可分析模型在流式场景下的性能：

```bash
python streaming_evaluation.py --model_path saved_models/fast_intent_model.pth --annotation_file data/test_annotations.csv

# 自定义置信度阈值
python streaming_evaluation.py --model_path saved_models/fast_intent_model.pth --annotation_file data/test_annotations.csv --confidence_threshold 0.85

# 禁用多数投票机制
python streaming_evaluation.py --model_path saved_models/fast_intent_model.pth --annotation_file data/test_annotations.csv --no_majority_voting
```

评估结果将包含：
- 流式识别准确率
- 早停比例（提前结束识别的样本比例）
- 平均决策延迟（从音频开始到得出决策的时间）
- 预测稳定性分析（预测结果变化次数统计）
- 各类别的性能指标
- 混淆矩阵和性能可视化图表

### 8.4 流式训练

系统支持专门的流式训练模式，使模型更适应增量数据处理场景：

```bash
# 以流式模式训练快速分类器
python train.py --annotation_file data/annotations.csv --model_type fast --streaming_mode

# 测试流式训练效果
python test_streaming_training.py
```

流式训练的特点：
- 使用分块特征训练模型
- 在训练过程中模拟流式处理场景
- 提高模型对部分音频片段的识别能力
- 减少对完整音频的依赖，适合早期识别

## 9. 模型优化技术

### 9.1 特征优化

系统使用了优化的MFCC特征提取：
- **维度优化**：使用16维MFCC特征（而非传统13维）
- **16倍数对齐**：总特征维度优化为48维（16×3=48），适应硬件加速要求
- **增强特征**：计算Delta和Delta-Delta捕捉时间动态信息
- **上下文融合**：使用±2帧上下文信息增强当前帧特征

### 9.2 Conformer优化

一级快速分类器采用轻量级Conformer架构：
- **平衡参数**：4层Conformer块，8个注意力头
- **256维隐藏层**：提供充分的表示能力且易于16倍数对齐
- **31大小卷积核**：有效捕捉局部语音特征
- **流式架构**：支持帧级增量处理，保持状态连续性

### 9.3 量化与剪枝

- **INT8量化**：将模型权重从FP32降至INT8，减少内存占用
- **模型剪枝**：移除贡献小的参数，减少计算量
- **硬件加速**：针对边缘设备进行特定优化，提高推理速度

## 10. 工具脚本

在 `tools` 目录中提供了一些实用工具脚本：

### 10.1 提示语生成工具

`prompt_generator.py` 脚本可以生成更多样化的语音意图提示语，增强模型训练数据的多样性：

```bash
python tools/prompt_generator.py --variants 5 --json_output expanded_prompts.json --py_output new_intent_prompts.py
```

该工具可以为每个意图类别生成更多自然的表达变体，使用模板和终止词组合来创建符合智能眼镜场景的语音指令。

### 10.2 数据收集工具

`data_collection_tool.py` 提供了用户友好的界面，用于收集和标注训练数据：

```bash
python tools/data_collection_tool.py
```

该工具支持：
- 按意图类别收集语音样本
- 指导用户录制指定意图的语音
- 自动生成符合格式的标注数据