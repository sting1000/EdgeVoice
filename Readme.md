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

#### 仅评估快速模型

如果只想评估快速模型的性能，可以省略`--precise_model`参数：

```bash
python evaluate.py --annotation_file data/test_annotations.csv --fast_model saved_models/fast_intent_model.pth
```

#### 评估结果

评估结果将保存在`output_dir`指定的目录中，包括：
- Excel报告文件（包含多个工作表）
- 可视化图表（PNG格式）