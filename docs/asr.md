# EdgeVoice中的WeNet ASR功能

本文档介绍EdgeVoice系统中的WeNet ASR（自动语音识别）功能，包括安装步骤、使用方法、模型训练和参数配置。

## 功能介绍

EdgeVoice现在集成了WeNet ASR模型，用于提供高质量的中文语音识别能力。它将ASR（自动语音识别）和NLU（自然语言理解）结合，通过"ASR+NLU"路径提供更精确的语音意图识别功能。

主要特点：
- 使用Conformer架构，支持端到端语音识别
- 支持中文和英文识别
- 支持多种格式模型（ONNX、TorchScript）
- 可配置的置信度阈值
- 结果缓存和保存功能
- 性能优化，支持低延迟应用场景

## 安装步骤

### 1. 安装依赖

要使用WeNet ASR功能，您需要安装额外的依赖项：

```bash
# 基础依赖
pip install torchaudio librosa webrtcvad 

# WeNet 推理依赖（二选一）
# 选项1: 使用wenet-runtime（推荐用于部署）
pip install wenetruntime

# 选项2: 安装完整的WeNet（用于训练和开发）
git clone https://github.com/wenet-e2e/wenet.git
cd wenet
pip install -e .
```

### 2. 准备模型

您需要一个WeNet ASR模型和对应的词典文件。模型可以是以下格式之一：

- ONNX模型（.onnx）
- TorchScript模型（.pt）
- WeNet原生模型（.pt）

如果您没有预训练模型，可以：

1. 使用我们提供的脚本获取预训练模型：

```bash
python tools/download_model.py --model chinese_conformer
```

2. 使用自己的数据训练模型（见下文的模型训练部分）

## 使用方法

### 1. 命令行演示

使用更新后的demo.py脚本可以直接测试ASR功能：

```bash
# 使用麦克风实时测试，启用ASR
python demo.py --mode mic --asr_model /path/to/model.onnx --asr_dict /path/to/dict.txt

# 处理单个文件
python demo.py --mode file --input sample.wav --asr_model /path/to/model.onnx --asr_dict /path/to/dict.txt

# 处理目录中的所有音频文件
python demo.py --mode dir --input audio_dir --recursive --save --output results_dir
```

主要参数说明：
- `--asr_model`: ASR模型路径
- `--asr_dict`: 字典文件路径
- `--asr_threshold`: ASR置信度阈值，默认0.6
- `--disable_asr`: 禁用ASR处理路径
- `--save_asr_results`: 保存ASR转写结果

### 2. 在代码中使用

您可以在自己的Python应用中集成ASR功能：

```python
from inference import IntentInferenceEngine

# 初始化引擎
engine = IntentInferenceEngine(
    fast_model_path="saved_models/fast_model.pt",
    precise_model_path="saved_models/precise_model.pt",
    asr_model_path="saved_models/asr/model.onnx",
    asr_dict_path="saved_models/asr/lang_char.txt",
    save_asr_results=True
)

# 处理音频文件
result = engine.process_audio_file("sample.wav")

# 从麦克风获取音频并处理
# ...音频获取代码...
result = engine.process_audio_stream(audio_data)

# 处理结果
intent = result["intent"]
confidence = result["confidence"]
path = result["path"]
if "transcription" in result:
    transcription = result["transcription"]
    asr_confidence = result["asr_confidence"]
```

## 模型训练

我们提供了训练WeNet ASR模型的脚本：

```bash
python models/asr_train.py \
    --data_dir data/audio \
    --annotation_file data/transcriptions.csv \
    --model_save_path saved_models/asr \
    --model_type conformer \
    --num_layers 12 \
    --num_heads 4
```

训练参数：
- `--data_dir`: 音频数据目录
- `--annotation_file`: 标注文件（CSV格式，包含file_path和text列）
- `--model_save_path`: 模型保存路径
- `--model_type`: 模型类型，支持"conformer"和"transformer"
- `--num_layers`: 模型层数
- `--num_heads`: 注意力头数
- `--batch_size`: 批次大小
- `--epochs`: 训练轮数

训练完成后，模型和词典文件将被保存在`model_save_path`指定的目录中。

## 配置参数

您可以在`config.py`中调整ASR相关的配置参数：

### 音频处理配置
```python
ASR_MAX_LENGTH = 50000     # 最大音频长度（帧）
ASR_MIN_LENGTH = 10        # 最小音频长度（帧）
ASR_TOKEN_MAX_LENGTH = 200 # 最大标记长度
ASR_TOKEN_MIN_LENGTH = 1   # 最小标记长度
```

### VAD配置
```python
ASR_VAD_ENABLED = True
ASR_VAD_MODE = 3           # 0-3范围
ASR_VAD_PADDING_MS = 300
ASR_SILENCE_THRESHOLD = 0.1
```

### 特征提取配置
```python
ASR_NUM_MEL_BINS = 80
ASR_FRAME_LENGTH = 25
ASR_FRAME_SHIFT = 10
ASR_USE_SPECAUG = True
ASR_SPEC_DROPOUT = 0.1
ASR_TIME_MASK = 2
ASR_FREQ_MASK = 2
```

### 模型参数
```python
ASR_MODEL_TYPE = "conformer"  # conformer, transformer
ASR_NUM_LAYERS = 12
ASR_NUM_HEADS = 4
ASR_NUM_DECODER_LAYERS = 6
ASR_HIDDEN_SIZE = 256
ASR_DROPOUT = 0.1
```

### 推理参数
```python
ASR_BEAM_SIZE = 10
ASR_CONFIDENCE_THRESHOLD = 0.6
ASR_DECODING_METHOD = "attention_rescoring"  # ctc_greedy, ctc_prefix_beam, attention, attention_rescoring
```

## ASR结果分析

当启用`save_asr_results=True`时，系统会将ASR转写结果保存在`ASR_CACHE_DIR`指定的目录中。
每个转写结果以JSON格式保存，包含以下信息：

```json
{
  "audio_id": "sample_123456",
  "text": "转写的文本内容",
  "confidence": 0.87,
  "timestamp": "2023-06-01T14:30:45.123",
  "metadata": {
    "time_ms": 320.5,
    "sample_rate": 16000,
    "audio_length": 3.2
  }
}
```

您可以使用这些数据进行ASR质量分析和错误排查。

## 故障排除

### 常见问题

1. **ASR模型加载失败**
   - 确保模型格式正确（ONNX或TorchScript）
   - 检查字典文件路径是否正确
   - 验证已安装wenetruntime或wenet库

2. **ASR路径未被使用**
   - 检查ASR置信度阈值是否过高
   - 确认ASR模型是否成功加载
   - 验证精确分类器模型是否可用

3. **ASR结果质量差**
   - 调整特征提取参数
   - 使用更大/更适合的模型
   - 考虑在特定领域数据上微调模型

## 高级功能

### 模型导出

您可以将WeNet模型导出为ONNX格式，以便在更多平台上部署：

```python
from models.wenet_asr import WeNetASR

# 初始化ASR模型
asr = WeNetASR(model_path="saved_models/asr/model.pt", dict_path="saved_models/asr/lang_char.txt")

# 导出为ONNX
onnx_path = asr.export_onnx("saved_models/asr/model.onnx")
```

### 自定义ASR+NLU流程

您可以自定义ASR和NLU的集成方式，例如添加文本预处理步骤：

```python
# 获取原始ASR转写结果
text, asr_confidence, _ = engine.asr_inference(audio)

# 自定义文本处理
processed_text = my_text_processor(text)

# 使用处理后的文本进行NLU
intent, confidence, _ = engine.precise_inference(processed_text)
```

## 版本历史

- **v1.0**: 初始版本，集成WeNet ASR基本功能
- **v1.1**: 添加模型导出、中间结果保存和更多配置选项 