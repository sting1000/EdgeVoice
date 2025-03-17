# EdgeVoice系统Wenet ASR整合需求

## 系统架构
- **一级处理**：Fast模型直接从音频特征预测意图
- **二级处理**：当Fast模型置信度不足时，启用Precise模型路径(ASR+NLU)

## 技术细节
1. **Wenet模型配置**：
   - 使用Conformer架构模型
   - 仅需支持中文识别
   - 需要支持导出为ONNX格式的功能

2. **集成方式**：
   - 使用Python代码方式集成，而非独立服务
   - 模型加载机制与现有DistilBERT类似

3. **性能要求**：
   - 整个系统的端到端延迟(语音结束到意图输出)控制在1秒以内
   - ASR推理性能需要据此合理优化

4. **训练支持**：
   - 支持使用数据集继续训练Wenet模型，实现持续优化

5. **结果持久化**：
   - 保存ASR中间结果(文本转写)用于后续分析
   - 添加开关控制持久化功能，以平衡性能和分析需求

## 系统集成目标
构建一个无缝整合的系统，可以根据Fast模型的置信度判断，智能切换到ASR+NLU路径，确保高准确率的同时控制整体延迟在可接受范围内。

# Wenet ASR整合到EdgeVoice框架的实施指导

以下是为Agent提供的详细实施指导，用于将Wenet ASR模型集成到EdgeVoice系统中：

## 一、架构设计与文件组织

### 1. 新增文件
```
models/
  └── wenet_asr.py        # Wenet ASR模型封装类
  └── asr_train.py        # ASR模型训练脚本
utils/
  └── asr_utils.py        # ASR相关工具函数
config.py                 # 添加ASR配置参数
inference.py              # 修改推理引擎，支持ASR+NLU路径
```

### 2. 配置参数添加
在`config.py`中添加ASR相关配置：

```python
# ASR相关配置
ASR_MODEL_PATH = os.path.join(MODEL_DIR, "wenet_conformer")
ASR_CACHE_DIR = os.path.join(MODEL_DIR, "asr_cache")
ASR_ONNX_PATH = os.path.join(MODEL_DIR, "wenet_conformer.onnx")
ASR_RESULT_SAVE = True
ASR_DICT_PATH = os.path.join(ASR_MODEL_PATH, "units.txt")
ASR_CONF_THRESHOLD = 0.6
PRECISE_TOTAL_TIMEOUT_MS = 800  # 总超时时间(ms)
```

## 二、核心类实现

### 1. WeNetASR类设计 (`models/wenet_asr.py`)

```python
# 创建WeNetASR类，实现以下功能：
# 1. 加载预训练Wenet模型
# 2. 提供音频到文本的转写功能
# 3. 支持导出ONNX模型
# 4. 提供中间结果保存机制

class WeNetASR:
    def __init__(self, model_path, dict_path, device=None, save_results=False, result_dir=None):
        """初始化Wenet ASR模型
        
        Args:
            model_path: Wenet模型路径
            dict_path: 字典文件路径
            device: 运行设备(cuda/cpu)
            save_results: 是否保存ASR中间结果
            result_dir: 结果保存目录
        """
        # 初始化参数
        self.device = device if device else DEVICE
        self.save_results = save_results
        self.result_dir = result_dir if result_dir else ASR_CACHE_DIR
        if self.save_results:
            os.makedirs(self.result_dir, exist_ok=True)
        
        # 加载词典
        self.load_dictionary(dict_path)
        
        # 加载模型
        self.load_model(model_path)
        
        # 初始化特征提取器
        self.feature_extractor = self._init_feature_extractor()
    
    def load_dictionary(self, dict_path):
        """加载字典文件"""
        # 实现词典加载逻辑
        pass
    
    def load_model(self, model_path):
        """加载Wenet模型"""
        # 实现模型加载逻辑
        pass
    
    def _init_feature_extractor(self):
        """初始化特征提取器"""
        # 实现特征提取器初始化
        pass
    
    def extract_features(self, audio, sample_rate):
        """从音频中提取特征
        
        Args:
            audio: 音频数据
            sample_rate: 采样率
            
        Returns:
            特征张量
        """
        # 特征提取逻辑
        pass
    
    def transcribe(self, audio, sample_rate=16000):
        """将音频转写为文本
        
        Args:
            audio: 音频数据
            sample_rate: 采样率
            
        Returns:
            文本结果, 置信度
        """
        # 转写逻辑实现
        pass
    
    def save_result(self, audio_id, text, confidence):
        """保存ASR结果
        
        Args:
            audio_id: 音频ID或时间戳
            text: 转写文本
            confidence: 置信度
        """
        if not self.save_results:
            return
        
        # 实现结果保存逻辑
        pass
    
    def export_onnx(self, onnx_path):
        """导出ONNX模型
        
        Args:
            onnx_path: ONNX模型保存路径
        """
        # 实现ONNX导出逻辑
        pass
```

### 2. 修改IntentInferenceEngine类 (`inference.py`)

```python
# 修改IntentInferenceEngine类，添加ASR+NLU处理路径：
# 1. 添加WeNetASR模型加载
# 2. 修改predict_intent方法，支持ASR+NLU路径
# 3. 添加ASR结果持久化支持

from models.wenet_asr import WeNetASR

class IntentInferenceEngine:
    def __init__(self, fast_model_path, precise_model_path=None, 
                 asr_model_path=None, asr_dict_path=None,
                 fast_confidence_threshold=FAST_CONFIDENCE_THRESHOLD,
                 asr_result_save=ASR_RESULT_SAVE):
        """初始化推理引擎
        
        Args:
            fast_model_path: 快速分类器模型路径
            precise_model_path: 精确分类器模型路径（可选）
            asr_model_path: ASR模型路径（可选）
            asr_dict_path: ASR字典路径（可选）
            fast_confidence_threshold: 快速分类器置信度阈值
            asr_result_save: 是否保存ASR结果
        """
        # 现有初始化代码...
        
        # 加载ASR模型（如果提供）
        self.asr_model = None
        if asr_model_path:
            print("Loading ASR model...")
            asr_dict_path = asr_dict_path if asr_dict_path else ASR_DICT_PATH
            self.asr_model = WeNetASR(
                model_path=asr_model_path,
                dict_path=asr_dict_path,
                device=self.device,
                save_results=asr_result_save,
                result_dir=ASR_CACHE_DIR
            )
    
    def predict_intent(self, audio, audio_text=None):
        """预测音频意图
        
        Args:
            audio: 音频数据
            audio_text: 预先转写的文本（可选）
            
        Returns:
            (intent_class, confidence, preprocessing_time, inference_time, path)
        """
        # 预处理音频并提取特征
        features, preprocess_time = self.preprocess_audio(audio)
        
        # 首先尝试快速分类器
        intent_class, confidence, inference_time = self.fast_inference(features)
        
        # 如果置信度高，直接返回结果
        if confidence >= self.fast_confidence_threshold:
            return intent_class, confidence, preprocess_time, inference_time, "fast"
        
        # 如果没有精确分类器或ASR模型，返回快速分类器结果
        if self.precise_model is None and self.asr_model is None:
            return intent_class, confidence, preprocess_time, inference_time, "fast"
        
        # 检查是否已提供文本
        if audio_text is None and self.asr_model is not None:
            # 使用ASR模型转写音频
            asr_start_time = time.time()
            audio_text, asr_confidence = self.asr_model.transcribe(audio)
            asr_time = time.time() - asr_start_time
            
            # 生成唯一ID并保存ASR结果
            audio_id = int(time.time() * 1000)
            self.asr_model.save_result(audio_id, audio_text, asr_confidence)
            
            print(f"ASR转写结果: {audio_text}")
            print(f"ASR置信度: {asr_confidence:.4f}")
            print(f"ASR转写时间: {asr_time*1000:.2f}ms")
        
        # 如果有文本且有精确分类器，使用精确分类器
        if audio_text and self.precise_model is not None:
            precise_intent, precise_confidence, precise_time = self.precise_inference(audio_text)
            return precise_intent, precise_confidence, preprocess_time, precise_time, "precise"
        
        # 否则返回快速分类器结果
        return intent_class, confidence, preprocess_time, inference_time, "fast"
```

### 3. ASR训练支持 (`models/asr_train.py`)

```python
# 创建ASR训练脚本，实现以下功能：
# 1. 准备训练数据
# 2. 加载和微调Wenet模型
# 3. 保存训练后的模型

import os
import argparse
import torch
import numpy as np
import json
from tqdm import tqdm
from config import *

def prepare_data(data_dir, annotation_file, output_dir):
    """准备ASR训练数据
    
    Args:
        data_dir: 音频数据目录
        annotation_file: 标注文件
        output_dir: 输出目录
    """
    # 实现数据准备逻辑
    pass

def train_asr_model(data_dir, model_path, output_path, num_epochs=10):
    """训练ASR模型
    
    Args:
        data_dir: 训练数据目录
        model_path: 预训练模型路径
        output_path: 输出模型路径
        num_epochs: 训练轮数
    """
    # 实现训练逻辑
    pass

def export_onnx_model(model_path, onnx_path):
    """导出ONNX模型
    
    Args:
        model_path: PyTorch模型路径
        onnx_path: ONNX输出路径
    """
    # 实现ONNX导出逻辑
    pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='训练ASR模型')
    parser.add_argument('--data_dir', type=str, default=DATA_DIR, help='数据目录')
    parser.add_argument('--annotation_file', type=str, required=True, help='标注文件')
    parser.add_argument('--model_path', type=str, default=ASR_MODEL_PATH, help='预训练模型路径')
    parser.add_argument('--output_path', type=str, default=ASR_MODEL_PATH, help='输出模型路径')
    parser.add_argument('--prepare_only', action='store_true', help='只准备数据不训练')
    parser.add_argument('--export_onnx', action='store_true', help='导出ONNX模型')
    parser.add_argument('--onnx_path', type=str, default=ASR_ONNX_PATH, help='ONNX模型路径')
    parser.add_argument('--epochs', type=int, default=10, help='训练轮数')
    
    args = parser.parse_args()
    
    # 准备数据
    data_prepared_dir = os.path.join(args.data_dir, 'asr_prepared')
    prepare_data(args.data_dir, args.annotation_file, data_prepared_dir)
    
    if not args.prepare_only:
        # 训练模型
        train_asr_model(data_prepared_dir, args.model_path, args.output_path, args.epochs)
    
    if args.export_onnx:
        # 导出ONNX模型
        export_onnx_model(args.output_path, args.onnx_path)
```

## 三、使用示例和接口

### 1. ASR模型加载和推理示例

```python
# 加载ASR模型
asr_model = WeNetASR(
    model_path=ASR_MODEL_PATH,
    dict_path=ASR_DICT_PATH,
    save_results=True
)

# 转写音频
audio, sr = librosa.load("test.wav", sr=16000)
text, confidence = asr_model.transcribe(audio)
print(f"转写结果: {text}")
print(f"置信度: {confidence:.4f}")

# 导出ONNX模型
asr_model.export_onnx(ASR_ONNX_PATH)
```

### 2. ASR+NLU推理示例

```python
# 创建推理引擎
engine = IntentInferenceEngine(
    fast_model_path="saved_models/fast_intent_model.pth",
    precise_model_path="saved_models/precise_intent_model.pth",
    asr_model_path=ASR_MODEL_PATH,
    asr_dict_path=ASR_DICT_PATH,
    asr_result_save=True
)

# 处理音频
audio, sr = librosa.load("test.wav", sr=16000)
intent, confidence, preprocess_time, inference_time, path = engine.predict_intent(audio)

print(f"检测到意图: {intent}")
print(f"置信度: {confidence:.4f}")
print(f"使用路径: {path}")
print(f"处理时间: {(preprocess_time + inference_time)*1000:.2f}ms")
```

### 3. ASR模型训练示例

```bash
# 准备数据并训练模型
python models/asr_train.py --annotation_file data/annotations.csv --epochs 5

# 仅准备数据
python models/asr_train.py --annotation_file data/annotations.csv --prepare_only

# 导出ONNX模型
python models/asr_train.py --export_onnx --model_path saved_models/wenet_conformer --onnx_path saved_models/wenet_conformer.onnx
```

## 四、性能优化建议

1. **推理性能优化**：
   - 使用批处理处理音频
   - 实现特征缓存，避免重复计算
   - 使用线程池并行处理ASR和NLU任务

2. **延迟控制策略**：
   - 设置ASR推理超时（如500ms）
   - 流式输出ASR结果，获得部分结果即可进行NLU
   - 允许中断长时间运行的ASR过程

3. **模型优化技术**：
   - 对Wenet模型进行剪枝和量化
   - 使用ONNX Runtime加速推理
   - 考虑使用TensorRT进一步优化（如适用）

4. **内存优化**：
   - 及时释放不需要的模型和中间结果
   - 使用内存池管理临时变量
   - 设置合理的批处理大小

## 五、注意事项与实施建议

1. **Wenet安装**：
   - 使用pip: `pip install wenetruntime`或从源码编译
   - 确保下载正确版本的预训练模型

2. **依赖管理**：
   - 将依赖添加到requirements.txt
   - 处理可能的版本冲突

3. **错误处理**：
   - ASR模型加载失败时的回退策略
   - 提供清晰的错误信息和日志

4. **测试与验证**：
   - 创建单元测试验证每个组件
   - 进行端到端测试测量整体延迟
   - 验证在各种音频条件下的性能

以上实施指导应该可以帮助Agent将Wenet ASR模型成功集成到EdgeVoice框架中，实现一个完整的两级意图识别系统。
