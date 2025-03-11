# EdgeVoice 部署指南

本文档介绍如何在华为设备上使用HiAI推理框架部署EdgeVoice模型。

## 简介

EdgeVoice部署模块提供了以下功能：

1. 音频处理：包括重采样、预加重、VAD、静音消除和降噪等功能
2. MFCC特征提取：支持生成带上下文的MFCC特征
3. 意图识别推理接口：支持使用OMC模型进行意图识别
4. 应用示例：提供使用上述功能的完整示例

## 目录结构

```
deployment/
├── CMakeLists.txt        # CMake构建文件
├── include/              # 头文件目录
│   ├── audio_processor.h # 音频处理头文件
│   └── inference_engine.h # 推理引擎头文件
├── src/                  # 源文件目录
│   ├── audio_processor.cpp # 音频处理实现
│   ├── inference_engine.cpp # 推理引擎实现
│   └── main.cpp         # 主程序示例
├── utils/               # 辅助工具脚本
│   ├── model_converter.py # 模型转换工具
│   └── performance_tester.py # 性能测试工具
└── README.md            # 部署说明文档
```

## 编译要求

- CMake 3.10 或更高版本
- 支持C++17的编译器 (如GCC 7+)
- 华为HiAI SDK (根据目标设备选择适当版本)

## 编译步骤

1. 创建构建目录

```bash
mkdir -p build
cd build
```

2. 配置构建

```bash
cmake -DCMAKE_BUILD_TYPE=Release ..
```

3. 编译代码

```bash
make -j4
```

4. (可选) 安装可执行文件

```bash
make install
```

编译成功后，您将在`build`目录（或安装后的路径）中找到`edgevoice`可执行文件。

## 使用方法

### 基本用法

```bash
./edgevoice <模型路径> <音频文件路径>
```

例如：

```bash
./edgevoice model.omc test.wav
```

### 批处理多个音频文件

```bash
./edgevoice <模型路径> <音频文件1> <音频文件2> ...
```

例如：

```bash
./edgevoice model.omc test1.wav test2.wav test3.wav
```

### 输出说明

程序会输出每个音频文件的识别结果，包括：

- 识别的意图类别
- 置信度百分比
- 预处理时间（毫秒）
- 推理时间（毫秒）
- 总处理时间（毫秒）

## 集成到其他应用

如果要将EdgeVoice集成到其他应用中，可以参考以下步骤：

1. 包含必要的头文件：

```cpp
#include "inference_engine.h"
```

2. 创建并初始化推理引擎：

```cpp
edgevoice::InferenceEngine engine("path/to/model.omc", 0.7f);
if (!engine.init()) {
    // 处理初始化失败
    return;
}
```

3. 从WAV文件或音频数据进行预测：

```cpp
// 从WAV文件预测
auto result = engine.predictFromWavFile("audio.wav");

// 或从音频数据预测
std::vector<float> audio_data = /*...获取音频数据...*/;
int sample_rate = 16000;
auto result = engine.predictIntent(audio_data, sample_rate);

// 处理结果
std::cout << "识别的意图: " << result.intent_class << std::endl;
std::cout << "置信度: " << result.confidence << std::endl;
```

## 注意事项

1. 本实现兼容华为HiAI框架，可以在华为单框架设备上运行
2. 音频文件应为WAV格式，支持16位、24位和32位PCM格式
3. 默认支持的意图类别包括：
   - CAPTURE_AND_DESCRIBE
   - CAPTURE_REMEMBER
   - CAPTURE_SCAN_QR
   - TAKE_PHOTO
   - START_RECORDING
   - STOP_RECORDING
   - GET_BATTERY_LEVEL
   - OTHERS

## 工具说明

### 模型转换工具

位于`utils/model_converter.py`，用于将PyTorch模型转换为OMC格式。

```bash
python utils/model_converter.py --onnx model.onnx --output model.omc
```

### 性能测试工具

位于`utils/performance_tester.py`，用于测试模型在设备上的性能。

```bash
python utils/performance_tester.py --model model.omc --data test_data/ --iterations 100
```

---

如有问题或需要进一步帮助，请联系技术支持团队。 