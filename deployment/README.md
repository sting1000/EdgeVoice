# EdgeVoice HiAI部署

本项目是EdgeVoice语音意图识别系统在华为HiAI框架上的部署实现。该实现使用C++语言，可以在支持HiAI框架的华为设备上运行。

## 项目结构

```
deployment/
├── include/                  # 头文件目录
│   ├── audio_processor.h     # 音频处理和特征提取
│   └── inference_engine.h    # 推理引擎接口
├── src/                      # 源文件目录
│   ├── audio_processor.cpp   # 音频处理和特征提取实现
│   ├── inference_engine.cpp  # 推理引擎实现
│   └── main.cpp              # 主程序
├── tools/                    # 工具脚本
│   ├── model_converter.py    # 模型转换工具
│   └── performance_tester.py # 性能测试工具
├── CMakeLists.txt            # CMake构建文件
└── README.md                 # 本文档
```

## 编译要求

- CMake 3.8+
- C++11兼容的编译器
- 华为HiAI开发环境
- 华为设备开发工具链

## 编译步骤

1. 设置HiAI开发环境变量：

```bash
export HIAI_INCLUDE_DIR=/path/to/hiai/include
export HIAI_LIB_DIR=/path/to/hiai/lib
```

2. 创建构建目录并编译：

```bash
mkdir -p build && cd build
cmake ..
make -j4
```

3. 安装（可选）：

```bash
make install
```

## 使用方法

### 基本用法

```bash
./bin/edgevoice [模型路径] [音频文件路径]
```

参数说明：
- `模型路径`：OMC模型文件路径，默认为`/data/local/tmp/edgevoice_fast.omc`
- `音频文件路径`：要识别的WAV音频文件路径

示例：
```bash
./bin/edgevoice /data/local/tmp/edgevoice_fast.omc test.wav
```

### 批处理多个音频文件

```bash
./bin/edgevoice [模型路径] [音频文件1] [音频文件2] ...
```

示例：
```bash
./bin/edgevoice /data/local/tmp/edgevoice_fast.omc test1.wav test2.wav test3.wav
```

## 集成到应用中

要将EdgeVoice集成到您的应用中，您需要：

1. 包含头文件：
```cpp
#include "inference_engine.h"
```

2. 创建并初始化推理引擎：
```cpp
edgevoice::InferenceEngine engine("/data/local/tmp/edgevoice_fast.omc");
if (!engine.init()) {
    // 处理初始化失败
}
```

3. 进行意图识别：
```cpp
// 从WAV文件识别
auto result = engine.predictFromWavFile("audio.wav");

// 或从音频数据识别
std::vector<float> audio_data = /* 获取音频数据 */;
int sample_rate = 16000;
auto result = engine.predictIntent(audio_data, sample_rate);

// 使用识别结果
std::cout << "意图: " << result.intent_class << std::endl;
std::cout << "置信度: " << result.confidence << std::endl;
```

## 模型转换

使用`tools/model_converter.py`将ONNX模型转换为HiAI OMC格式：

```bash
python tools/model_converter.py --input model.onnx --output model.omc
```

## 性能测试

使用`tools/performance_tester.py`测试模型在设备上的性能：

```bash
python tools/performance_tester.py --model /data/local/tmp/edgevoice_fast.omc --test_dir /path/to/test/wavs
```

## 注意事项

- 确保音频文件为WAV格式，采样率为16kHz，单声道
- 模型文件必须是HiAI支持的OMC格式
- 在设备上运行时，确保有足够的权限访问模型和音频文件 