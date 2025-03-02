
### 1. 环境准备

首先需要安装必要的Python依赖：

```bash
pip install -r requirements.txt
```

### 2. 准备数据集

创建一个CSV格式的注释文件，包含以下列：
- `file_path`: 音频文件相对路径
- `intent`: 意图标签（来自8个预定义类别）
- `transcript`: (可选，用于精确分类器) 语音内容文本

将音频文件放置在`data`目录下，按照注释文件中的路径组织。

### 3. 训练模型

#### 训练一级快速分类器:

```bash
python train.py --annotation_file data/annotations.csv --model_type fast --epochs 20
```

#### 训练二级精确分类器:

```bash
python train.py --annotation_file data/annotations.csv --model_type precise --epochs 20
```

这将在`saved_models`目录中保存训练好的模型。

### 4. 执行演示

使用训练好的模型运行演示程序:

```bash
python demo.py --fast_model saved_models/fast_intent_model.pth --precise_model saved_models/precise_intent_model.pth
```

演示程序会:
1. 启动音频流，持续捕获麦克风输入
2. 等待用户输入"wake"来模拟唤醒词检测
3. 当唤醒词被检测到后，处理最近的音频并识别意图
4. 显示识别结果，包括意图类型、置信度和处理时间