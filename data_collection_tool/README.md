# EdgeVoice 语音数据采集工具套件

本工具套件提供了一系列便于采集和管理EdgeVoice语音数据的工具，旨在快速构建高质量的语音意图识别训练数据集。

## 功能特点

- **数据采集工具**：用户友好的GUI界面，实现录音、标注和数据管理
- **数据统计分析**：提供数据集可视化和统计分析功能
- **数据集拆分**：将数据集拆分为训练集、验证集和测试集
- **提示语生成**：扩充语音提示语，增加数据多样性

## 目录结构

```
tools/
├── README.md                 # 本文档
├── run_tool.py               # 启动脚本
├── data_collection_tool.py   # 数据采集GUI工具
├── audio_utilities.py        # 音频处理工具
├── intent_prompts.py         # 意图提示语库
├── data_stats.py             # 数据统计分析工具
└── prompt_generator.py       # 提示语生成器
```

## 安装依赖

确保已安装以下Python包：

```bash
pip install numpy pandas matplotlib seaborn librosa pyaudio sounddevice soundfile scikit-learn
```

## 快速开始

### 1. 采集数据

启动数据采集工具：

```bash
python run_tool.py collect
```

### 2. 生成数据统计报告

分析当前数据集的统计信息：

```bash
python run_tool.py stats
```

### 3. 拆分数据集

将数据集拆分为训练集、验证集和测试集：

```bash
python run_tool.py split
```

### 4. 扩充提示语

生成更多样化的语音提示语：

```bash
python run_tool.py prompts
```

### 5. 查看帮助信息

```bash
python run_tool.py help
```

## 数据采集工具使用说明

### 基本流程

1. 填写用户信息（用户名、性别、年龄组、录音环境）
2. 从下拉菜单选择要采集的意图类别
3. 点击"随机提示"获取新的提示语
4. 点击"开始录音"按钮，朗读屏幕上显示的文本
5. 录音完成后点击"停止录音"
6. 点击"播放录音"检查录音质量
7. 满意后点击"保存录音"将样本添加到数据集

### 注意事项

- 录音时尽量保持环境安静
- 每个样本录音不要超过10秒
- 确保清晰地朗读提示语
- 每个会话默认录制10个样本，完成后会自动创建新会话

## 数据格式说明

### 音频文件

- 格式：WAV
- 采样率：16kHz
- 位深：16-bit
- 通道：单声道

### 注释文件 (annotations.csv)

CSV格式，包含以下字段：
- file_path: 音频文件相对路径
- intent: 意图标签
- transcript: 语音内容文本
- gender: 说话者性别
- age_group: 说话者年龄组
- environment: 录音环境
- session_id: 会话ID
- timestamp: 录制时间戳

## 高级用法

### 自定义提示语

编辑 `intent_prompts.py` 文件可添加或修改各意图的提示语。

### 扩充提示语

运行提示语生成器可自动为基础提示语生成更多样化的表达：

```bash
python prompt_generator.py --variants 5 --output_json expanded_prompts.json
```

### 自定义数据拆分

```bash
python -m data_stats --annotation_file ../data/annotations.csv --split --train_ratio 0.7 --val_ratio 0.15 --test_ratio 0.15
```

## 常见问题解答

**Q: 录音没有声音或音量太小？**  
A: 检查麦克风设置和系统音量，确保麦克风已正确连接并设置为默认录音设备。

**Q: 如何修改每个会话的录音数量？**  
A: 在data_collection_tool.py文件中修改RECORDINGS_PER_SESSION变量。

**Q: 如何更改录音保存位置？**  
A: 修改DATA_DIR和ANNOTATION_FILE变量为您想要的路径。

## 贡献与反馈

欢迎提交问题报告或功能建议，帮助我们改进工具套件。

## 许可证

MIT许可证 