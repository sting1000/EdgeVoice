# GE2E-KWS Loss 实现文档

## 概述

本项目实现了基于论文 **arXiv:2410.16647v1** 的 **Generalized End-to-End (GE2E) Loss**，专门用于关键词检测任务中的嵌入向量学习。GE2E Loss通过对比学习的方式，使得同一关键词的嵌入向量相互聚集，不同关键词的嵌入向量相互分离，从而提升关键词检测的准确性。

## 核心理论

### 1. 批次结构要求
GE2E Loss的工作依赖于特殊构造的训练批次：
- 每个批次包含 `X` 个不同的关键词（或意图）
- 每个关键词包含 `Y` 条不同的音频样本
- 因此，批次的总大小为 `X × Y`

### 2. 注册/测试分离
在每个批次内，对于每个关键词的 `Y` 条音频：
- **注册集 (Enrollment Set)**: 前 `Y/2` 条音频，用于计算该关键词的代表性"质心"
- **测试集 (Test Set)**: 后 `Y/2` 条音频，用于与所有质心进行比较并计算损失

### 3. 质心计算
每个关键词的质心是其注册集中所有音频嵌入向量的算术平均值：
```
c_i = (1/|E_i|) ∑_{e ∈ E_i} e
```
其中 `E_i` 是第 `i` 个关键词的注册集。

### 4. 损失函数
核心损失函数的目标是最大化测试集中的嵌入向量与其对应质心的余弦相似度，同时最小化其与其他质心的相似度：

```
L(c_i) = log∑_{n∈N_i} exp(cos(c_i, n)) - log∑_{p∈P_i} exp(cos(c_i, p))
```

其中：
- `N_i` 是不属于第 `i` 个关键词的所有测试样本（负样本）
- `P_i` 是属于第 `i` 个关键词的测试样本（正样本）
- `cos(·,·)` 是余弦相似度函数

## 实现文件结构

```
EdgeVoice/
├── models/
│   ├── ge2e_loss.py           # GE2E Loss核心实现
│   └── streaming_conformer.py # 修改后的模型，支持嵌入向量输出
├── examples/
│   └── ge2e_example.py        # 完整使用示例
├── train_ge2e.py              # GE2E训练脚本
└── README_GE2E.md            # 本文档
```

## 核心组件

### 1. GE2ELoss 类

```python
from models.ge2e_loss import GE2ELoss

# 创建损失函数
criterion = GE2ELoss(init_w=10.0, init_b=-5.0)

# 计算损失
loss = criterion(embeddings, num_phrases, num_utterances_per_phrase)
```

**参数说明：**
- `init_w`: 缩放因子的初始值，用于调整相似度的动态范围
- `init_b`: 偏置的初始值，用于调整相似度基线
- `embeddings`: 模型输出的嵌入向量，形状为 `(X×Y, embedding_dim)`
- `num_phrases`: 批次中的关键词数量 `X`
- `num_utterances_per_phrase`: 每个关键词的音频数量 `Y`

### 2. GE2EBatchSampler 类

```python
from models.ge2e_loss import GE2EBatchSampler

# 创建批次采样器
batch_sampler = GE2EBatchSampler(
    labels=dataset.get_labels(),
    num_phrases_per_batch=4,
    num_utterances_per_phrase=8,
    shuffle=True
)
```

该采样器确保每个批次都符合GE2E Loss的要求，包含指定数量的关键词和每个关键词的音频样本。

### 3. StreamingConformer 模型增强

在原有的 `StreamingConformer` 模型基础上，我们添加了以下方法：

```python
# 获取嵌入向量
embeddings = model.get_embeddings(audio_features)

# 同时获取分类结果和嵌入向量
logits, embeddings = model.forward_with_embeddings(audio_features)
```

## 使用指南

### 1. 基本使用

```python
import torch
from models.streaming_conformer import StreamingConformer
from models.ge2e_loss import GE2ELoss

# 创建模型
model = StreamingConformer(
    input_dim=48,
    hidden_dim=128,
    num_classes=4,
    num_layers=6,
    dropout=0.1
)

# 创建GE2E Loss
criterion = GE2ELoss(init_w=10.0, init_b=-5.0)

# 创建优化器（注意包含损失函数的参数）
optimizer = torch.optim.Adam(
    list(model.parameters()) + list(criterion.parameters()),
    lr=1e-4
)

# 训练循环
for batch in dataloader:
    features, labels = batch
    
    # 获取嵌入向量
    embeddings = model.get_embeddings(features)
    
    # 计算GE2E Loss
    loss = criterion(embeddings, num_phrases=4, num_utterances_per_phrase=8)
    
    # 反向传播
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

### 2. 数据准备

为了使用GE2E Loss，您的数据集需要满足以下要求：

1. **标注文件格式** (CSV):
```csv
file_path,intent
audio1.wav,hello
audio2.wav,hello
audio3.wav,thanks
audio4.wav,thanks
...
```

2. **每个关键词需要足够的样本**:
   - 至少需要 `num_utterances_per_phrase` 个样本
   - 推荐每个关键词有 20-50 个样本

3. **使用专用的数据加载器**:
```python
from torch.utils.data import DataLoader
from models.ge2e_loss import GE2EBatchSampler

# 创建数据集
dataset = YourDataset(...)

# 创建GE2E批次采样器
batch_sampler = GE2EBatchSampler(
    labels=dataset.get_labels(),
    num_phrases_per_batch=4,
    num_utterances_per_phrase=8,
    shuffle=True
)

# 创建数据加载器
dataloader = DataLoader(
    dataset,
    batch_sampler=batch_sampler,
    collate_fn=your_collate_fn,
    num_workers=4
)
```

### 3. 训练配置建议

- **批次配置**:
  - `num_phrases_per_batch`: 4-8（取决于GPU内存）
  - `num_utterances_per_phrase`: 8-12
  - 总批次大小: 32-96

- **学习率**:
  - 模型参数: 1e-4 到 1e-3
  - GE2E参数会自动学习

- **训练策略**:
  - 使用梯度裁剪（max_norm=1.0）
  - 学习率衰减（StepLR或CosineAnnealingLR）
  - 早停机制

### 4. 推理使用

```python
# 加载训练好的模型
model.load_state_dict(torch.load('best_model.pth'))
model.eval()

# 注册关键词（计算质心）
registration_features = get_registration_features()  # 每个关键词多个样本
with torch.no_grad():
    registration_embeddings = model.get_embeddings(registration_features)
    centroids = compute_centroids(registration_embeddings)  # 计算质心

# 实时检测
test_audio = get_test_audio()
with torch.no_grad():
    test_embedding = model.get_embeddings(test_audio)
    similarities = torch.matmul(test_embedding, centroids.T)
    
    # 做出决策
    best_match = torch.argmax(similarities)
    confidence = torch.max(similarities)
    
    if confidence > threshold:
        print(f"检测到关键词: {keyword_names[best_match]}")
    else:
        print("未检测到已注册的关键词")
```

## 运行示例

### 1. 运行基本测试
```bash
python models/ge2e_loss.py
```

### 2. 运行完整示例
```bash
python examples/ge2e_example.py
```

### 3. 运行训练脚本
```bash
python train_ge2e.py \
    --data_dir ./data \
    --annotation_file ./data/annotations.csv \
    --num_phrases 4 \
    --num_utterances 8 \
    --epochs 100 \
    --lr 1e-4
```

## 性能分析

### 评估指标
1. **GE2E Loss值**: 损失越小，表示嵌入质量越好
2. **类内距离**: 同一关键词嵌入向量间的平均距离
3. **类间距离**: 不同关键词嵌入向量间的平均距离
4. **分离度**: 类间距离/类内距离的比值，越大越好
5. **类内相似度**: 同一关键词嵌入向量的余弦相似度

### 示例输出
```
   类内平均距离: 0.8379 ± 0.0845
   类间平均距离: 0.8239 ± 0.0886
   分离度 (类间/类内): 0.9833
   类内平均余弦相似度: 0.6454 ± 0.0699
```

理想情况下：
- 类内距离应该较小（< 1.0）
- 类间距离应该较大
- 分离度应该 > 1.0（越大越好）
- 类内相似度应该较高（> 0.5）

## 常见问题

### Q1: 为什么GE2E Loss有时为负值？
A: 这是正常现象。GE2E Loss = log(负样本和) - log(正样本和)，当正样本相似度很高时，损失可能为负。

### Q2: 如何选择合适的num_phrases和num_utterances？
A: 
- `num_phrases`: 取决于您的关键词数量和GPU内存，通常4-8个
- `num_utterances`: 至少4个（保证注册/测试各2个），推荐8-12个

### Q3: 训练过程中损失不下降怎么办？
A: 
- 检查批次结构是否正确
- 降低学习率
- 确保数据质量和标签正确性
- 检查梯度是否正常

### Q4: 如何在现有项目中集成？
A: 
1. 修改您的模型添加`get_embeddings()`方法
2. 使用`GE2EBatchSampler`替换原有的采样器
3. 将GE2E Loss添加到优化器参数中
4. 在训练循环中使用嵌入向量而非logits

## 论文引用

```bibtex
@article{ge2e_kws_2024,
    title={GE2E-KWS: Generalized End-to-End Loss for Keyword Spotting},
    author={Authors},
    journal={arXiv preprint arXiv:2410.16647v1},
    year={2024}
}
```

## 许可证

本实现遵循项目的开源许可证。

## 贡献

欢迎提交问题和改进建议！如果您在使用过程中遇到任何问题，请在GitHub上提交Issue。 