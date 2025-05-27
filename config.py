# config.py  
import os  
import torch

# 音频参数（核心）  
SAMPLE_RATE = 48000  # 原始采样率
TARGET_SAMPLE_RATE = 16000  # 处理用采样率
FRAME_LENGTH_MS = 25  # 帧长度（毫秒）
FRAME_SHIFT_MS = 10  # 帧移（毫秒）

# VAD参数（语音活动检测）
VAD_ENERGY_THRESHOLD = 0.05  # 能量阈值，值越小检测越敏感
VAD_ZCR_THRESHOLD = 0.15  # 过零率阈值
MIN_SPEECH_MS = 100  # 最小语音段长度（毫秒）
MIN_SILENCE_MS = 300  # 最小静音段长度（毫秒）
MAX_COMMAND_DURATION_S = 5  # 最大命令时长（秒）

# 特征提取参数（核心）
N_MFCC = 32  # MFCC系数数量，增加可提供更多频域细节，推荐范围：13-40
N_FFT = 512  # FFT窗口大小，影响频域分辨率，通常为2的幂
HOP_LENGTH = int(FRAME_SHIFT_MS * TARGET_SAMPLE_RATE / 1000)  # 帧移对应的采样点数
CONTEXT_FRAMES = 4  # 上下文帧数，影响时序建模能力

# 模型参数（核心）
INTENT_CLASSES = ["TAKE_PHOTO_ONE", "START_RECORDING_ONE", 
                  "STOP_RECORDING_ONE", "CAPTURE_AND_DESCRIBE_ONE", "OTHERS"]  

# Conformer模型参数（核心，影响模型能力和大小）
CONFORMER_LAYERS = 4  # 层数，增加可提高模型容量，也增加计算量，推荐范围：2-6
CONFORMER_ATTENTION_HEADS = 8  # 注意力头数，通常为2的幂，推荐范围：4-16
CONFORMER_HIDDEN_SIZE = 96  # 隐藏层大小，影响模型容量，推荐为16的倍数（满足硬件优化）
CONFORMER_CONV_KERNEL_SIZE = 9  # 卷积核大小，影响感受野，推荐范围：9-31（奇数）
CONFORMER_FF_EXPANSION_FACTOR = 4  # 前馈网络扩展因子，通常为4
CONFORMER_DROPOUT = 0.25  # Dropout比例，防止过拟合，推荐范围：0.1-0.5

# 训练参数（核心）
BATCH_SIZE = 32  # 批大小，根据GPU内存调整，通常为2的幂
LEARNING_RATE = 2e-4  # 学习率，影响收敛速度和稳定性，推荐范围：1e-5至5e-4
NUM_EPOCHS = 30  # 训练轮数
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"  

# 路径  
DATA_DIR = "data"  
MODEL_DIR = "saved_models"  
os.makedirs(DATA_DIR, exist_ok=True)  
os.makedirs(MODEL_DIR, exist_ok=True)

# 流式处理参数（核心，影响实时推理）
# 基于语音数据特点优化：主要分布在5s以下，绝大部分2s以下
# 2秒 = 200帧，5秒 = 500帧

# 自适应chunk size策略
STREAMING_CHUNK_SIZE = 200  # 增加到200帧(2秒)，覆盖大部分完整命令
STREAMING_CHUNK_SIZE_SMALL = 100  # 备选小chunk size(1秒)，用于低延迟场景
STREAMING_CHUNK_SIZE_LARGE = 300  # 备选大chunk size(3秒)，用于复杂命令

MAX_CACHED_FRAMES = 150  # 增加缓存帧数，保留1.5秒历史信息
STREAMING_STEP_SIZE = 100  # 增加步长，减少重叠计算，提高效率

# 自适应策略参数
ADAPTIVE_CHUNK_SIZE = True  # 是否启用自适应chunk size
CONFIDENCE_THRESHOLD_EARLY = 0.9  # 高置信度提前决策阈值
CONFIDENCE_THRESHOLD_EXTEND = 0.7  # 低置信度延长处理阈值

# 数据增强参数
AUGMENT_PROB = 0.8  # 总体数据增强概率，建议范围：0.5-0.9
MIXUP_ALPHA = 0.2   # MixUp增强强度，影响混合程度，推荐范围：0.1-0.4
NOISE_STD = 0.01    # 高斯噪声标准差，建议范围：0.001-0.02
USE_MIXUP = True    # 是否使用MixUp增强

# 训练策略参数
USE_LABEL_SMOOTHING = True  # 是否使用标签平滑
LABEL_SMOOTHING = 0.15  # 标签平滑强度，推荐范围：0.05-0.2
USE_COSINE_SCHEDULER = True  # 是否使用余弦学习率调度
USE_EARLY_STOPPING = True  # 是否使用早停
EARLY_STOPPING_PATIENCE = 8  # 早停耐心值，推荐范围：5-10
PROGRESSIVE_TRAINING = True  # 是否使用渐进式长度训练

# 渐进式流式训练参数
PROGRESSIVE_STREAMING_TRAINING = True  # 是否启用渐进式流式训练
STREAMING_TRAINING_SCHEDULE = {
    'phase1': {'epochs': (1, 10), 'streaming_ratio': 0.0},   # 纯完整序列训练
    'phase2': {'epochs': (11, 20), 'streaming_ratio': 0.3},  # 30% 流式训练
    'phase3': {'epochs': (21, 30), 'streaming_ratio': 0.7}   # 70% 流式训练
}

# EdgeVoice验证参数
EDGEVOICE_VALIDATION = True  # 是否启用EdgeVoice特定验证
TARGET_ACCURACY_QUIET = 0.95  # 安静环境目标准确率
TARGET_ACCURACY_NOISY = 0.90  # 噪声环境目标准确率
TARGET_STABILITY_SCORE = 0.85  # 目标稳定性评分
MAX_PREDICTION_CHANGES = 2  # 最大允许预测变化次数

# 核心指令定义（用于重点评估）
CORE_COMMANDS = ['TAKE_PHOTO', 'START_RECORDING', 'STOP_RECORDING']

# 流式训练损失权重
FINAL_PREDICTION_WEIGHT = 1.0  # 最终预测损失权重
STABILITY_LOSS_WEIGHT = 0.1    # 稳定性损失权重（可选）