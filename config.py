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
STREAMING_CHUNK_SIZE = 100  # 每次处理的帧数，减小可降低延迟但可能降低准确率，推荐范围：10-30
MAX_CACHED_FRAMES = 100  # 历史缓存帧数，增加可保留更多上下文，推荐范围：40-100
STREAMING_STEP_SIZE = 50  # 步长，影响特征重叠率，推荐范围：5-20

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