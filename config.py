# config.py  
import os  
import torch

# 音频参数  
SAMPLE_RATE = 48000
TARGET_SAMPLE_RATE = 16000  
FRAME_LENGTH_MS = 25  
FRAME_SHIFT_MS = 10  
BIT_DEPTH = 24  
TARGET_BIT_DEPTH = 16  

# VAD参数  
VAD_ENERGY_THRESHOLD = 0.05  
VAD_ZCR_THRESHOLD = 0.15  
MIN_SPEECH_MS = 100  
MIN_SILENCE_MS = 300  
MAX_COMMAND_DURATION_S = 5  

# 特征提取参数  
N_MFCC = 20  # 增加MFCC系数数量，提供更多频域细节
N_FFT = 512  # 增加FFT窗口大小，获取更好的频域分辨率
HOP_LENGTH = int(FRAME_SHIFT_MS * TARGET_SAMPLE_RATE / 1000)  
CONTEXT_FRAMES = 4  # 增加上下文帧，提高长期依赖关系建模

# 模型参数 
INTENT_CLASSES = ["TAKE_PHOTO_ONE", "START_RECORDING_ONE", 
                  "STOP_RECORDING_ONE", "CAPTURE_AND_DESCRIBE_ONE"]  
FAST_CONFIDENCE_THRESHOLD = 0.9  

# Conformer模型参数 - 优化方案
CONFORMER_LAYERS = 6  # 增加层数，提高模型容量
CONFORMER_ATTENTION_HEADS = 8
CONFORMER_HIDDEN_SIZE = 192  # 增加隐藏层维度
CONFORMER_CONV_KERNEL_SIZE = 15  # 减小卷积核，适应短命令特性
CONFORMER_FF_EXPANSION_FACTOR = 4
CONFORMER_DROPOUT = 0.1  # 微调dropout

# DistilBERT模型本地路径
DISTILBERT_MODEL_PATH = os.path.join("models", "distilbert-base-uncased")

# 训练参数  
BATCH_SIZE = 32  
LEARNING_RATE = 2e-4  # 减小学习率，适合更深的模型
NUM_EPOCHS = 30  # 增加训练轮数
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"  

# 路径  
DATA_DIR = "data"  
MODEL_DIR = "saved_models"  
os.makedirs(DATA_DIR, exist_ok=True)  
os.makedirs(MODEL_DIR, exist_ok=True)

# 流式处理参数 - 优化后
STREAMING_CHUNK_SIZE = 15  # 减小每次处理的帧数，适应短命令
MAX_CACHED_FRAMES = 60    # 增加历史缓存，保留更多上下文
STREAMING_STEP_SIZE = 5    # 减小步长，增加特征重叠

# 数据增强参数
AUGMENT_PROB = 0.7  # 总体数据增强概率
MIXUP_ALPHA = 0.2   # MixUp增强强度
TIME_MASK_MAX_LEN = 5  # 时间掩码最大长度
FREQ_MASK_MAX_LEN = 3  # 频率掩码最大长度
NOISE_STD = 0.005    # 高斯噪声标准差
USE_MIXUP = True     # 是否使用MixUp
USE_SPECAUGMENT = True  # 是否使用SpecAugment风格增强

# 训练策略参数
USE_LABEL_SMOOTHING = True  # 是否使用标签平滑
LABEL_SMOOTHING = 0.1  # 标签平滑强度
USE_COSINE_SCHEDULER = True  # 是否使用余弦学习率调度
WARMUP_EPOCHS = 3  # 预热轮数
USE_EARLY_STOPPING = True  # 是否使用早停
EARLY_STOPPING_PATIENCE = 5  # 早停耐心值
PROGRESSIVE_TRAINING = True  # 是否使用渐进式长度训练