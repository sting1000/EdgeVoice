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
N_MFCC = 24  # 增加MFCC系数数量为24，提高特征表示能力
N_FFT = 512  # 保持FFT窗口大小
HOP_LENGTH = int(FRAME_SHIFT_MS * TARGET_SAMPLE_RATE / 1000)  
CONTEXT_FRAMES = 7  # 增加上下文帧到7，进一步提高长期依赖关系建模能力

# 模型参数 
INTENT_CLASSES = ["TAKE_PHOTO_ONE", "START_RECORDING_ONE", 
                  "STOP_RECORDING_ONE", "CAPTURE_AND_DESCRIBE_ONE"]  
FAST_CONFIDENCE_THRESHOLD = 0.8  # 降低置信度阈值，平衡准确率和召回率

# Conformer模型参数 - 优化方案
CONFORMER_LAYERS = 6  # 进一步增加到6层，提高模型容量和表示能力
CONFORMER_ATTENTION_HEADS = 8
CONFORMER_HIDDEN_SIZE = 256  # 增加隐藏层维度到256，增强特征提取能力
CONFORMER_CONV_KERNEL_SIZE = 15  # 增大卷积核到15，增强长序列特征捕捉能力
CONFORMER_FF_EXPANSION_FACTOR = 4
CONFORMER_DROPOUT = 0.25  # 增加到0.25，提高泛化能力

# DistilBERT模型本地路径
DISTILBERT_MODEL_PATH = os.path.join("models", "distilbert-base-uncased")

# 训练参数  
BATCH_SIZE = 16  # 保持批量大小
LEARNING_RATE = 2e-5  # 进一步降低学习率到2e-5，更精细优化
WEIGHT_DECAY = 0.03  # 增加权重衰减，减轻过拟合
NUM_EPOCHS = 40  # 增加训练轮数到40
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"  

# 路径  
DATA_DIR = "data"  
MODEL_DIR = "saved_models"  
os.makedirs(DATA_DIR, exist_ok=True)  
os.makedirs(MODEL_DIR, exist_ok=True)

# 流式处理参数 - 优化后
STREAMING_CHUNK_SIZE = 10  # 减小每次处理的帧数到10，增加处理频率
MAX_CACHED_FRAMES = 120  # 增加历史缓存到120，保留更多历史信息
STREAMING_STEP_SIZE = 3  # 减小步长为3，增加特征重叠，提高连续性
STREAMING_DECISION_THRESHOLD = 0.75  # 降低流式决策阈值
DECISION_SMOOTHING_WINDOW = 5  # 连续5帧预测一致才决策
MIN_DECISION_FRAMES = 3  # 至少需要3帧达到阈值才能形成决策

# 数据增强参数
AUGMENT_PROB = 0.8  # 增加数据增强概率到0.8
MIXUP_ALPHA = 0.4  # 增加MixUp增强强度
TIME_MASK_MAX_LEN = 10  # 增加时间掩码最大长度
FREQ_MASK_MAX_LEN = 8  # 增加频率掩码最大长度
NOISE_STD = 0.015  # 增加高斯噪声标准差，提高鲁棒性
USE_MIXUP = True
USE_SPECAUGMENT = True
USE_TIME_WARP = True  # 启用时间扭曲增强
USE_FEATURE_JITTER = True  # 启用特征抖动增强

# 训练策略参数
USE_LABEL_SMOOTHING = True
LABEL_SMOOTHING = 0.05  # 减小标签平滑强度到0.05，在保持泛化能力的同时增强准确率
USE_COSINE_SCHEDULER = True
WARMUP_EPOCHS = 3
USE_EARLY_STOPPING = True
EARLY_STOPPING_PATIENCE = 10  # 增加早停耐心值，确保模型有足够时间收敛
PROGRESSIVE_TRAINING = True
INITIAL_SEQ_LENGTH = 40  # 初始序列长度
MAX_SEQ_LENGTH = 100  # 最大序列长度
SEQ_LENGTH_STEP = 10  # 序列长度步长

# 流式推理优化参数
USE_LOGIT_SMOOTHING = True  # 启用对数平滑
LOGIT_TEMPERATURE = 0.7  # 温度缩放因子
EMA_ALPHA = 0.8  # 指数移动平均平滑因子
USE_POSITION_BIAS = True  # 使用位置偏置，偏向最近帧
POS_BIAS_SCALE = 0.2  # 位置偏置缩放因子
DYNAMIC_THRESHOLD_ADJUST = True  # 启用动态阈值调整
CONFIDENCE_SMOOTHING = True  # 启用置信度平滑