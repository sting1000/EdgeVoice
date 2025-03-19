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
N_MFCC = 16  # 从13修改为16，使得总特征维度为48(16x3) - 16的倍数
N_FFT = int(FRAME_LENGTH_MS * TARGET_SAMPLE_RATE / 1000)  
HOP_LENGTH = int(FRAME_SHIFT_MS * TARGET_SAMPLE_RATE / 1000)  
CONTEXT_FRAMES = 2  

# 模型参数 
INTENT_CLASSES = ["CAPTURE_AND_DESCRIBE", "CAPTURE_REMEMBER",   
                  "CAPTURE_SCAN_QR", "TAKE_PHOTO", "START_RECORDING",   
                  "STOP_RECORDING", "GET_BATTERY_LEVEL", "OTHERS"]  
FAST_CONFIDENCE_THRESHOLD = 0.9  

# Conformer模型参数 - 平衡方案
CONFORMER_LAYERS = 4
CONFORMER_ATTENTION_HEADS = 8
CONFORMER_HIDDEN_SIZE = 128
CONFORMER_CONV_KERNEL_SIZE = 31
CONFORMER_FF_EXPANSION_FACTOR = 4
CONFORMER_DROPOUT = 0.2

# DistilBERT模型本地路径
DISTILBERT_MODEL_PATH = os.path.join("models", "distilbert-base-uncased")

# 训练参数  
BATCH_SIZE = 32  
LEARNING_RATE = 1e-3  
NUM_EPOCHS = 20  
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"  

# 路径  
DATA_DIR = "data"  
MODEL_DIR = "saved_models"  
os.makedirs(DATA_DIR, exist_ok=True)  
os.makedirs(MODEL_DIR, exist_ok=True)


# 流式处理参数
STREAMING_CHUNK_SIZE = 10  # 每次处理的帧数（10ms/帧）
MAX_CACHED_FRAMES = 100    # 最大缓存的历史帧数
STREAMING_STEP_SIZE = 5    # 流式处理的步长