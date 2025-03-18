# config.py  
import os  
import torch

# 音频参数  
SAMPLE_RATE = 16000  #部署时使用48000
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
N_MFCC = 13  
N_FFT = int(FRAME_LENGTH_MS * TARGET_SAMPLE_RATE / 1000)  
HOP_LENGTH = int(FRAME_SHIFT_MS * TARGET_SAMPLE_RATE / 1000)  
CONTEXT_FRAMES = 2  

# 模型参数  
FAST_MODEL_HIDDEN_SIZE = 64  
PRECISE_MODEL_HIDDEN_SIZE = 128  
INTENT_CLASSES = ["CAPTURE_AND_DESCRIBE", "CAPTURE_REMEMBER",   
                  "CAPTURE_SCAN_QR", "TAKE_PHOTO", "START_RECORDING",   
                  "STOP_RECORDING", "GET_BATTERY_LEVEL", "OTHERS"]  
FAST_CONFIDENCE_THRESHOLD = 0.9  

# Conformer模型参数 - 平衡方案
CONFORMER_LAYERS = 4
CONFORMER_ATTENTION_HEADS = 8
CONFORMER_HIDDEN_SIZE = 256
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

# 增强特征配置
USE_ENHANCED_FEATURES = True      # 是否使用增强特征

# 混淆类别处理
# 一些类别在声学上很相似，可能会混淆，可通过增加它们的权重来改进
CONFUSION_CLASS_WEIGHTS = {
    "TAKE_PHOTO": 1.2,            # 增加TAKE_PHOTO的权重，避免与START_RECORDING混淆
    "START_RECORDING": 1.2,        # 增加START_RECORDING的权重，避免与TAKE_PHOTO混淆
    "STOP_RECORDING": 1.3,         # 增加STOP_RECORDING的权重，避免与GET_BATTERY_LEVEL混淆
}