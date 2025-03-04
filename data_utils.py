# data_utils.py  
import os  
import numpy as np  
import torch  
import librosa  
import pandas as pd  
import time
import warnings
from torch.utils.data import Dataset, DataLoader  
from transformers import DistilBertTokenizer  
from config import *  
from audio_preprocessing import AudioPreprocessor  
from feature_extraction import FeatureExtractor  

def load_audio(file_path, sample_rate=SAMPLE_RATE):  
    """加载音频文件"""  
    audio, sr = librosa.load(file_path, sr=sample_rate, mono=True)  
    return audio, sr

def standardize_audio_length(audio, sr, target_length=MAX_COMMAND_DURATION_S, min_length=0.5):
    """
    标准化音频长度:
    1. 如果音频太长，使用VAD找到最可能的语音段
    2. 如果音频太短，进行居中填充
    3. 确保所有音频都处于合理的长度范围内
    
    Args:
        audio: 音频数据
        sr: 采样率
        target_length: 目标长度(秒)
        min_length: 最小长度(秒)
    
    Returns:
        标准化后的音频
    """
    # 检查当前长度
    current_length = len(audio) / sr
    target_samples = int(target_length * sr)
    
    # 如果长度合适，直接返回
    if min_length <= current_length <= target_length:
        return audio
    
    # 如果音频太长，使用VAD找到主要语音段
    if current_length > target_length:
        # 初始化预处理器
        preprocessor = AudioPreprocessor(sample_rate=sr)
        
        # 检测语音段
        speech_segments = preprocessor.detect_voice_activity(audio)
        
        if speech_segments:
            # 找到最长的语音段
            longest_segment = max(speech_segments, key=lambda x: x[1] - x[0])
            
            # 转换帧索引到样本索引
            frame_shift = int(FRAME_SHIFT_MS * sr / 1000)
            frame_length = int(FRAME_LENGTH_MS * sr / 1000)
            
            # 计算段长度(样本数)
            segment_length = (longest_segment[1] - longest_segment[0]) * frame_shift + frame_length
            
            # 如果最长段仍然太长，选择中间部分
            if segment_length > target_samples:
                sample_start = longest_segment[0] * frame_shift
                # 从语音段中间向两边扩展
                middle = sample_start + segment_length // 2
                half_target = target_samples // 2
                start = max(0, middle - half_target)
                end = min(len(audio), middle + half_target)
                return audio[start:end]
            else:
                # 如果最长段适合目标长度，提取该段
                sample_start = longest_segment[0] * frame_shift
                sample_end = min(sample_start + segment_length, len(audio))
                return audio[sample_start:sample_end]
        else:
            # 如果没有检测到语音段，截取中间部分
            start = (len(audio) - target_samples) // 2
            return audio[start:start+target_samples]
    
    # 如果音频太短，居中填充
    if current_length < min_length:
        padded_audio = np.zeros(int(sr * min_length), dtype=np.float32)
        start_idx = (len(padded_audio) - len(audio)) // 2
        padded_audio[start_idx:start_idx+len(audio)] = audio
        return padded_audio
    
    return audio

def prepare_audio_features(audio, preprocessor, feature_extractor, normalize_length=True):  
    """准备音频特征用于模型输入"""  
    # 标准化音频长度(如果启用)
    if normalize_length:
        sample_rate = preprocessor.target_sample_rate
        audio = standardize_audio_length(audio, sample_rate)
    
    # 预处理音频  
    processed_audio = preprocessor.process(audio)  
    
    # 提取特征  
    features = feature_extractor.extract_features(processed_audio)  
    
    return features  

def text_to_input_ids(text, tokenizer, max_length=128):  
    """将文本转换为模型输入ID"""  
    encoding = tokenizer(  
        text,  
        max_length=max_length,  
        padding='max_length',  
        truncation=True,  
        return_tensors='pt'  
    )  
    
    return encoding['input_ids'], encoding['attention_mask']  

class AudioIntentDataset(Dataset):  
    """语音意图数据集"""  
    def __init__(self, data_dir, annotation_file, transform=True, mode='fast', analyze_audio=False):  
        """  
        data_dir: 音频文件目录  
        annotation_file: 包含文件路径和标签的CSV文件  
        transform: 是否应用预处理和特征提取  
        mode: 'fast'使用纯音频特征，'precise'使用文本+音频特征
        analyze_audio: 是否分析音频长度分布(用于数据集探索)
        """  
        self.data_dir = data_dir  
        self.transform = transform  
        self.mode = mode  
        
        # 加载注释文件  
        self.annotations = pd.read_csv(annotation_file)  
        
        # 获取类别列表和类别到索引的映射  
        self.classes = INTENT_CLASSES  
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}  
        
        # 初始化预处理器和特征提取器  
        self.preprocessor = AudioPreprocessor() if transform else None  
        self.feature_extractor = FeatureExtractor() if transform else None  
        
        # 对于精确模式，初始化分词器  
        if mode == 'precise':
            try:
                # 先尝试从本地路径加载
                self.tokenizer = DistilBertTokenizer.from_pretrained(DISTILBERT_MODEL_PATH)
                print(f"已从本地路径加载DistilBERT分词器: {DISTILBERT_MODEL_PATH}")
            except Exception as e:
                print(f"无法从本地加载分词器，错误: {e}")
                print("尝试从在线资源加载分词器...")
                self.tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        
        # 分析音频长度分布(可选)
        if analyze_audio:
            self._analyze_audio_lengths()
    
    def _analyze_audio_lengths(self):
        """分析数据集中音频文件的长度分布"""
        print("正在分析数据集音频长度分布...")
        durations = []
        too_long = 0
        too_short = 0
        
        for idx in range(min(len(self.annotations), 100)):  # 只分析前100个样本以节省时间
            audio_path = os.path.join(self.data_dir, self.annotations.iloc[idx]['file_path'])
            try:
                audio, sr = load_audio(audio_path)
                duration = len(audio) / sr
                durations.append(duration)
                
                if duration > MAX_COMMAND_DURATION_S:
                    too_long += 1
                elif duration < 0.5:  # 小于0.5秒可能太短
                    too_short += 1
            except Exception as e:
                print(f"无法加载音频文件 {audio_path}: {e}")
        
        if durations:
            print(f"音频长度统计(基于{len(durations)}个样本):")
            print(f"  最小长度: {min(durations):.2f}秒")
            print(f"  最大长度: {max(durations):.2f}秒")
            print(f"  平均长度: {sum(durations)/len(durations):.2f}秒")
            print(f"  过长音频(>{MAX_COMMAND_DURATION_S}秒): {too_long}个")
            print(f"  过短音频(<0.5秒): {too_short}个")
        else:
            print("无法分析音频长度。")
    
    def __len__(self):  
        return len(self.annotations)  
    
    def __getitem__(self, idx):  
        start_time = time.time()
        
        # 获取音频文件路径和标签  
        audio_path = os.path.join(self.data_dir, self.annotations.iloc[idx]['file_path'])  
        intent_label = self.annotations.iloc[idx]['intent']  
        
        # 将标签转换为索引  
        label_idx = self.class_to_idx[intent_label]  
        
        # 加载音频文件  
        try:
            audio, sr = load_audio(audio_path)
            
            # 检查音频长度
            duration = len(audio) / sr
            if duration > MAX_COMMAND_DURATION_S * 1.5:  # 允许一定的超长容忍度
                warnings.warn(f"音频文件过长({duration:.2f}秒): {audio_path}")
            elif duration < 0.3:  # 非常短的音频可能存在问题
                warnings.warn(f"音频文件过短({duration:.2f}秒): {audio_path}")
            
            # 应用预处理和特征提取  
            if self.transform:  
                features = prepare_audio_features(audio, self.preprocessor, self.feature_extractor, normalize_length=True)  
                features_tensor = torch.FloatTensor(features)  
            else:  
                features_tensor = torch.FloatTensor(audio)
                
        except Exception as e:
            print(f"处理音频文件时出错 {audio_path}: {e}")
            # 返回空特征张量，确保训练不会中断
            if self.transform:
                # 创建一个合理大小的空特征张量
                n_features = self.feature_extractor.n_mfcc * 3  # MFCC + delta + delta2
                context_size = 2 * self.feature_extractor.context_frames + 1
                features_tensor = torch.zeros((10, context_size * n_features))  # 10帧作为默认长度
            else:
                features_tensor = torch.zeros((self.preprocessor.target_sample_rate,))  # 1秒静音
        
        # 根据模式返回不同格式的数据  
        if self.mode == 'fast':  
            return features_tensor, label_idx  
        else:  # 'precise'模式  
            # 获取文本转录  
            transcript = self.annotations.iloc[idx]['transcript']  
            
            # 将文本转换为输入ID  
            input_ids, attention_mask = text_to_input_ids(transcript, self.tokenizer)  
            
            return {  
                'features': features_tensor,  
                'input_ids': input_ids.squeeze(0),  
                'attention_mask': attention_mask.squeeze(0),  
                'label': label_idx  
            }  

def prepare_dataloader(data_dir, annotation_file, batch_size=BATCH_SIZE, mode='fast', analyze_audio=False):  
    """准备数据加载器"""  
    dataset = AudioIntentDataset(data_dir, annotation_file, mode=mode, analyze_audio=analyze_audio)  
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)  
    return dataloader