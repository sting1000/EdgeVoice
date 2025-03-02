# feature_extraction.py  
import numpy as np  
import librosa  
from config import *  

class FeatureExtractor:  
    def __init__(self, sample_rate=TARGET_SAMPLE_RATE,   
                 n_mfcc=N_MFCC, n_fft=N_FFT,   
                 hop_length=HOP_LENGTH, context_frames=CONTEXT_FRAMES):  
        self.sample_rate = sample_rate  
        self.n_mfcc = n_mfcc  
        self.n_fft = n_fft  
        self.hop_length = hop_length  
        self.context_frames = context_frames  
        
    def extract_mfcc(self, audio):  
        """提取MFCC特征"""  
        mfcc = librosa.feature.mfcc(y=audio, sr=self.sample_rate,   
                                   n_mfcc=self.n_mfcc, n_fft=self.n_fft,   
                                   hop_length=self.hop_length)  
        
        # 计算delta和delta-delta特征  
        delta_mfcc = librosa.feature.delta(mfcc)  
        delta2_mfcc = librosa.feature.delta(mfcc, order=2)  
        
        # 合并特征  
        features = np.concatenate([mfcc, delta_mfcc, delta2_mfcc], axis=0)  
        
        # 转置以便每行对应一帧  
        features = features.T  # (n_frames, n_features)  
        
        return features  
        
    def add_context(self, features):  
        """添加上下文帧信息"""  
        n_frames = features.shape[0]  
        n_features = features.shape[1]  
        context_size = 2 * self.context_frames + 1  
        
        # 带上下文的特征矩阵  
        context_features = np.zeros((n_frames, context_size * n_features))  
        
        # 填充上下文特征  
        for i in range(n_frames):  
            # 对于每一帧，收集上下文帧  
            context = np.zeros((context_size, n_features))  
            
            for j in range(-self.context_frames, self.context_frames + 1):  
                context_idx = j + self.context_frames  
                frame_idx = i + j  
                
                if 0 <= frame_idx < n_frames:  
                    context[context_idx] = features[frame_idx]  
                else:  
                    # 边界情况使用零填充  
                    context[context_idx] = np.zeros(n_features)  
            
            # 将上下文扁平化为一个向量  
            context_features[i] = context.flatten()  
            
        return context_features  
        
    def extract_features(self, audio):  
        """提取所有声学特征"""  
        # 提取基础MFCC特征  
        mfcc_features = self.extract_mfcc(audio)  
        
        # 添加上下文信息  
        context_features = self.add_context(mfcc_features)  
        
        return context_features