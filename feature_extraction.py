# feature_extraction.py  
import numpy as np  
import librosa  
from config import *  

class FeatureExtractor:  
    def __init__(self, sample_rate=TARGET_SAMPLE_RATE,   
                 n_mfcc=N_MFCC, n_fft=N_FFT,   
                 hop_length=HOP_LENGTH, context_frames=CONTEXT_FRAMES,
                 enhanced_features=False):  
        self.sample_rate = sample_rate  
        self.n_mfcc = n_mfcc  
        self.n_fft = n_fft  
        self.hop_length = hop_length  
        self.context_frames = context_frames  
        self.enhanced_features = enhanced_features
        
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
    
    def extract_enhanced_features(self, audio):
        """提取增强型特征，包括MFCC、梅尔谱和特定声学特征"""
        # 基础特征 - MFCC
        mfcc = librosa.feature.mfcc(y=audio, sr=self.sample_rate,   
                                  n_mfcc=self.n_mfcc, n_fft=self.n_fft,   
                                  hop_length=self.hop_length)
        
        # 计算能量特征
        energy = np.sum(np.abs(librosa.stft(audio, n_fft=self.n_fft, hop_length=self.hop_length))**2, axis=0)
        energy = energy.reshape(1, -1)  # 调整形状以便拼接
        
        # 计算梅尔谱图特征
        mel_spec = librosa.feature.melspectrogram(
            y=audio, 
            sr=self.sample_rate, 
            n_fft=self.n_fft, 
            hop_length=self.hop_length, 
            n_mels=20  # 使用20个梅尔频带
        )
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        
        # 计算声音响度特征
        spectral_contrast = librosa.feature.spectral_contrast(
            y=audio, 
            sr=self.sample_rate, 
            n_fft=self.n_fft, 
            hop_length=self.hop_length
        )
        
        # 计算声谱平坦度
        spectral_flatness = librosa.feature.spectral_flatness(
            y=audio, 
            n_fft=self.n_fft, 
            hop_length=self.hop_length
        )
        
        # 计算声谱质心（区分不同声音音色的重要特征）
        spectral_centroid = librosa.feature.spectral_centroid(
            y=audio, 
            sr=self.sample_rate, 
            n_fft=self.n_fft, 
            hop_length=self.hop_length
        )
        
        # 计算过零率特征（区分浊音和清音）
        zcr = librosa.feature.zero_crossing_rate(
            audio, 
            frame_length=self.n_fft, 
            hop_length=self.hop_length
        )
        
        # 计算delta和delta-delta特征
        delta_mfcc = librosa.feature.delta(mfcc)  
        delta2_mfcc = librosa.feature.delta(mfcc, order=2)
        
        # 合并所有特征
        features = np.concatenate([
            mfcc, 
            delta_mfcc, 
            delta2_mfcc, 
            energy, 
            mel_spec_db[:5],  # 使用前5个梅尔频带特征
            spectral_contrast,
            spectral_flatness,
            spectral_centroid,
            zcr
        ], axis=0)
        
        # 转置以便每行对应一帧
        features = features.T  # (n_frames, n_features)
        
        return features
        
    def add_context(self, features, context_size=None):  
        """添加上下文帧信息，形成最终特征"""  
        if context_size is None:  
            context_size = self.context_frames  
            
        num_frames, feat_dim = features.shape  
        
        # 如果特征帧数太少，使用填充  
        if num_frames < 2 * context_size + 1:  
            pad_size = (2 * context_size + 1) - num_frames  
            features = np.pad(features, ((0, pad_size), (0, 0)), 'constant')  
            num_frames = features.shape[0]  
        
        # 构建上下文特征  
        context_features = []  
        for i in range(context_size, num_frames - context_size):  
            # 提取当前帧的上下文  
            context = []  
            for j in range(i - context_size, i + context_size + 1):  
                context.append(features[j])  
            
            # 合并上下文特征  
            context_feat = np.concatenate(context)  
            context_features.append(context_feat)  
        
        return np.array(context_features)  
        
    def extract_features(self, audio):  
        """提取特征向量，可包含上下文信息"""  
        if self.enhanced_features:
            features = self.extract_enhanced_features(audio)
        else:
            features = self.extract_mfcc(audio)
        
        # 添加上下文信息  
        if self.context_frames > 0:  
            features = self.add_context(features, self.context_frames)  
            
        return features
    
    def get_feature_dim(self):
        """获取特征维度"""
        if self.enhanced_features:
            # 计算增强特征的维度
            # MFCC(16) + Delta(16) + Delta2(16) + Energy(1) + Mel Spectrogram(5) 
            # + Spectral Contrast(7) + Flatness(1) + Centroid(1) + ZCR(1)
            base_dim = N_MFCC * 3 + 1 + 5 + 7 + 1 + 1 + 1  # 总共64维
        else:
            # 基本特征：MFCC + Delta + Delta2
            base_dim = self.n_mfcc * 3  # 通常是16*3=48维
        
        if self.context_frames > 0:
            # 如果有上下文特征，每帧的特征维度乘以上下文窗口大小
            feature_dim = base_dim * (2 * self.context_frames + 1)
        else:
            feature_dim = base_dim
        
        return feature_dim