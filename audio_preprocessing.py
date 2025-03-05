# audio_preprocessing.py  
import numpy as np  
import librosa  
import torch  
from config import *  

class AudioPreprocessor:  
    def __init__(self, sample_rate=SAMPLE_RATE, target_sample_rate=TARGET_SAMPLE_RATE,  
                 vad_energy_threshold=VAD_ENERGY_THRESHOLD,   
                 vad_zcr_threshold=VAD_ZCR_THRESHOLD,  
                 frame_length_ms=FRAME_LENGTH_MS,   
                 frame_shift_ms=FRAME_SHIFT_MS):  
        self.sample_rate = sample_rate  
        self.target_sample_rate = target_sample_rate  
        self.vad_energy_threshold = vad_energy_threshold  
        self.vad_zcr_threshold = vad_zcr_threshold  
        self.frame_length = int(frame_length_ms * target_sample_rate / 1000)  
        self.frame_shift = int(frame_shift_ms * target_sample_rate / 1000)  
        self.min_speech_frames = int(MIN_SPEECH_MS / frame_shift_ms)  
        self.min_silence_frames = int(MIN_SILENCE_MS / frame_shift_ms)  
        
    def resample(self, audio):  
        """将音频重采样到目标采样率"""  
        if self.sample_rate != self.target_sample_rate:  
            audio = librosa.resample(audio,   
                                    orig_sr=self.sample_rate,   
                                    target_sr=self.target_sample_rate)  
        return audio  
    
    def preemphasis(self, audio, coef=0.97):  
        """应用预加重滤波器"""  
        return np.append(audio[0], audio[1:] - coef * audio[:-1])  
    
    def convert_bit_depth(self, audio, source_depth=BIT_DEPTH, target_depth=TARGET_BIT_DEPTH):  
        """转换位深度"""  
        if source_depth == target_depth:  
            return audio  
            
        # 假设输入已经是归一化的[-1, 1]范围  
        if target_depth == 16:  
            # 转换到16位整数范围  
            return audio  
        else:  
            # 其他转换可以根据需要添加  
            return audio  
    
    def detect_voice_activity(self, audio):  
        """基于能量和过零率的VAD"""  
        frames = librosa.util.frame(audio, frame_length=self.frame_length,   
                                  hop_length=self.frame_shift)  
        
        # 计算每一帧的能量  
        energy = np.sum(frames**2, axis=0)  
        energy = energy / np.max(energy + 1e-10)  
        
        # 计算过零率  
        zcr = librosa.feature.zero_crossing_rate(audio,   
                                              frame_length=self.frame_length,   
                                              hop_length=self.frame_shift)[0]  
        zcr = zcr / np.max(zcr + 1e-10)  
        
        # 确保两个数组长度一致  
        min_length = min(len(energy), len(zcr))  
        energy = energy[:min_length]  
        zcr = zcr[:min_length]  
        
        # 结合能量和过零率进行VAD  
        is_speech = (energy > self.vad_energy_threshold) | (zcr > self.vad_zcr_threshold)  
        zcr = zcr / np.max(zcr + 1e-10)  
        
        # 应用最小语音/静音持续时间约束  
        speech_segments = []  
        in_speech = False  
        count = 0  
        speech_start = 0  
        
        for i, speech_frame in enumerate(is_speech):  
            if speech_frame and not in_speech:  
                # 潜在的语音开始  
                in_speech = True  
                speech_start = i  
                count = 1  
            elif speech_frame and in_speech:  
                # 继续语音状态  
                count += 1  
            elif not speech_frame and in_speech:  
                # 潜在的语音结束  
                if count >= self.min_speech_frames:  
                    # 足够长的语音段  
                    speech_segments.append((speech_start, i))  
                in_speech = False  
                count = 0  
        
        # 处理最后一段可能的语音  
        if in_speech and count >= self.min_speech_frames:  
            speech_segments.append((speech_start, len(is_speech)))  
            
        return speech_segments  
    
    def remove_silence(self, audio):  
        """移除音频中的静音段"""  
        speech_segments = self.detect_voice_activity(audio)  
        if not speech_segments:  
            return audio  # 如果没有检测到语音段，返回原始音频  
            
        # 合并语音段  
        audio_out = np.array([])  
        for start, end in speech_segments:  
            # 转换帧索引到样本索引  
            sample_start = start * self.frame_shift  
            sample_end = min(end * self.frame_shift + self.frame_length, len(audio))  
            audio_out = np.append(audio_out, audio[sample_start:sample_end])  
            
        return audio_out  
    
    def denoise(self, audio):  
        """简单的噪声抑制（这里只是一个示例，实际中可能需要更复杂的算法）"""  
        # 使用简单的频谱减法进行降噪  
        # 这里仅作为示例，实际实现会更复杂  
        S = librosa.stft(audio, n_fft=self.frame_length, hop_length=self.frame_shift)  
        S_mag = np.abs(S)  
        
        # 估计噪声功率谱（使用前几帧）  
        noise_frames = min(10, S_mag.shape[1])  
        noise_power = np.mean(S_mag[:, :noise_frames]**2, axis=1, keepdims=True)  
        
        # 应用谱减法  
        gain = 1 - np.sqrt(noise_power / (S_mag**2 + 1e-10))  
        gain = np.maximum(gain, 0.1)  # 设置最小增益  
        S_denoised = S * gain  
        
        # 逆变换回时域  
        audio_denoised = librosa.istft(S_denoised, hop_length=self.frame_shift,   
                                      length=len(audio))  
        
        return audio_denoised  
    
    def process(self, audio):  
        """应用所有预处理步骤"""  
        # 重采样  
        audio = self.resample(audio)  
        
        # 位深度转换  
        audio = self.convert_bit_depth(audio)  
        
        # 预加重  
        audio = self.preemphasis(audio)  
        
        # 静音移除  
        audio = self.remove_silence(audio)  
        
        # 降噪  
        audio = self.denoise(audio)  
        
        return audio
