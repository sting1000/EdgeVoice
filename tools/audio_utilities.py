"""
音频处理工具
提供录音、播放和保存功能
"""

import pyaudio
import wave
import numpy as np
import sounddevice as sd
import soundfile as sf
import os
import time
import uuid
from pathlib import Path
import librosa
import librosa.display
import matplotlib.pyplot as plt

# 音频参数
SAMPLE_RATE = 16000  # 采样率
CHANNELS = 1         # 单声道
CHUNK = 1024         # 每个缓冲区的帧数
FORMAT = pyaudio.paInt16  # 16位格式
MAX_RECORDING_SECONDS = 10  # 最大录音时长

class AudioRecorder:
    """音频录制类"""
    
    def __init__(self, sample_rate=SAMPLE_RATE, channels=CHANNELS, chunk=CHUNK, format=FORMAT):
        """
        初始化录音器
        
        参数:
            sample_rate: 采样率
            channels: 声道数
            chunk: 块大小
            format: 音频格式
        """
        self.sample_rate = sample_rate
        self.channels = channels
        self.chunk = chunk
        self.format = format
        self.p = None
        self.stream = None
        self.frames = []
        self.is_recording = False
        self.audio_data = None
    
    def start_recording(self):
        """开始录音"""
        self.p = pyaudio.PyAudio()
        self.frames = []
        self.stream = self.p.open(
            format=self.format,
            channels=self.channels,
            rate=self.sample_rate,
            input=True,
            frames_per_buffer=self.chunk
        )
        self.is_recording = True
        
        # 开始收集音频数据
        start_time = time.time()
        try:
            while self.is_recording and (time.time() - start_time) < MAX_RECORDING_SECONDS:
                data = self.stream.read(self.chunk)
                self.frames.append(data)
        except Exception as e:
            print(f"录音时出错: {e}")
            self.stop_recording()
    
    def stop_recording(self):
        """停止录音"""
        self.is_recording = False
        
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
            self.stream = None
        
        if self.p:
            self.p.terminate()
            self.p = None
        
        # 将帧转换为音频数据
        if self.frames:
            audio_bytes = b''.join(self.frames)
            self.audio_data = np.frombuffer(audio_bytes, dtype=np.int16)
        else:
            self.audio_data = None
    
    def play_audio(self):
        """播放录制的音频"""
        if self.audio_data is not None:
            sd.play(self.audio_data, self.sample_rate)
            sd.wait()
        else:
            print("没有可播放的音频数据")
    
    def save_audio(self, file_path):
        """
        保存音频到WAV文件
        
        参数:
            file_path: 保存的文件路径
        
        返回:
            是否成功保存
        """
        if not self.frames:
            print("没有要保存的音频数据")
            return False
        
        try:
            # 确保目录存在
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            
            # 保存为WAV文件
            with wave.open(file_path, 'wb') as wf:
                wf.setnchannels(self.channels)
                wf.setsampwidth(pyaudio.PyAudio().get_sample_size(self.format))
                wf.setframerate(self.sample_rate)
                wf.writeframes(b''.join(self.frames))
            
            print(f"音频已保存: {file_path}")
            return True
        
        except Exception as e:
            print(f"保存音频时出错: {e}")
            return False
    
    def get_audio_length(self):
        """获取录音时长(秒)"""
        if self.audio_data is None:
            return 0
        
        return len(self.audio_data) / self.sample_rate
    
    def visualize_waveform(self, output_path=None):
        """
        可视化音频波形
        
        参数:
            output_path: 保存图像的路径，为None时显示图像
        """
        if self.audio_data is None:
            print("没有音频数据可视化")
            return
        
        plt.figure(figsize=(10, 4))
        plt.plot(np.arange(len(self.audio_data)) / self.sample_rate, self.audio_data)
        plt.title('音频波形')
        plt.xlabel('时间 (秒)')
        plt.ylabel('振幅')
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path)
            plt.close()
        else:
            plt.show()

def generate_unique_filename(base_dir, intent, extension=".wav"):
    """
    生成唯一的文件名
    
    参数:
        base_dir: 基础目录
        intent: 意图标签
        extension: 文件扩展名
    
    返回:
        唯一的文件路径
    """
    # 确保目录存在
    intent_dir = os.path.join(base_dir, intent)
    os.makedirs(intent_dir, exist_ok=True)
    
    # 生成文件名: 意图_时间戳_UUID.wav
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    unique_id = str(uuid.uuid4())[:8]
    filename = f"{intent}_{timestamp}_{unique_id}{extension}"
    
    return os.path.join(intent_dir, filename)

def load_audio_file(file_path, sr=SAMPLE_RATE):
    """
    加载音频文件
    
    参数:
        file_path: 音频文件路径
        sr: 采样率
    
    返回:
        音频数据和采样率
    """
    try:
        audio, sr = librosa.load(file_path, sr=sr)
        return audio, sr
    except Exception as e:
        print(f"加载音频文件时出错: {e}")
        return None, None 