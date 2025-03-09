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
import threading
import atexit

# 音频参数
SAMPLE_RATE = 16000  # 采样率
CHANNELS = 1         # 单声道
CHUNK = 1024         # 每个缓冲区的帧数
FORMAT = pyaudio.paInt16  # 16位格式
MAX_RECORDING_SECONDS = 10  # 最大录音时长

# 全局PyAudio实例，避免重复创建和销毁
_global_pyaudio = None

def get_pyaudio_instance():
    """获取全局PyAudio实例"""
    global _global_pyaudio
    if _global_pyaudio is None:
        _global_pyaudio = pyaudio.PyAudio()
    return _global_pyaudio

def cleanup_pyaudio():
    """清理全局PyAudio资源"""
    global _global_pyaudio
    if _global_pyaudio is not None:
        _global_pyaudio.terminate()
        _global_pyaudio = None

# 程序退出时确保清理资源
atexit.register(cleanup_pyaudio)

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
        self.stream = None
        self.frames = []
        self.is_recording = False
        self.audio_data = None
        self.recording_thread = None
        self._lock = threading.Lock()
    
    def start_recording(self):
        """开始录音"""
        with self._lock:
            if self.is_recording:
                return
                
            try:
                self.p = get_pyaudio_instance()
                self.frames = []
                self.stream = self.p.open(
                    format=self.format,
                    channels=self.channels,
                    rate=self.sample_rate,
                    input=True,
                    frames_per_buffer=self.chunk
                )
                self.is_recording = True
            except Exception as e:
                print(f"打开音频流时出错: {e}")
                return
        
        # 开始收集音频数据
        start_time = time.time()
        try:
            while self.is_recording and (time.time() - start_time) < MAX_RECORDING_SECONDS:
                if self.stream and self.is_recording:
                    data = self.stream.read(self.chunk, exception_on_overflow=False)
                    self.frames.append(data)
                else:
                    break
                time.sleep(0.001)  # 防止CPU过载
        except Exception as e:
            print(f"录音时出错: {e}")
        finally:
            self._close_stream()
    
    def _close_stream(self):
        """安全关闭音频流"""
        with self._lock:
            if self.stream:
                try:
                    self.stream.stop_stream()
                    self.stream.close()
                except Exception as e:
                    print(f"关闭音频流时出错: {e}")
                finally:
                    self.stream = None
    
    def stop_recording(self):
        """停止录音"""
        # 设置标志，通知录音线程停止
        self.is_recording = False
        
        # 等待录音线程结束（如果存在）
        if self.recording_thread and self.recording_thread.is_alive():
            self.recording_thread.join(timeout=1.0)
        
        # 确保流被关闭
        self._close_stream()
        
        # 将帧转换为音频数据
        with self._lock:
            if self.frames:
                try:
                    audio_bytes = b''.join(self.frames)
                    self.audio_data = np.frombuffer(audio_bytes, dtype=np.int16)
                except Exception as e:
                    print(f"处理录音数据时出错: {e}")
                    self.audio_data = None
            else:
                self.audio_data = None
    
    def play_audio(self):
        """播放录制的音频"""
        if self.audio_data is not None:
            try:
                sd.play(self.audio_data, self.sample_rate)
                sd.wait()
            except Exception as e:
                print(f"播放音频时出错: {e}")
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
            
            # 使用全局PyAudio实例获取样本大小
            p = get_pyaudio_instance()
            
            # 保存为WAV文件
            with wave.open(file_path, 'wb') as wf:
                wf.setnchannels(self.channels)
                wf.setsampwidth(p.get_sample_size(self.format))
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