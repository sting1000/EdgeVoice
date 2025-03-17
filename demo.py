#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
EdgeVoice命令行演示程序
支持实时麦克风输入和文件处理两种模式
"""

import os
import sys
import time
import argparse
import numpy as np
import sounddevice as sd
import soundfile as sf
import wave
import threading
import json
from collections import deque
import librosa
from pathlib import Path

# 引入推理引擎
from inference import IntentInferenceEngine
from config import *

# 检测PyAudio可用性
try:
    import pyaudio
    PYAUDIO_AVAILABLE = True
except ImportError:
    PYAUDIO_AVAILABLE = False
    print("[警告] PyAudio未安装，将使用sounddevice进行音频处理")

# 检测VAD可用性
try:
    import webrtcvad
    VAD_AVAILABLE = True
except ImportError:
    VAD_AVAILABLE = False
    print("[警告] webrtcvad未安装，语音检测功能将不可用")

# 颜色输出
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    
    @staticmethod
    def disable():
        Colors.HEADER = ''
        Colors.BLUE = ''
        Colors.GREEN = ''
        Colors.YELLOW = ''
        Colors.RED = ''
        Colors.ENDC = ''
        Colors.BOLD = ''
        Colors.UNDERLINE = ''

# Windows上禁用颜色输出
if sys.platform.startswith('win'):
    Colors.disable()

def print_intent(result, show_timing=True):
    """打印识别结果"""
    intent = result.get("intent")
    confidence = result.get("confidence", 0.0)
    path = result.get("path", "unknown")
    
    # 打印意图结果
    print(f"\n{Colors.BOLD}识别结果:{Colors.ENDC}")
    print(f"  意图: {Colors.GREEN}{intent}{Colors.ENDC}")
    print(f"  置信度: {Colors.YELLOW}{confidence:.4f}{Colors.ENDC}")
    print(f"  处理路径: {Colors.BLUE}{path}{Colors.ENDC}")
    
    # 如果是ASR路径，打印转写结果
    if path == "asr+nlu" or result.get("transcription"):
        print(f"  转写文本: {Colors.GREEN}{result.get('transcription')}{Colors.ENDC}")
        if "asr_confidence" in result:
            print(f"  ASR置信度: {Colors.YELLOW}{result.get('asr_confidence', 0.0):.4f}{Colors.ENDC}")
    
    # 打印时间信息
    if show_timing:
        preprocessing_time = result.get("preprocessing_time", 0.0)
        inference_time = result.get("inference_time", 0.0)
        asr_time = result.get("asr_time", 0.0)
        total_time = result.get("total_time", 0.0)
        
        print(f"\n{Colors.BOLD}处理时间:{Colors.ENDC}")
        print(f"  预处理: {Colors.YELLOW}{preprocessing_time:.2f}ms{Colors.ENDC}")
        if asr_time:
            print(f"  ASR: {Colors.YELLOW}{asr_time:.2f}ms{Colors.ENDC}")
        print(f"  推理: {Colors.YELLOW}{inference_time:.2f}ms{Colors.ENDC}")
        print(f"  总计: {Colors.BOLD}{Colors.YELLOW}{total_time:.2f}ms{Colors.ENDC}")

class MicrophoneStream:
    """麦克风流处理类"""
    
    def __init__(self, rate=16000, chunk_size=1024, device=None):
        self.rate = rate
        self.chunk_size = chunk_size
        self.device = device
        self.stream = None
        self.frames = deque(maxlen=100)  # 存储最近的音频帧
        self.stopped = False
        self.vad = None
        self.voice_detected = False
        self.silence_frames = 0
        
        # 如果VAD可用，初始化
        if VAD_AVAILABLE:
            self.vad = webrtcvad.Vad(VAD_MODE)
    
    def start(self):
        """开始麦克风流"""
        self.stopped = False
        
        # 根据可用性选择音频处理库
        if PYAUDIO_AVAILABLE:
            self._start_pyaudio()
        else:
            self._start_sounddevice()
        
        # 启动处理线程
        self.thread = threading.Thread(target=self._process_frames)
        self.thread.daemon = True
        self.thread.start()
        
        return self
    
    def _start_pyaudio(self):
        """使用PyAudio启动麦克风流"""
        self.audio = pyaudio.PyAudio()
        self.stream = self.audio.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=self.rate,
            input=True,
            frames_per_buffer=self.chunk_size,
            input_device_index=self.device,
            stream_callback=self._pyaudio_callback
        )
        self.stream.start_stream()
    
    def _pyaudio_callback(self, in_data, frame_count, time_info, status):
        """PyAudio回调函数"""
        if not self.stopped:
            self.frames.append(in_data)
        return None, pyaudio.paContinue
    
    def _start_sounddevice(self):
        """使用sounddevice启动麦克风流"""
        def callback(indata, frames, time, status):
            """sounddevice回调函数"""
            if status:
                print(f"[警告] {status}")
            if not self.stopped:
                # 将float32转换为int16
                audio_int16 = (indata * 32767).astype(np.int16)
                self.frames.append(audio_int16.tobytes())
        
        self.stream = sd.InputStream(
            samplerate=self.rate,
            blocksize=self.chunk_size,
            channels=1,
            dtype='float32',
            callback=callback,
            device=self.device
        )
        self.stream.start()
    
    def _process_frames(self):
        """处理音频帧，进行语音检测"""
        while not self.stopped:
            if self.vad and len(self.frames) > 0:
                # 只在需要时取出一帧进行VAD
                frame = self.frames[-1]
                is_speech = self.vad.is_speech(frame, self.rate)
                
                if is_speech:
                    if not self.voice_detected:
                        print("\r检测到语音...   ", end='', flush=True)
                    self.voice_detected = True
                    self.silence_frames = 0
                else:
                    if self.voice_detected:
                        self.silence_frames += 1
                        if self.silence_frames > int(self.rate / self.chunk_size * (VAD_PADDING_DURATION_MS / 1000)):
                            self.voice_detected = False
                            print("\r等待语音输入...", end='', flush=True)
            
            time.sleep(0.01)  # 减少CPU使用率
    
    def get_audio_data(self, duration_seconds=None):
        """获取指定时长的音频数据"""
        if duration_seconds is None:
            # 默认返回所有缓存的帧
            frames_list = list(self.frames)
            self.frames.clear()
        else:
            # 计算需要多少帧
            frames_needed = int(duration_seconds * self.rate / self.chunk_size)
            frames_list = []
            
            # 获取最近的N帧
            i = 0
            while i < frames_needed and len(self.frames) > 0:
                frames_list.append(self.frames.popleft())
                i += 1
        
        # 拼接所有帧
        audio_data = b''.join(frames_list)
        
        # 将字节转换为numpy数组
        if PYAUDIO_AVAILABLE:
            # PyAudio使用Int16格式
            audio_np = np.frombuffer(audio_data, dtype=np.int16)
        else:
            # sounddevice已经是float32
            audio_np = np.frombuffer(audio_data, dtype=np.int16)
        
        # 转换为float32，范围[-1, 1]
        audio_float = audio_np.astype(np.float32) / 32767
        
        return audio_float
    
    def stop(self):
        """停止麦克风流"""
        self.stopped = True
        
        if self.stream:
            if PYAUDIO_AVAILABLE:
                self.stream.stop_stream()
                self.stream.close()
                self.audio.terminate()
            else:
                self.stream.stop()
                self.stream.close()
        
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=1.0)

def process_audio_file(filepath, engine, use_asr=True, save_output=False, output_dir=None):
    """处理音频文件"""
    
    if not os.path.exists(filepath):
        print(f"文件不存在: {filepath}")
        return
    
    print(f"处理文件: {filepath}")
    
    # 获取文件名（不带路径和扩展名）
    filename = os.path.basename(filepath)
    filename_no_ext = os.path.splitext(filename)[0]
    
    # 处理音频并预测
    result = engine.process_audio_file(filepath, use_asr=use_asr)
    
    # 打印结果
    print_intent(result)
    
    # 保存结果
    if save_output and output_dir:
        os.makedirs(output_dir, exist_ok=True)
        result_filename = os.path.join(output_dir, f"{filename_no_ext}_result.json")
        
        with open(result_filename, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        
        print(f"结果已保存到: {result_filename}")
    
    return result

def process_directory(dirpath, engine, use_asr=True, save_output=False, output_dir=None, recursive=False):
    """处理目录中的所有音频文件"""
    if not os.path.isdir(dirpath):
        print(f"目录不存在: {dirpath}")
        return
    
    # 支持的音频格式
    audio_extensions = {'.wav', '.mp3', '.flac', '.ogg'}
    
    # 设置输出目录
    if save_output and not output_dir:
        output_dir = os.path.join(dirpath, "results")
    
    # 搜索文件
    if recursive:
        files = [str(f) for f in Path(dirpath).rglob('*') if f.suffix.lower() in audio_extensions]
    else:
        files = [os.path.join(dirpath, f) for f in os.listdir(dirpath) 
                if os.path.isfile(os.path.join(dirpath, f)) and os.path.splitext(f)[1].lower() in audio_extensions]
    
    if not files:
        print(f"未找到音频文件: {dirpath}")
        return
    
    # 处理每个文件
    results = {}
    for i, file in enumerate(files):
        print(f"\n处理文件 [{i+1}/{len(files)}]: {file}")
        result = process_audio_file(file, engine, use_asr, save_output, output_dir)
        results[file] = result
    
    # 打印总结
    print(f"\n总结: 已处理 {len(files)} 个文件")
    return results

def process_microphone(engine, input_device=None, use_asr=True, silence_threshold=SILENCE_THRESHOLD, min_duration=1.0):
    """从麦克风实时处理语音"""
    print("\n初始化麦克风...")
    
    # 设置采样率和块大小
    sample_rate = TARGET_SAMPLE_RATE
    chunk_size = int(sample_rate * VAD_FRAME_DURATION_MS / 1000)  # 与VAD匹配的帧大小
    
    # 初始化数据
    audio_buffer = np.array([], dtype=np.float32)
    is_recording = False
    silence_count = 0
    start_time = None
    
    # 实时音频处理参数
    silence_frames_threshold = int(sample_rate / chunk_size * (MIN_SILENCE_DURATION_MS / 1000))
    min_frames_threshold = int(sample_rate / chunk_size * (min_duration))
    
    # 创建麦克风流
    stream = MicrophoneStream(rate=sample_rate, chunk_size=chunk_size, device=input_device)
    stream.start()
    
    try:
        print("\r等待语音输入...", end='', flush=True)
        
        while True:
            # 检测键盘输入（退出）
            if sys.stdin in select.select([sys.stdin], [], [], 0)[0]:
                line = input()
                if line.strip().lower() in ['q', 'quit', 'exit']:
                    break
            
            # 检测语音活动
            if stream.voice_detected and not is_recording:
                is_recording = True
                start_time = time.time()
                audio_buffer = np.array([], dtype=np.float32)
                print("\r录音中...        ", end='', flush=True)
                silence_count = 0
            
            # 如果是录音状态，则收集音频数据
            if is_recording:
                # 获取最新的音频数据
                new_audio = stream.get_audio_data(0.1)  # 获取100ms的数据
                
                if len(new_audio) > 0:
                    audio_buffer = np.append(audio_buffer, new_audio)
                    
                    # 检测静音
                    if not stream.voice_detected:
                        silence_count += 1
                        
                        # 如果静音持续足够长，结束录音
                        if silence_count >= silence_frames_threshold:
                            duration = time.time() - start_time
                            
                            # 确保录音时长至少为1秒
                            if duration >= min_duration:
                                print("\r处理中...        ", end='', flush=True)
                                
                                # 处理收集到的音频
                                result = engine.process_audio_stream(audio_buffer, use_asr=use_asr)
                                
                                # 打印结果
                                print_intent(result)
                                
                                # 重置状态
                                is_recording = False
                                audio_buffer = np.array([], dtype=np.float32)
                                print("\r等待语音输入...", end='', flush=True)
                    else:
                        # 如果检测到语音，重置静音计数
                        silence_count = 0
            
            # 短暂休息，减少CPU使用
            time.sleep(0.01)
            
    except KeyboardInterrupt:
        print("\n\n退出...")
    except Exception as e:
        print(f"\n错误: {e}")
    finally:
        stream.stop()

def list_audio_devices():
    """列出可用的音频设备"""
    print("\n可用的音频设备:")
    
    if PYAUDIO_AVAILABLE:
        # 使用PyAudio列出设备
        p = pyaudio.PyAudio()
        info = p.get_host_api_info_by_index(0)
        numdevices = info.get('deviceCount')
        
        for i in range(numdevices):
            device_info = p.get_device_info_by_host_api_device_index(0, i)
            if device_info.get('maxInputChannels') > 0:
                print(f"  ID {i}: {device_info.get('name')}")
        
        p.terminate()
    else:
        # 使用sounddevice列出设备
        devices = sd.query_devices()
        for i, device in enumerate(devices):
            if device['max_input_channels'] > 0:
                print(f"  ID {i}: {device['name']}")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="EdgeVoice命令行演示程序")
    parser.add_argument("--fast_model", type=str, default=os.path.join(MODELS_DIR, "fast_model.pt"),
                        help="快速分类器模型路径")
    parser.add_argument("--precise_model", type=str, default=os.path.join(MODELS_DIR, "precise_model.pt"),
                        help="精确分类器模型路径")
    parser.add_argument("--asr_model", type=str, default=ASR_MODEL_PATH,
                        help="ASR模型路径")
    parser.add_argument("--asr_dict", type=str, default=ASR_DICT_PATH,
                        help="ASR字典文件路径")
    parser.add_argument("--mode", type=str, choices=["file", "mic", "dir"], default="mic",
                        help="运行模式:file(处理文件),mic(麦克风输入),dir(处理目录)")
    parser.add_argument("--input", type=str, help="输入文件或目录路径")
    parser.add_argument("--output", type=str, help="结果输出目录")
    parser.add_argument("--save", action="store_true", help="保存处理结果")
    parser.add_argument("--device", type=int, help="麦克风设备ID")
    parser.add_argument("--list_devices", action="store_true", help="列出可用的音频设备")
    parser.add_argument("--threshold", type=float, default=FAST_CONFIDENCE_THRESHOLD,
                        help=f"快速分类器置信度阈值(默认:{FAST_CONFIDENCE_THRESHOLD})")
    parser.add_argument("--asr_threshold", type=float, default=ASR_CONFIDENCE_THRESHOLD,
                        help=f"ASR置信度阈值(默认:{ASR_CONFIDENCE_THRESHOLD})")
    parser.add_argument("--recursive", action="store_true", help="递归处理子目录中的文件")
    parser.add_argument("--disable_asr", action="store_true", help="禁用ASR处理路径")
    parser.add_argument("--save_asr_results", action="store_true", help="保存ASR中间结果")
    
    args = parser.parse_args()
    
    # 列出设备并退出
    if args.list_devices:
        list_audio_devices()
        return
    
    # 检查快速模型路径是否存在
    if not os.path.exists(args.fast_model):
        print(f"错误: 快速分类器模型文件不存在: {args.fast_model}")
        return
    
    # 如果选择了文件或目录模式，但没有提供输入
    if args.mode in ["file", "dir"] and not args.input:
        print(f"错误: {args.mode}模式需要提供--input参数")
        return
    
    # 加载推理引擎
    print(f"加载推理引擎...")
    
    # 检查精确模型是否存在
    precise_model_path = None
    if args.precise_model and os.path.exists(args.precise_model):
        precise_model_path = args.precise_model
    
    # 检查ASR模型是否存在
    asr_model_path = None
    asr_dict_path = None
    if not args.disable_asr:
        if args.asr_model and os.path.exists(args.asr_model):
            asr_model_path = args.asr_model
        if args.asr_dict and os.path.exists(args.asr_dict):
            asr_dict_path = args.asr_dict
    
    # 初始化推理引擎
    engine = IntentInferenceEngine(
        fast_model_path=args.fast_model,
        precise_model_path=precise_model_path,
        asr_model_path=asr_model_path,
        asr_dict_path=asr_dict_path,
        fast_confidence_threshold=args.threshold,
        asr_confidence_threshold=args.asr_threshold,
        save_asr_results=args.save_asr_results
    )
    
    use_asr = not args.disable_asr
    
    # 根据模式运行
    if args.mode == "file":
        process_audio_file(args.input, engine, use_asr, args.save, args.output)
    elif args.mode == "dir":
        process_directory(args.input, engine, use_asr, args.save, args.output, args.recursive)
    else:  # mic模式
        # 在Windows上，select模块只适用于套接字，因此我们需要特殊处理
        if sys.platform.startswith('win'):
            global select
            class DummySelect:
                def select(self, *args, **kwargs):
                    return [], [], []
            select = DummySelect()
        else:
            import select
        
        process_microphone(engine, args.device, use_asr)

if __name__ == "__main__":
    main()