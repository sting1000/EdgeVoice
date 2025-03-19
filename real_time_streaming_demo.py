#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import argparse
import torch
import numpy as np
import time
import librosa
import sounddevice as sd
import threading
import queue
from collections import deque

from config import *
from models.fast_classifier import FastIntentClassifier
from utils.feature_extraction import extract_features_streaming

def parse_args():
    parser = argparse.ArgumentParser(description="EdgeVoice实时流式处理演示")
    parser.add_argument("--model_path", type=str, required=True, help="Fast模型路径")
    parser.add_argument("--audio_file", type=str, default=None, help="测试音频文件路径(可选)")
    parser.add_argument("--use_mic", action="store_true", help="使用麦克风输入")
    parser.add_argument("--buffer_size", type=float, default=0.2, help="音频缓冲区大小(秒)")
    return parser.parse_args()

def load_model(model_path):
    """加载Fast模型"""
    # 加载模型配置和权重
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FastIntentClassifier(input_size=N_MFCC*3)  # 16*3=48维特征
    
    # 加载预训练权重
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
    return model, device

class RealTimeStreamingProcessor:
    """实时流式处理器"""
    def __init__(self, model, device, buffer_size=0.2):
        self.model = model
        self.device = device
        self.buffer_size = buffer_size
        
        # 音频缓冲区
        self.buffer_samples = int(buffer_size * TARGET_SAMPLE_RATE)
        self.audio_buffer = deque(maxlen=int(MAX_COMMAND_DURATION_S * TARGET_SAMPLE_RATE))
        
        # 特征缓存
        self.cached_audio_frames = None
        self.cached_model_states = None
        
        # 处理标志
        self.is_processing = False
        self.result_queue = queue.Queue()
        
        # 结果历史
        self.history = []
        self.confidence_history = []
        
        # 声音活动检测状态
        self.is_speech_active = False
        self.silence_counter = 0
        self.max_silence_frames = int(MIN_SILENCE_MS / 1000 * TARGET_SAMPLE_RATE / self.buffer_samples)
        
    def reset(self):
        """重置处理器状态"""
        self.audio_buffer.clear()
        self.cached_audio_frames = None
        self.cached_model_states = None
        self.is_speech_active = False
        self.silence_counter = 0
        self.history = []
        self.confidence_history = []
    
    def vad_detect(self, audio_chunk):
        """简单的语音活动检测"""
        energy = np.mean(np.abs(audio_chunk))
        is_speech = energy > VAD_ENERGY_THRESHOLD
        
        if is_speech:
            self.silence_counter = 0
            if not self.is_speech_active:
                print("检测到语音开始")
            self.is_speech_active = True
        else:
            if self.is_speech_active:
                self.silence_counter += 1
                if self.silence_counter >= self.max_silence_frames:
                    print("检测到语音结束")
                    self.is_speech_active = False
                    
        return self.is_speech_active
    
    def process_audio_chunk(self, audio_chunk):
        """处理一个音频块"""
        # 将音频块添加到缓冲区
        self.audio_buffer.extend(audio_chunk)
        
        # 检查是否有语音活动
        is_speech = self.vad_detect(audio_chunk)
        if not is_speech and len(self.audio_buffer) < self.buffer_samples:
            return None, 0.0
        
        # 从缓冲区获取固定大小的数据
        buffer_chunk = np.array(list(self.audio_buffer)[-self.buffer_samples:])
        
        # 提取特征
        features, self.cached_audio_frames = extract_features_streaming(
            buffer_chunk, 
            TARGET_SAMPLE_RATE, 
            n_mfcc=N_MFCC, 
            prev_frames=self.cached_audio_frames
        )
        
        # 没有足够的特征帧时跳过
        if features.shape[0] < 1:
            return None, 0.0
        
        # 转换为tensor
        x = torch.FloatTensor(features).unsqueeze(0).to(self.device)
        
        # 流式预测
        with torch.no_grad():
            pred, conf, self.cached_model_states = self.model.predict_streaming(x, self.cached_model_states)
        
        pred_class = INTENT_CLASSES[pred.item()]
        confidence = conf.item()
        
        # 保存到历史记录
        self.history.append(pred_class)
        self.confidence_history.append(confidence)
        
        # 如果检测到语音结束，或者置信度足够高，返回最终结果
        if (not is_speech and len(self.history) > 0) or confidence > FAST_CONFIDENCE_THRESHOLD:
            # 获取最后一个预测作为结果
            result = pred_class
            final_confidence = confidence
            
            # 如果语音结束，重置状态
            if not is_speech:
                self.reset()
                
            return result, final_confidence
        
        return None, 0.0
    
    def start_realtime_processing(self, audio_source):
        """启动实时处理线程"""
        self.is_processing = True
        
        def process_thread():
            for chunk in audio_source:
                if not self.is_processing:
                    break
                    
                result, confidence = self.process_audio_chunk(chunk)
                if result is not None and confidence > FAST_CONFIDENCE_THRESHOLD:
                    self.result_queue.put((result, confidence))
                    
                    # 如果置信度很高，可以重置状态，开始下一次识别
                    if confidence > 0.95:
                        self.reset()
        
        threading.Thread(target=process_thread, daemon=True).start()
    
    def stop(self):
        """停止处理"""
        self.is_processing = False

def mic_audio_generator(buffer_size):
    """麦克风音频生成器"""
    buffer_samples = int(buffer_size * TARGET_SAMPLE_RATE)
    
    # 输入音频回调函数
    input_queue = queue.Queue()
    def audio_callback(indata, frames, time, status):
        if status:
            print(f"状态: {status}")
        input_queue.put(indata.copy())
    
    # 打开音频流
    stream = sd.InputStream(
        samplerate=TARGET_SAMPLE_RATE,
        channels=1,
        blocksize=buffer_samples,
        callback=audio_callback
    )
    
    with stream:
        print("开始录音，按Ctrl+C停止")
        while True:
            try:
                # 从队列获取音频数据
                data = input_queue.get()
                yield data.flatten()
            except KeyboardInterrupt:
                break

def file_audio_generator(file_path, buffer_size):
    """文件音频生成器，模拟实时输入"""
    # 加载音频
    audio, sr = librosa.load(file_path, sr=TARGET_SAMPLE_RATE)
    
    # 计算块大小
    buffer_samples = int(buffer_size * TARGET_SAMPLE_RATE)
    
    # 分块处理
    for i in range(0, len(audio) - buffer_samples, buffer_samples // 2):  # 50%重叠
        chunk = audio[i:i+buffer_samples]
        if len(chunk) < buffer_samples:
            # 填充最后一个块
            chunk = np.pad(chunk, (0, buffer_samples - len(chunk)))
        
        yield chunk
        
        # 模拟实时处理的延迟
        time.sleep(buffer_size / 2)  # 半个缓冲区的延迟

def main():
    args = parse_args()
    
    # 验证输入选项
    if not args.audio_file and not args.use_mic:
        print("错误: 必须指定音频文件(--audio_file)或使用麦克风(--use_mic)")
        return
    
    # 加载模型
    print("加载模型...")
    model, device = load_model(args.model_path)
    
    # 创建流处理器
    processor = RealTimeStreamingProcessor(model, device, args.buffer_size)
    
    # 选择音频源
    if args.use_mic:
        audio_source = mic_audio_generator(args.buffer_size)
    else:
        print(f"模拟实时处理音频文件: {args.audio_file}")
        audio_source = file_audio_generator(args.audio_file, args.buffer_size)
    
    # 启动实时处理
    processor.start_realtime_processing(audio_source)
    
    # 主线程显示结果
    try:
        while True:
            try:
                result, confidence = processor.result_queue.get(timeout=0.5)
                print(f"\n识别结果: {result} (置信度: {confidence:.4f})\n")
            except queue.Empty:
                pass
            
    except KeyboardInterrupt:
        print("\n用户中断，停止处理")
    finally:
        processor.stop()

if __name__ == "__main__":
    main() 