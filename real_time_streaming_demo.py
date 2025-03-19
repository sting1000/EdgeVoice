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
    
    # 尝试加载模型
    try:
        # 首先尝试加载两阶段训练的模型格式（保存为字典）
        checkpoint = torch.load(model_path, map_location=device)
        
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            # 两阶段训练模型格式
            model_state = checkpoint['model_state_dict']
            
            # 获取意图标签
            if 'intent_labels' in checkpoint:
                intent_labels = checkpoint['intent_labels']
                num_classes = len(intent_labels)
                print(f"已加载意图标签: {intent_labels}")
            else:
                num_classes = len(INTENT_CLASSES)
                intent_labels = INTENT_CLASSES
                print(f"使用默认意图标签: {intent_labels}")
                
            # 初始化模型
            model = FastIntentClassifier(input_size=N_MFCC*3, num_classes=num_classes)
            model.load_state_dict(model_state)
            print("已加载两阶段训练模型")
        else:
            # 直接保存的模型状态
            model = FastIntentClassifier(input_size=N_MFCC*3)  # 16*3=48维特征
            model.load_state_dict(checkpoint)
            intent_labels = INTENT_CLASSES
            print("已加载常规模型")
            
    except Exception as e:
        print(f"加载模型时出错: {e}")
        print("使用默认模型配置")
        model = FastIntentClassifier(input_size=N_MFCC*3)
        intent_labels = INTENT_CLASSES
    
    model.to(device)
    model.eval()
    
    return model, device, intent_labels

class RealTimeStreamingProcessor:
    """实时流式处理器"""
    def __init__(self, model, device, intent_labels, buffer_size=0.2):
        self.model = model
        self.device = device
        self.buffer_size = buffer_size
        self.intent_labels = intent_labels
        
        # 音频缓冲区
        self.buffer_samples = int(buffer_size * TARGET_SAMPLE_RATE)
        self.audio_buffer = deque(maxlen=int(MAX_COMMAND_DURATION_S * TARGET_SAMPLE_RATE))
        
        # 特征缓存
        self.feature_buffer = []
        self.prev_audio_frames = None
        
        # 状态变量
        self.is_speaking = False
        self.silence_frames = 0
        self.speech_frames = 0
        self.cached_states = None
        self.last_prediction = None
        self.last_confidence = 0
        
        # 控制信号
        self.running = True
        self.result_queue = queue.Queue()
        
        # VAD参数
        self.min_speech_frames = int(MIN_SPEECH_MS / 1000 / buffer_size)
        self.min_silence_frames = int(MIN_SILENCE_MS / 1000 / buffer_size)
    
    def reset(self):
        """重置处理状态"""
        self.audio_buffer.clear()
        self.feature_buffer = []
        self.prev_audio_frames = None
        self.is_speaking = False
        self.silence_frames = 0
        self.speech_frames = 0
        self.cached_states = None
        self.last_prediction = None
        self.last_confidence = 0
        print("状态已重置")
    
    def vad_detect(self, audio_chunk):
        """基于能量和过零率的简单VAD"""
        if len(audio_chunk) == 0:
            return False
            
        # 计算能量
        energy = np.mean(np.abs(audio_chunk))
        
        # 计算过零率
        zero_crossings = np.sum(np.abs(np.diff(np.sign(audio_chunk)))) / (2 * len(audio_chunk))
        
        # 判断是否为语音
        is_speech = energy > VAD_ENERGY_THRESHOLD or zero_crossings > VAD_ZCR_THRESHOLD
        
        return is_speech
    
    def process_audio_chunk(self, audio_chunk):
        """处理一个音频块，执行VAD和意图识别"""
        # 检测是否有语音
        is_speech = self.vad_detect(audio_chunk)
        
        # 添加到缓冲区
        self.audio_buffer.extend(audio_chunk)
        
        # 更新语音/静音状态
        if is_speech:
            self.speech_frames += 1
            self.silence_frames = 0
            
            if not self.is_speaking and self.speech_frames > self.min_speech_frames:
                self.is_speaking = True
                print("检测到语音开始...")
        else:
            self.silence_frames += 1
            self.speech_frames = 0
            
            # 如果持续静音且之前在说话，可能是语音结束
            if self.is_speaking and self.silence_frames > self.min_silence_frames:
                self.is_speaking = False
                print("检测到语音结束，处理中...")
                
                # 处理完整的语音命令
                if self.last_prediction is not None:
                    intent_label = self.intent_labels[self.last_prediction]
                    result = {
                        "intent": intent_label,
                        "confidence": self.last_confidence,
                        "audio": list(self.audio_buffer)
                    }
                    self.result_queue.put(result)
                    
                    # 输出结果
                    intent_text = f"{intent_label} ({self.last_confidence:.2f})"
                    print(f"识别结果: {intent_text}")
                    
                # 重置状态为下一个命令做准备
                self.reset()
                return
        
        # 提取特征
        if len(audio_chunk) > 0:
            features, self.prev_audio_frames = extract_features_streaming(
                audio_chunk, 
                TARGET_SAMPLE_RATE, 
                prev_frames=self.prev_audio_frames
            )
            
            # 如果提取到有效特征
            if features.shape[0] > 0:
                # 转换为tensor
                features_tensor = torch.FloatTensor(features).unsqueeze(0).to(self.device)
                
                # 执行推理
                with torch.no_grad():
                    # 使用流式预测模式
                    predicted, confidence, self.cached_states = self.model.predict_streaming(
                        features_tensor, 
                        self.cached_states
                    )
                    
                    # 获取预测结果
                    pred_class = predicted.item()
                    conf_value = confidence.item()
                    
                    # 更新最新预测
                    self.last_prediction = pred_class
                    self.last_confidence = conf_value
                    
                    # 如果置信度高，可以提前输出结果（早停）
                    if conf_value > FAST_CONFIDENCE_THRESHOLD and self.is_speaking:
                        intent_label = self.intent_labels[pred_class]
                        print(f"高置信度预测: {intent_label} ({conf_value:.2f})")
                        
                        # 如果是简单命令，可以提前结束
                        if intent_label in ["TAKE_PHOTO", "GET_BATTERY_LEVEL"]:
                            result = {
                                "intent": intent_label,
                                "confidence": conf_value,
                                "early_stopping": True,
                                "audio": list(self.audio_buffer)
                            }
                            self.result_queue.put(result)
    
    def start_realtime_processing(self, audio_source):
        """启动实时处理线程"""
        self.running = True
        
        def process_thread():
            for chunk in audio_source:
                if not self.running:
                    break
                    
                self.process_audio_chunk(chunk)
        
        threading.Thread(target=process_thread, daemon=True).start()
    
    def stop(self):
        """停止处理"""
        self.running = False

def mic_audio_generator(buffer_size):
    """生成器函数：从麦克风获取音频块"""
    # 计算缓冲区大小
    buffer_samples = int(buffer_size * TARGET_SAMPLE_RATE)
    buffer_queue = queue.Queue()
    
    def audio_callback(indata, frames, time, status):
        """sounddevice回调函数"""
        if status:
            print(f"音频输入状态: {status}")
        # 将输入数据放入队列
        buffer_queue.put(indata.copy())
    
    # 启动音频流
    stream = sd.InputStream(
        samplerate=TARGET_SAMPLE_RATE,
        blocksize=buffer_samples,
        channels=1,
        callback=audio_callback,
        dtype='float32'
    )
    
    with stream:
        print("麦克风已启动，开始监听...")
        try:
            while True:
                # 从队列获取音频数据
                indata = buffer_queue.get()
                # 返回单声道数据
                yield indata.flatten()
        except KeyboardInterrupt:
            pass
        finally:
            print("麦克风已关闭")

def file_audio_generator(file_path, buffer_size):
    """生成器函数：模拟流式加载音频文件"""
    # 加载音频文件
    audio, sr = librosa.load(file_path, sr=TARGET_SAMPLE_RATE)
    
    # 设置缓冲区大小
    buffer_samples = int(buffer_size * TARGET_SAMPLE_RATE)
    
    # 计算块数
    num_chunks = int(np.ceil(len(audio) / buffer_samples))
    
    print(f"音频长度: {len(audio)/TARGET_SAMPLE_RATE:.2f}秒, 分为{num_chunks}个块")
    
    # 逐块产生数据
    for i in range(num_chunks):
        start_idx = i * buffer_samples
        end_idx = min(start_idx + buffer_samples, len(audio))
        chunk = audio[start_idx:end_idx]
        
        # 模拟实时性，添加短暂延迟
        time.sleep(buffer_size)
        
        yield chunk

def main():
    """主函数"""
    args = parse_args()
    
    # 确保参数有效
    if not args.use_mic and args.audio_file is None:
        print("错误: 必须指定音频文件或使用麦克风")
        return
    
    # 加载模型
    print("加载模型...")
    model, device, intent_labels = load_model(args.model_path)
    
    # 创建流处理器
    processor = RealTimeStreamingProcessor(model, device, intent_labels, args.buffer_size)
    
    # 选择音频源
    if args.use_mic:
        print("使用麦克风输入...")
        audio_source = mic_audio_generator(args.buffer_size)
    else:
        print(f"从文件加载音频: {args.audio_file}")
        audio_source = file_audio_generator(args.audio_file, args.buffer_size)
    
    # 开始处理
    print("开始实时处理，按Ctrl+C停止...")
    try:
        processor.start_realtime_processing(audio_source)
        
        # 主循环中处理结果
        while True:
            try:
                if not processor.result_queue.empty():
                    result = processor.result_queue.get(block=False)
                    intent = result["intent"]
                    confidence = result["confidence"]
                    print(f"已识别命令: {intent} (置信度: {confidence:.2f})")
                    
                    # 执行相应操作（示例）
                    if intent == "TAKE_PHOTO":
                        print("执行操作: 拍照")
                    elif intent == "START_RECORDING":
                        print("执行操作: 开始录像")
                    elif intent == "STOP_RECORDING":
                        print("执行操作: 停止录像")
                
                time.sleep(0.1)  # 减少CPU使用
            except queue.Empty:
                pass
                
    except KeyboardInterrupt:
        print("停止处理...")
        processor.stop()
        
    print("程序结束")

if __name__ == "__main__":
    main() 