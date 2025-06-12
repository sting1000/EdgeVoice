#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
实时流式语音识别推理系统
支持部署优化的Conformer模型，满足limits.md约束
"""

import os
import time
import threading
import queue
import numpy as np
import torch
import torch.nn.functional as F
import sounddevice as sd
import argparse
from collections import deque

from config import *
from models.streaming_conformer import StreamingConformer
from utils.feature_extraction import streaming_feature_extractor

class StreamingInferenceEngine:
    """流式推理引擎，支持实时音频处理"""
    
    def __init__(self, model_path, device='cpu', confidence_threshold=0.8,
                 min_confidence_frames=3, sample_rate=TARGET_SAMPLE_RATE):
        """
        初始化推理引擎
        
        Args:
            model_path: 模型文件路径
            device: 计算设备 ('cpu', 'cuda')
            confidence_threshold: 置信度阈值
            min_confidence_frames: 最小连续高置信度帧数
            sample_rate: 音频采样率
        """
        self.device = torch.device(device)
        self.confidence_threshold = confidence_threshold
        self.min_confidence_frames = min_confidence_frames
        self.sample_rate = sample_rate
        
        # 加载模型
        self.model = self._load_model(model_path)
        print(f"模型加载完成，使用设备: {self.device}")
        
        # 初始化流式状态
        self.reset_state()
        
        # 音频缓冲区
        self.audio_buffer = deque(maxlen=int(sample_rate * 2))  # 2秒缓冲
        self.is_recording = False
        
        # 结果平滑
        self.recent_predictions = deque(maxlen=self.min_confidence_frames)
        
    def _load_model(self, model_path):
        """加载训练好的模型"""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"模型文件不存在: {model_path}")
            
        # 创建模型实例 - 使用部署优化的参数
        model = StreamingConformer(
            input_dim=N_MFCC * (2 * CONTEXT_FRAMES + 1),  # MFCC + 上下文
            hidden_dim=CONFORMER_HIDDEN_SIZE,
            num_classes=len(INTENT_CLASSES),
            num_layers=CONFORMER_LAYERS,
            num_heads=CONFORMER_ATTENTION_HEADS,
            dropout=0.0,  # 推理时不使用dropout
            kernel_size=CONFORMER_CONV_KERNEL_SIZE,
            expansion_factor=CONFORMER_FF_EXPANSION_FACTOR
        )
        
        # 加载权重
        checkpoint = torch.load(model_path, map_location=self.device)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
            
        model.to(self.device)
        model.eval()
        
        return model
    
    def reset_state(self):
        """重置流式推理状态"""
        self.model.reset_streaming_state()
        self.cached_states = None
        self.recent_predictions.clear()
        print("流式状态已重置")
        
    def preprocess_audio_chunk(self, audio_chunk):
        """
        预处理音频块，提取特征
        
        Args:
            audio_chunk: 音频数据 (numpy数组)
            
        Returns:
            features: 特征张量 [1, seq_len, feature_dim]
        """
        try:
            # 使用流式特征提取器
            features = streaming_feature_extractor(
                audio_chunk,
                sr=self.sample_rate,
                chunk_size=STREAMING_CHUNK_SIZE,
                step_size=STREAMING_STEP_SIZE
            )
            
            if features is None or len(features) == 0:
                return None
                
            # 转换为张量并添加batch维度
            features_tensor = torch.FloatTensor(features).unsqueeze(0).to(self.device)
            
            return features_tensor
            
        except Exception as e:
            print(f"特征提取失败: {e}")
            return None
    
    def predict_streaming(self, features):
        """
        执行流式预测
        
        Args:
            features: 输入特征 [1, seq_len, feature_dim]
            
        Returns:
            prediction: 预测类别
            confidence: 置信度
            intent_name: 意图名称
        """
        try:
            with torch.no_grad():
                # 流式推理
                pred, conf, new_cached_states = self.model.predict_streaming(
                    features, self.cached_states
                )
                
                # 更新缓存状态
                self.cached_states = new_cached_states
                
                # 获取预测结果
                prediction = pred.item()
                confidence = conf.item()
                intent_name = INTENT_CLASSES[prediction]
                
                return prediction, confidence, intent_name
                
        except Exception as e:
            print(f"推理失败: {e}")
            return -1, 0.0, "ERROR"
    
    def smooth_predictions(self, prediction, confidence, intent_name):
        """
        平滑预测结果，避免抖动
        
        Args:
            prediction: 当前预测
            confidence: 当前置信度
            intent_name: 当前意图名称
            
        Returns:
            final_prediction: 最终预测结果 (None如果还未稳定)
        """
        # 只考虑高置信度的预测
        if confidence >= self.confidence_threshold:
            self.recent_predictions.append((prediction, confidence, intent_name))
        else:
            self.recent_predictions.append(None)
        
        # 检查是否有连续的高置信度预测
        if len(self.recent_predictions) >= self.min_confidence_frames:
            # 检查最近的预测是否一致且高置信度
            recent_valid = [p for p in self.recent_predictions if p is not None]
            
            if len(recent_valid) >= self.min_confidence_frames:
                # 检查预测一致性
                recent_preds = [p[0] for p in recent_valid[-self.min_confidence_frames:]]
                if len(set(recent_preds)) == 1:  # 所有预测都相同
                    final_pred = recent_preds[0]
                    avg_conf = np.mean([p[1] for p in recent_valid[-self.min_confidence_frames:]])
                    intent = recent_valid[-1][2]
                    return final_pred, avg_conf, intent
        
        return None
    
    def process_audio_chunk(self, audio_chunk):
        """
        处理单个音频块
        
        Args:
            audio_chunk: 音频数据
            
        Returns:
            result: 识别结果字典或None
        """
        # 预处理
        features = self.preprocess_audio_chunk(audio_chunk)
        if features is None:
            return None
            
        # 推理
        prediction, confidence, intent_name = self.predict_streaming(features)
        if prediction == -1:
            return None
            
        # 平滑
        smoothed_result = self.smooth_predictions(prediction, confidence, intent_name)
        
        if smoothed_result is not None:
            final_pred, avg_conf, final_intent = smoothed_result
            return {
                'intent': final_intent,
                'confidence': avg_conf,
                'timestamp': time.time(),
                'prediction_id': final_pred
            }
        
        return None

class RealTimeAudioProcessor:
    """实时音频处理器"""
    
    def __init__(self, inference_engine, chunk_duration=0.1):
        """
        初始化音频处理器
        
        Args:
            inference_engine: 推理引擎实例
            chunk_duration: 音频块持续时间(秒)
        """
        self.inference_engine = inference_engine
        self.chunk_duration = chunk_duration
        self.chunk_size = int(TARGET_SAMPLE_RATE * chunk_duration)
        
        self.audio_queue = queue.Queue()
        self.result_queue = queue.Queue()
        self.is_running = False
        
        self.audio_buffer = np.array([])
        
    def audio_callback(self, indata, frames, time, status):
        """音频输入回调函数"""
        if status:
            print(f"音频输入状态: {status}")
            
        # 将音频数据加入队列
        audio_data = indata[:, 0]  # 取单声道
        self.audio_queue.put(audio_data.copy())
    
    def start_recording(self):
        """开始录音"""
        self.is_running = True
        
        # 启动音频流
        self.audio_stream = sd.InputStream(
            samplerate=TARGET_SAMPLE_RATE,
            channels=1,
            dtype=np.float32,
            blocksize=self.chunk_size,
            callback=self.audio_callback
        )
        
        # 启动处理线程
        self.processing_thread = threading.Thread(target=self._process_audio_loop)
        self.processing_thread.daemon = True
        
        self.audio_stream.start()
        self.processing_thread.start()
        
        print(f"开始实时录音，采样率: {TARGET_SAMPLE_RATE}Hz")
        print(f"音频块大小: {self.chunk_size}样本 ({self.chunk_duration*1000:.1f}ms)")
        
    def stop_recording(self):
        """停止录音"""
        self.is_running = False
        
        if hasattr(self, 'audio_stream'):
            self.audio_stream.stop()
            self.audio_stream.close()
            
        if hasattr(self, 'processing_thread'):
            self.processing_thread.join(timeout=1.0)
        
        print("录音已停止")
    
    def _process_audio_loop(self):
        """音频处理主循环"""
        while self.is_running:
            try:
                # 获取音频数据 (非阻塞，超时0.1秒)
                audio_chunk = self.audio_queue.get(timeout=0.1)
                
                # 累积音频数据
                self.audio_buffer = np.concatenate([self.audio_buffer, audio_chunk])
                
                # 当累积足够的数据时进行处理
                process_size = int(TARGET_SAMPLE_RATE * 0.5)  # 处理0.5秒的数据
                
                if len(self.audio_buffer) >= process_size:
                    # 处理音频
                    result = self.inference_engine.process_audio_chunk(
                        self.audio_buffer[:process_size]
                    )
                    
                    # 移除已处理的数据，保留重叠部分
                    overlap_size = int(TARGET_SAMPLE_RATE * 0.2)  # 保留0.2秒重叠
                    self.audio_buffer = self.audio_buffer[process_size-overlap_size:]
                    
                    # 如果有结果，加入结果队列
                    if result is not None:
                        self.result_queue.put(result)
                        
            except queue.Empty:
                continue
            except Exception as e:
                print(f"音频处理错误: {e}")
    
    def get_result(self):
        """获取识别结果 (非阻塞)"""
        try:
            return self.result_queue.get_nowait()
        except queue.Empty:
            return None

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='实时流式语音识别')
    parser.add_argument('--model', '-m', required=True, help='模型文件路径')
    parser.add_argument('--device', '-d', default='cpu', choices=['cpu', 'cuda'], help='计算设备')
    parser.add_argument('--confidence', '-c', type=float, default=0.8, help='置信度阈值')
    parser.add_argument('--frames', '-f', type=int, default=3, help='最小连续高置信度帧数')
    parser.add_argument('--duration', '-t', type=int, default=30, help='录音时长(秒，0为无限)')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("EdgeVoice 实时流式语音识别系统")
    print("=" * 60)
    print(f"模型文件: {args.model}")
    print(f"计算设备: {args.device}")
    print(f"置信度阈值: {args.confidence}")
    print(f"最小连续帧数: {args.frames}")
    print(f"意图类别: {', '.join(INTENT_CLASSES)}")
    print("=" * 60)
    
    try:
        # 初始化推理引擎
        print("正在加载模型...")
        inference_engine = StreamingInferenceEngine(
            model_path=args.model,
            device=args.device,
            confidence_threshold=args.confidence,
            min_confidence_frames=args.frames
        )
        
        # 初始化音频处理器
        audio_processor = RealTimeAudioProcessor(inference_engine)
        
        # 开始录音
        audio_processor.start_recording()
        
        print("\n🎤 开始实时语音识别...")
        print("说出以下命令之一:")
        for i, intent in enumerate(INTENT_CLASSES):
            print(f"  {i+1}. {intent}")
        print("\n按 Ctrl+C 停止\n")
        
        start_time = time.time()
        
        # 主循环
        while True:
            # 检查录音时长
            if args.duration > 0 and time.time() - start_time > args.duration:
                print(f"\n录音时长达到 {args.duration} 秒，自动停止")
                break
                
            # 获取识别结果
            result = audio_processor.get_result()
            if result is not None:
                timestamp = time.strftime("%H:%M:%S", time.localtime(result['timestamp']))
                print(f"[{timestamp}] 🎯 识别结果: {result['intent']} "
                      f"(置信度: {result['confidence']:.3f})")
                
                # 如果是停止录音命令，自动退出
                if result['intent'] == 'STOP_RECORDING':
                    print("检测到停止录音命令，正在退出...")
                    break
            
            time.sleep(0.05)  # 50ms检查间隔
            
    except KeyboardInterrupt:
        print("\n\n用户中断，正在退出...")
    except Exception as e:
        print(f"\n错误: {e}")
    finally:
        # 清理资源
        if 'audio_processor' in locals():
            audio_processor.stop_recording()
        print("程序已退出")

if __name__ == "__main__":
    main() 