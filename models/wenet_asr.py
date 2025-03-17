#!/usr/bin/env python
# -*- coding: utf-8 -*-
# models/wenet_asr.py

"""
WeNet ASR模型封装类，提供模型加载、音频转写和ONNX导出功能。
"""

import os
import time
import json
import logging
import numpy as np
import torch
import torchaudio
from torch.nn.utils.rnn import pad_sequence
import threading
import multiprocessing
from contextlib import contextmanager

# 引入工具函数
from utils.asr_utils import wav_to_fbank, normalize_fbank, save_asr_result
from config import *

# 设置日志
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("WeNetASR")

# 设置转写超时（秒）
TRANSCRIBE_TIMEOUT = PRECISE_TOTAL_TIMEOUT_MS / 1000.0

class WeNetASR:
    """WeNet ASR模型封装类"""
    
    def __init__(self, model_path, dict_path, device=None, save_results=False, result_dir=None):
        """初始化Wenet ASR模型
        
        Args:
            model_path: Wenet模型路径，可以是TorchScript或ONNX模型
            dict_path: 字典文件路径
            device: 运行设备(cuda/cpu)
            save_results: 是否保存ASR中间结果
            result_dir: 结果保存目录
        """
        # 初始化参数
        self.device = device if device else DEVICE
        self.model_path = model_path
        self.dict_path = dict_path
        self.save_results = save_results
        self.result_dir = result_dir if result_dir else ASR_CACHE_DIR
        if self.save_results:
            os.makedirs(self.result_dir, exist_ok=True)
        
        # 加载词典
        self.load_dictionary(dict_path)
        
        # 加载模型
        self.load_model(model_path)
        
        # 特征提取配置
        self.feature_config = {
            'n_mels': 80,
            'frame_length': 25,
            'frame_shift': 10
        }
        
        # 转写结果缓存
        self.result_cache = {}
        self.cache_lock = threading.Lock()
    
    def load_dictionary(self, dict_path):
        """加载字典文件
        
        Args:
            dict_path: 字典文件路径
        """
        try:
            logger.info(f"加载字典文件: {dict_path}")
            
            # 字典映射: 索引->字符
            self.id2char = {}
            # 字典映射: 字符->索引
            self.char2id = {}
            
            with open(dict_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        parts = line.split()
                        if len(parts) == 2:
                            char, idx = parts
                            idx = int(idx)
                            self.id2char[idx] = char
                            self.char2id[char] = idx
            
            logger.info(f"字典加载完成，共{len(self.id2char)}个字符")
            
            # 特殊标记
            self.blank_id = 0  # 默认blank ID为0
            self.unk_id = self.char2id.get('<unk>', 1)
            self.sos_eos_id = self.char2id.get('<sos/eos>', len(self.id2char)-1)
            
        except Exception as e:
            logger.error(f"加载字典文件失败: {e}")
            raise
    
    def load_model(self, model_path):
        """加载Wenet模型
        
        Args:
            model_path: 模型路径
        """
        try:
            logger.info(f"加载ASR模型: {model_path}")
            
            # 判断模型类型
            if model_path.endswith('.onnx'):
                self._load_onnx_model(model_path)
            else:
                self._load_torch_model(model_path)
            
            logger.info("ASR模型加载完成")
        except Exception as e:
            logger.error(f"加载ASR模型失败: {e}")
            raise
    
    def _load_torch_model(self, model_path):
        """加载TorchScript模型
        
        Args:
            model_path: TorchScript模型路径
        """
        try:
            # 尝试引入wenet相关模块
            try:
                import wenet
                from wenet.transformer.asr_model import init_asr_model
                from wenet.utils.checkpoint import load_checkpoint
                from wenet.utils.config import override_config
                
                # 加载配置文件
                config_path = os.path.join(os.path.dirname(model_path), 'train.yaml')
                with open(config_path, 'r') as fin:
                    configs = yaml.load(fin, Loader=yaml.FullLoader)
                
                # 创建模型
                model = init_asr_model(configs)
                load_checkpoint(model, model_path)
                model.to(self.device)
                model.eval()
                self.model = model
                self.model_type = 'torch'
                logger.info("成功加载PyTorch模型")
            except ImportError:
                # 如果没有安装wenet，尝试加载TorchScript模型
                logger.info("未安装wenet模块，尝试加载TorchScript模型")
                self.model = torch.jit.load(model_path, map_location=self.device)
                self.model.eval()
                self.model_type = 'torchscript'
                logger.info("成功加载TorchScript模型")
                
        except Exception as e:
            logger.error(f"加载TorchScript模型失败: {e}")
            # 尝试使用wenetruntime加载
            try:
                import wenetruntime
                logger.info("尝试使用wenetruntime加载模型")
                self.model = wenetruntime.Decoder(model_path)
                self.model_type = 'wenetruntime'
                logger.info("成功使用wenetruntime加载模型")
            except:
                raise Exception(f"无法加载Wenet模型，请确保安装了wenet或wenetruntime: {e}")
    
    def _load_onnx_model(self, model_path):
        """加载ONNX模型
        
        Args:
            model_path: ONNX模型路径
        """
        try:
            import onnxruntime as ort
            
            logger.info(f"加载ONNX模型: {model_path}")
            
            # 创建ONNX会话
            providers = ['CUDAExecutionProvider'] if torch.cuda.is_available() else ['CPUExecutionProvider']
            session_options = ort.SessionOptions()
            session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            
            # 尝试加载模型
            self.model = ort.InferenceSession(
                model_path, 
                sess_options=session_options,
                providers=providers
            )
            self.model_type = 'onnx'
            
            # 获取输入输出信息
            self.model_inputs = [input.name for input in self.model.get_inputs()]
            self.model_outputs = [output.name for output in self.model.get_outputs()]
            
            logger.info(f"ONNX模型加载完成，输入: {self.model_inputs}，输出: {self.model_outputs}")
            
        except Exception as e:
            logger.error(f"加载ONNX模型失败: {e}")
            raise
    
    def extract_features(self, audio, sample_rate=16000):
        """从音频中提取特征
        
        Args:
            audio: 音频数据
            sample_rate: 采样率
            
        Returns:
            特征张量
        """
        # 确保音频是numpy数组
        if isinstance(audio, torch.Tensor):
            audio = audio.numpy()
        
        # 提取Fbank特征
        fbank = wav_to_fbank(
            audio, 
            sample_rate, 
            n_mels=self.feature_config['n_mels'], 
            frame_length=self.feature_config['frame_length'],
            frame_shift=self.feature_config['frame_shift']
        )
        
        # 规范化特征
        fbank, _, _ = normalize_fbank(fbank)
        
        # 转换为tensor
        fbank_tensor = torch.from_numpy(fbank).float()
        
        # 增加批次维度
        fbank_tensor = fbank_tensor.unsqueeze(0)  # [1, T, D]
        
        return fbank_tensor
    
    @contextmanager
    def _timeout(self, seconds):
        """超时控制上下文管理器
        
        Args:
            seconds: 超时时间（秒）
        """
        def timeout_handler(signum, frame):
            raise TimeoutError(f"转写超时（{seconds}秒）")
        
        # 仅在非Windows平台使用信号
        if hasattr(multiprocessing, 'current_process') and multiprocessing.current_process().name == 'MainProcess':
            import signal
            original_handler = signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(int(seconds))
            try:
                yield
            finally:
                signal.alarm(0)
                signal.signal(signal.SIGALRM, original_handler)
        else:
            try:
                yield
            finally:
                pass
    
    def _decode_output(self, output):
        """解码模型输出为文本
        
        Args:
            output: 模型输出（ID序列）
            
        Returns:
            解码后的文本
        """
        result = []
        for idx in output:
            if idx == self.blank_id:
                continue
            if idx == self.sos_eos_id:
                break
            text = self.id2char.get(idx, '<unk>')
            if text != '<unk>':
                result.append(text)
        
        return ''.join(result)
    
    def transcribe(self, audio, sample_rate=16000):
        """将音频转写为文本
        
        Args:
            audio: 音频数据
            sample_rate: 采样率
            
        Returns:
            (文本结果, 置信度)
        """
        try:
            # 检查缓存，如果有相同音频的转写结果则直接返回
            audio_hash = hash(audio.tobytes()) if hasattr(audio, 'tobytes') else hash(str(audio))
            
            with self.cache_lock:
                if audio_hash in self.result_cache:
                    logger.info("使用缓存的转写结果")
                    return self.result_cache[audio_hash]
            
            # 设置超时控制
            with self._timeout(TRANSCRIBE_TIMEOUT):
                # 根据模型类型选择转写方法
                if self.model_type == 'wenetruntime':
                    # 对于wenetruntime，直接使用其API
                    audio_bytes = audio.astype(np.float32).tobytes()
                    start_time = time.time()
                    text = self.model.decode(audio_bytes, sample_rate, True)
                    end_time = time.time()
                    logger.info(f"转写完成，耗时: {(end_time - start_time) * 1000:.2f}ms")
                    
                    # 由于wenetruntime不提供置信度，我们使用一个固定值
                    confidence = 0.9
                    
                elif self.model_type == 'onnx':
                    # 对于ONNX模型，提取特征并进行推理
                    features = self.extract_features(audio, sample_rate)
                    
                    # 将特征转换为ONNX输入格式
                    inputs = {
                        self.model_inputs[0]: features.numpy()
                    }
                    
                    # 运行推理
                    start_time = time.time()
                    outputs = self.model.run(self.model_outputs, inputs)
                    end_time = time.time()
                    
                    # 假设输出是logits和长度
                    logits = outputs[0]
                    
                    # 解码
                    output_ids = np.argmax(logits, axis=-1)[0]
                    text = self._decode_output(output_ids)
                    
                    # 计算置信度（使用softmax概率平均值作为近似）
                    probs = np.exp(logits) / np.sum(np.exp(logits), axis=-1, keepdims=True)
                    max_probs = np.max(probs, axis=-1)
                    confidence = float(np.mean(max_probs))
                    
                    logger.info(f"ONNX转写完成，耗时: {(end_time - start_time) * 1000:.2f}ms")
                    
                else:
                    # 对于TorchScript或PyTorch模型
                    features = self.extract_features(audio, sample_rate)
                    features = features.to(self.device)
                    
                    with torch.no_grad():
                        start_time = time.time()
                        # 针对torchscript模型
                        if self.model_type == 'torchscript':
                            outputs = self.model(features)
                            logits = outputs[0]
                            
                            # 解码
                            _, output_ids = torch.max(logits, dim=-1)
                            output_ids = output_ids[0].cpu().numpy()
                            text = self._decode_output(output_ids)
                            
                            # 计算置信度
                            probs = torch.softmax(logits, dim=-1)
                            max_probs = torch.max(probs, dim=-1)[0]
                            confidence = float(torch.mean(max_probs).item())
                            
                        # 针对torch模型
                        else:
                            # 假设是wenet的torch模型
                            feature_length = torch.IntTensor([features.size(1)])
                            outputs = self.model(features, feature_length)
                            
                            # 假设输出包含encoder_out和encoder_mask
                            if isinstance(outputs, tuple) and len(outputs) >= 2:
                                encoder_out, encoder_mask = outputs[0], outputs[1]
                                
                                # 使用CTC解码
                                ctc_probs = torch.nn.functional.log_softmax(encoder_out, dim=-1)
                                
                                # CTC贪心解码
                                _, output_ids = torch.max(ctc_probs, dim=-1)
                                output_ids = output_ids[0].cpu().numpy()
                                text = self._decode_output(output_ids)
                                
                                # 计算置信度
                                probs = torch.exp(ctc_probs)
                                max_probs = torch.max(probs, dim=-1)[0]
                                confidence = float(torch.mean(max_probs).item())
                            else:
                                # 假设输出是logits
                                logits = outputs
                                _, output_ids = torch.max(logits, dim=-1)
                                output_ids = output_ids[0].cpu().numpy()
                                text = self._decode_output(output_ids)
                                
                                # 计算置信度
                                probs = torch.softmax(logits, dim=-1)
                                max_probs = torch.max(probs, dim=-1)[0]
                                confidence = float(torch.mean(max_probs).item())
                        
                        end_time = time.time()
                        logger.info(f"转写完成，耗时: {(end_time - start_time) * 1000:.2f}ms")
            
            # 缓存结果
            result = (text, confidence)
            with self.cache_lock:
                self.result_cache[audio_hash] = result
            
            return result
                
        except TimeoutError as e:
            logger.warning(f"转写超时: {e}")
            return "转写超时", 0.0
        
        except Exception as e:
            logger.error(f"转写失败: {e}")
            return "转写失败", 0.0
    
    def save_result(self, audio_id, text, confidence, metadata=None):
        """保存ASR结果
        
        Args:
            audio_id: 音频ID或时间戳
            text: 转写文本
            confidence: 置信度
            metadata: 额外元数据信息，如时间戳等
        """
        if not self.save_results:
            return None
        
        # 使用工具函数保存结果
        return save_asr_result(self.result_dir, audio_id, text, confidence, metadata)
    
    def export_onnx(self, onnx_path, sample_audio=None, opset_version=13):
        """导出ONNX模型
        
        Args:
            onnx_path: ONNX模型保存路径
            sample_audio: 示例音频，用于生成动态输入形状
            opset_version: ONNX操作集版本
        """
        try:
            logger.info(f"正在将模型导出为ONNX格式: {onnx_path}")
            
            if self.model_type == 'onnx':
                logger.warning("模型已经是ONNX格式，无需再次导出")
                return
            
            if self.model_type == 'wenetruntime':
                raise ValueError("wenetruntime模型不支持导出为ONNX")
            
            # 创建导出目录
            os.makedirs(os.path.dirname(onnx_path), exist_ok=True)
            
            # 生成示例输入
            if sample_audio is None:
                # 生成一个随机的音频示例
                dummy_audio = np.random.randn(16000 * 5)  # 5秒音频
                features = self.extract_features(dummy_audio)
            else:
                features = self.extract_features(sample_audio)
            
            # 移动到设备
            features = features.to(self.device)
            
            # 动态轴
            dynamic_axes = {
                'input': {0: 'batch_size', 1: 'sequence_length'},
                'output': {0: 'batch_size', 1: 'sequence_length'}
            }
            
            # 导出模型
            if self.model_type == 'torchscript':
                # 设置输入输出名称
                input_names = ['input']
                output_names = ['output']
                
                # 导出
                torch.onnx.export(
                    self.model,
                    features,
                    onnx_path,
                    export_params=True,
                    opset_version=opset_version,
                    do_constant_folding=True,
                    input_names=input_names,
                    output_names=output_names,
                    dynamic_axes=dynamic_axes,
                    verbose=False
                )
            else:
                # 对于PyTorch模型，需要更复杂的处理
                # 这里仅做示例，实际应根据模型结构调整
                class ONNXWrapper(torch.nn.Module):
                    def __init__(self, model):
                        super().__init__()
                        self.model = model
                    
                    def forward(self, x):
                        feature_length = torch.IntTensor([x.size(1)])
                        outputs = self.model(x, feature_length)
                        if isinstance(outputs, tuple) and len(outputs) >= 2:
                            return outputs[0]  # 返回encoder_out
                        return outputs
                
                # 包装模型
                wrapper = ONNXWrapper(self.model)
                
                # 设置输入输出名称
                input_names = ['input']
                output_names = ['output']
                
                # 导出
                torch.onnx.export(
                    wrapper,
                    features,
                    onnx_path,
                    export_params=True,
                    opset_version=opset_version,
                    do_constant_folding=True,
                    input_names=input_names,
                    output_names=output_names,
                    dynamic_axes=dynamic_axes,
                    verbose=False
                )
            
            logger.info(f"ONNX模型导出成功: {onnx_path}")
            
            # 验证模型
            import onnx
            model = onnx.load(onnx_path)
            onnx.checker.check_model(model)
            logger.info("ONNX模型验证通过")
            
            return onnx_path
        
        except Exception as e:
            logger.error(f"导出ONNX模型失败: {e}")
            raise 