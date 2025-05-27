#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
渐进式流式训练器
实现EdgeVoice项目的渐进式流式训练策略
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional
import random

from config import *

class ProgressiveStreamingTrainer:
    """渐进式流式训练器"""
    
    def __init__(self, 
                 chunk_size: int = STREAMING_CHUNK_SIZE,
                 step_size: int = STREAMING_STEP_SIZE,
                 schedule: Dict = STREAMING_TRAINING_SCHEDULE):
        """
        初始化渐进式流式训练器
        
        Args:
            chunk_size: 流式处理的chunk大小
            step_size: 流式处理的步长
            schedule: 训练调度配置
        """
        self.chunk_size = chunk_size
        self.step_size = step_size
        self.schedule = schedule
        
    def get_streaming_ratio(self, epoch: int) -> float:
        """
        根据epoch获取流式训练比例
        
        Args:
            epoch: 当前epoch
            
        Returns:
            streaming_ratio: 流式训练比例 (0.0-1.0)
        """
        for phase_name, phase_config in self.schedule.items():
            start_epoch, end_epoch = phase_config['epochs']
            if start_epoch <= epoch <= end_epoch:
                return phase_config['streaming_ratio']
        
        # 如果超出定义范围，使用最后一个阶段的比例
        last_phase = list(self.schedule.values())[-1]
        return last_phase['streaming_ratio']
    
    def should_use_streaming(self, epoch: int, batch_idx: int = None) -> bool:
        """
        判断当前batch是否应该使用流式训练
        
        Args:
            epoch: 当前epoch
            batch_idx: 当前batch索引（可选，用于随机选择）
            
        Returns:
            bool: 是否使用流式训练
        """
        streaming_ratio = self.get_streaming_ratio(epoch)
        
        if streaming_ratio == 0.0:
            return False
        elif streaming_ratio == 1.0:
            return True
        else:
            # 随机决定是否使用流式训练
            return random.random() < streaming_ratio
    
    def split_sequence_to_chunks(self, features: torch.Tensor) -> List[torch.Tensor]:
        """
        将完整序列分割为chunks
        
        Args:
            features: 输入特征 [batch_size, seq_len, feature_dim]
            
        Returns:
            chunks: chunk列表，每个chunk为 [batch_size, chunk_size, feature_dim]
        """
        batch_size, seq_len, feature_dim = features.shape
        chunks = []
        
        # 如果序列长度小于chunk_size，直接返回原序列
        if seq_len <= self.chunk_size:
            return [features]
        
        # 分割序列
        start = 0
        while start < seq_len:
            end = min(start + self.chunk_size, seq_len)
            chunk = features[:, start:end, :]
            
            # 如果chunk太小，进行padding或跳过
            if chunk.shape[1] < self.chunk_size // 2:
                break
                
            chunks.append(chunk)
            start += self.step_size
            
            # 如果已经覆盖到序列末尾，停止
            if end >= seq_len:
                break
        
        return chunks
    
    def streaming_forward_pass(self, 
                             model: nn.Module, 
                             features: torch.Tensor, 
                             device: torch.device) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        执行流式前向传播
        
        Args:
            model: 模型
            features: 输入特征 [batch_size, seq_len, feature_dim]
            device: 设备
            
        Returns:
            final_output: 最终预测输出 [batch_size, num_classes]
            all_outputs: 所有chunk的输出列表
        """
        # 分割序列
        chunks = self.split_sequence_to_chunks(features)
        
        if not chunks:
            # 如果没有有效chunks，使用完整序列
            return model(features), [model(features)]
        
        # 重置模型的流式状态
        model.reset_streaming_state()
        cached_states = None
        
        all_outputs = []
        
        # 逐chunk处理
        for chunk in chunks:
            chunk = chunk.to(device)
            
            # 使用模型的流式预测方法
            with torch.no_grad():
                # 获取流式预测结果
                pred, conf, cached_states = model.predict_streaming(chunk, cached_states)
                
                # 重新启用梯度计算，获取完整输出
                chunk.requires_grad_(True)
                
            # 重新进行前向传播以获取梯度
            if hasattr(model, 'streaming_forward'):
                output = model.streaming_forward(chunk, cached_states)
            else:
                # 如果没有专门的streaming_forward，使用标准forward
                output = model(chunk)
            
            all_outputs.append(output)
        
        # 返回最终输出（最后一个chunk的输出）
        final_output = all_outputs[-1] if all_outputs else model(features)
        
        return final_output, all_outputs


class FinalPredictionLoss(nn.Module):
    """最终预测损失函数"""
    
    def __init__(self, 
                 base_criterion: nn.Module = None,
                 stability_weight: float = STABILITY_LOSS_WEIGHT):
        """
        初始化最终预测损失
        
        Args:
            base_criterion: 基础损失函数
            stability_weight: 稳定性损失权重
        """
        super().__init__()
        self.base_criterion = base_criterion or nn.CrossEntropyLoss()
        self.stability_weight = stability_weight
        
    def forward(self, 
                final_output: torch.Tensor, 
                labels: torch.Tensor,
                all_outputs: Optional[List[torch.Tensor]] = None) -> torch.Tensor:
        """
        计算最终预测损失
        
        Args:
            final_output: 最终预测输出 [batch_size, num_classes]
            labels: 真实标签 [batch_size]
            all_outputs: 所有chunk的输出（可选，用于稳定性损失）
            
        Returns:
            loss: 总损失
        """
        # 主要损失：最终预测损失
        main_loss = self.base_criterion(final_output, labels)
        
        # 可选：稳定性损失（减少预测跳变）
        stability_loss = 0.0
        if all_outputs and len(all_outputs) > 1 and self.stability_weight > 0:
            stability_loss = self._compute_stability_loss(all_outputs)
        
        total_loss = main_loss + self.stability_weight * stability_loss
        
        return total_loss
    
    def _compute_stability_loss(self, all_outputs: List[torch.Tensor]) -> torch.Tensor:
        """
        计算稳定性损失（减少预测跳变）
        
        Args:
            all_outputs: 所有chunk的输出列表
            
        Returns:
            stability_loss: 稳定性损失
        """
        if len(all_outputs) < 2:
            return torch.tensor(0.0, device=all_outputs[0].device)
        
        stability_loss = 0.0
        
        # 计算相邻chunk预测的差异
        for i in range(1, len(all_outputs)):
            prev_probs = torch.softmax(all_outputs[i-1], dim=-1)
            curr_probs = torch.softmax(all_outputs[i], dim=-1)
            
            # 使用KL散度衡量预测差异
            kl_div = torch.nn.functional.kl_div(
                torch.log(curr_probs + 1e-8), 
                prev_probs, 
                reduction='batchmean'
            )
            stability_loss += kl_div
        
        # 平均稳定性损失
        stability_loss = stability_loss / (len(all_outputs) - 1)
        
        return stability_loss


class EdgeVoiceMetrics:
    """EdgeVoice特定的评估指标"""
    
    def __init__(self, core_commands: List[str] = CORE_COMMANDS):
        """
        初始化EdgeVoice评估指标
        
        Args:
            core_commands: 核心指令列表
        """
        self.core_commands = core_commands
        
    def calculate_top1_accuracy(self, 
                               predictions: List[int], 
                               labels: List[int],
                               intent_labels: List[str]) -> Dict[str, float]:
        """
        计算Top1准确率
        
        Args:
            predictions: 预测结果列表
            labels: 真实标签列表
            intent_labels: 意图标签名称列表
            
        Returns:
            accuracy_dict: 包含总体和核心指令准确率的字典
        """
        # 总体准确率
        total_correct = sum(1 for p, l in zip(predictions, labels) if p == l)
        total_accuracy = total_correct / len(predictions) if predictions else 0.0
        
        # 核心指令准确率
        core_correct = 0
        core_total = 0
        
        for pred, label in zip(predictions, labels):
            true_intent = intent_labels[label] if label < len(intent_labels) else "UNKNOWN"
            if true_intent in self.core_commands:
                core_total += 1
                if pred == label:
                    core_correct += 1
        
        core_accuracy = core_correct / core_total if core_total > 0 else 0.0
        
        return {
            'total_accuracy': total_accuracy,
            'core_accuracy': core_accuracy,
            'core_samples': core_total,
            'total_samples': len(predictions)
        }
    
    def calculate_stability_score(self, prediction_sequences: List[List[int]]) -> Dict[str, float]:
        """
        计算预测稳定性评分
        
        Args:
            prediction_sequences: 每个样本的预测序列列表
            
        Returns:
            stability_dict: 稳定性指标字典
        """
        if not prediction_sequences:
            return {'stability_score': 0.0, 'avg_changes': 0.0}
        
        total_changes = 0
        total_sequences = len(prediction_sequences)
        
        for seq in prediction_sequences:
            if len(seq) > 1:
                changes = sum(1 for i in range(1, len(seq)) if seq[i] != seq[i-1])
                total_changes += changes
        
        avg_changes = total_changes / total_sequences
        
        # 稳定性评分：变化越少，评分越高
        max_possible_changes = max(len(seq) - 1 for seq in prediction_sequences if seq)
        stability_score = 1.0 - (avg_changes / max_possible_changes) if max_possible_changes > 0 else 1.0
        
        return {
            'stability_score': max(0.0, stability_score),
            'avg_changes': avg_changes,
            'total_changes': total_changes
        }
    
    def calculate_misidentification_rate(self, 
                                       predictions: List[int], 
                                       labels: List[int],
                                       intent_labels: List[str]) -> Dict[str, float]:
        """
        计算误识别率
        
        Args:
            predictions: 预测结果列表
            labels: 真实标签列表
            intent_labels: 意图标签名称列表
            
        Returns:
            misid_dict: 误识别率字典
        """
        total_samples = len(predictions)
        misidentified = 0
        core_misidentified = 0
        core_total = 0
        
        for pred, label in zip(predictions, labels):
            if pred != label:
                misidentified += 1
                
                # 检查是否涉及核心指令
                true_intent = intent_labels[label] if label < len(intent_labels) else "UNKNOWN"
                pred_intent = intent_labels[pred] if pred < len(intent_labels) else "UNKNOWN"
                
                if true_intent in self.core_commands or pred_intent in self.core_commands:
                    core_misidentified += 1
            
            # 统计核心指令总数
            true_intent = intent_labels[label] if label < len(intent_labels) else "UNKNOWN"
            if true_intent in self.core_commands:
                core_total += 1
        
        total_misid_rate = misidentified / total_samples if total_samples > 0 else 0.0
        core_misid_rate = core_misidentified / core_total if core_total > 0 else 0.0
        
        return {
            'total_misidentification_rate': total_misid_rate,
            'core_misidentification_rate': core_misid_rate,
            'misidentified_samples': misidentified,
            'core_misidentified_samples': core_misidentified
        } 