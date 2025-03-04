#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
音频数据增强脚本 - 用于离线生成增强的音频数据
"""

import os
import random
import argparse
import pandas as pd
import numpy as np
import librosa
import soundfile as sf
import matplotlib.pyplot as plt
from tqdm import tqdm
import warnings
from pathlib import Path

# 设置随机种子以确保可重现性
random.seed(42)
np.random.seed(42)


class AudioAugmenter:
    """音频增强器类，提供多种音频增强方法"""

    def __init__(self, sample_rate=16000):
        """
        初始化音频增强器
        
        参数:
            sample_rate: 采样率，默认16000Hz
        """
        self.sample_rate = sample_rate
        
    def time_stretch(self, audio, rate_range=(0.8, 1.2)):
        """
        时间伸缩（语速变化）
        
        参数:
            audio: 音频数据
            rate_range: 速率范围，如(0.8, 1.2)表示速度为原来的0.8-1.2倍
        
        返回:
            (增强后的音频, 使用的速率)
        """
        rate = random.uniform(*rate_range)
        augmented_audio = librosa.effects.time_stretch(audio, rate=rate)
        return augmented_audio, rate
    
    def pitch_shift(self, audio, n_steps_range=(-3, 3)):
        """
        音高变化
        
        参数:
            audio: 音频数据
            n_steps_range: 音高变化范围，半音数
        
        返回:
            (增强后的音频, 使用的半音数)
        """
        n_steps = random.uniform(*n_steps_range)
        augmented_audio = librosa.effects.pitch_shift(
            audio, sr=self.sample_rate, n_steps=n_steps
        )
        return augmented_audio, n_steps
    
    def adjust_volume(self, audio, gain_range=(0.5, 1.5)):
        """
        音量调整
        
        参数:
            audio: 音频数据
            gain_range: 增益范围，如(0.5, 1.5)表示音量为原来的0.5-1.5倍
        
        返回:
            (增强后的音频, 使用的增益)
        """
        gain = random.uniform(*gain_range)
        augmented_audio = audio * gain
        return augmented_audio, gain
    
    def add_noise(self, audio, noise_level_range=(0.001, 0.01)):
        """
        添加白噪声
        
        参数:
            audio: 音频数据
            noise_level_range: 噪声强度范围
        
        返回:
            (增强后的音频, 使用的噪声强度)
        """
        noise_level = random.uniform(*noise_level_range)
        noise = np.random.randn(len(audio))
        augmented_audio = audio + noise_level * noise
        return augmented_audio, noise_level
    
    def add_background(self, audio, background, level_range=(0.1, 0.3)):
        """
        添加背景音
        
        参数:
            audio: 音频数据
            background: 背景音数据
            level_range: 背景音量范围
        
        返回:
            (增强后的音频, 使用的背景音量)
        """
        level = random.uniform(*level_range)
        
        # 如果背景音较长，随机选择一段
        if len(background) > len(audio):
            start = random.randint(0, len(background) - len(audio))
            background = background[start:start + len(audio)]
        # 如果背景音较短，循环填充
        elif len(background) < len(audio):
            repeats = int(np.ceil(len(audio) / len(background)))
            background = np.tile(background, repeats)[:len(audio)]
            
        augmented_audio = audio + level * background
        
        # 归一化防止过载
        max_val = np.max(np.abs(augmented_audio))
        if max_val > 1.0:
            augmented_audio = augmented_audio / max_val * 0.9
            
        return augmented_audio, level
    
    def random_augment(self, audio, n_augmentations=1):
        """
        随机应用多种增强方法
        
        参数:
            audio: 音频数据
            n_augmentations: 要生成的增强版本数量
        
        返回:
            (增强后的音频列表, 操作描述列表)
        """
        augmented_audios = []
        descriptions = []
        
        aug_methods = [
            (self.time_stretch, "时间伸缩"),
            (self.pitch_shift, "音高变化"),
            (self.adjust_volume, "音量调整"),
            (self.add_noise, "添加噪声")
        ]
        
        for _ in range(n_augmentations):
            # 选择2-3种方法组合
            n_methods = random.randint(1, 3)
            selected_methods = random.sample(aug_methods, n_methods)
            
            current_audio = audio.copy()
            desc_parts = []
            
            for method, name in selected_methods:
                current_audio, param = method(current_audio)
                desc_parts.append(f"{name}({param:.2f})")
            
            description = " + ".join(desc_parts)
            
            augmented_audios.append(current_audio)
            descriptions.append(description)
        
        return augmented_audios, descriptions
    
    def augment_file(self, file_path, output_dir, n_augmentations=3, visualize=False):
        """
        增强单个音频文件并保存
        
        参数:
            file_path: 输入音频文件路径
            output_dir: 输出目录
            n_augmentations: 要生成的增强版本数量
            visualize: 是否生成可视化图表
        
        返回:
            增强文件的路径列表
        """
        # 加载音频
        try:
            audio, sr = librosa.load(file_path, sr=self.sample_rate)
        except Exception as e:
            warnings.warn(f"无法加载音频文件 {file_path}: {str(e)}")
            return []
        
        # 创建输出目录
        file_name = os.path.basename(file_path)
        file_dir = os.path.dirname(file_path)
        rel_dir = os.path.relpath(file_dir, start=os.path.dirname(output_dir))
        output_subdir = os.path.join(output_dir, rel_dir)
        os.makedirs(output_subdir, exist_ok=True)
        
        # 分离文件名和扩展名
        name, ext = os.path.splitext(file_name)
        
        # 应用增强
        augmented_audios, descriptions = self.random_augment(audio, n_augmentations)
        
        # 保存增强文件
        output_paths = []
        for i, (aug_audio, desc) in enumerate(zip(augmented_audios, descriptions)):
            aug_file_name = f"{name}_aug{i+1}{ext}"
            aug_file_path = os.path.join(output_subdir, aug_file_name)
            sf.write(aug_file_path, aug_audio, self.sample_rate)
            output_paths.append(os.path.join(rel_dir, aug_file_name))
            
            # 记录增强信息
            with open(f"{aug_file_path}.info.txt", "w") as f:
                f.write(f"原始文件: {file_path}\n")
                f.write(f"增强方法: {desc}\n")
        
        # 可视化
        if visualize:
            self.visualize_augmentation(audio, augmented_audios, descriptions, 
                                       output_dir, name)
        
        return output_paths
    
    def visualize_augmentation(self, original, augmented_list, descriptions, 
                              output_dir, base_name):
        """可视化原始音频与增强音频的对比"""
        # 创建可视化目录
        vis_dir = os.path.join(output_dir, "visualizations")
        os.makedirs(vis_dir, exist_ok=True)
        
        plt.figure(figsize=(12, 8))
        
        # 绘制原始音频
        plt.subplot(len(augmented_list) + 1, 1, 1)
        plt.plot(original)
        plt.title("原始音频")
        plt.grid(True)
        
        # 绘制增强音频
        for i, (audio, desc) in enumerate(zip(augmented_list, descriptions)):
            plt.subplot(len(augmented_list) + 1, 1, i + 2)
            plt.plot(audio)
            plt.title(f"增强 #{i+1}: {desc}")
            plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(vis_dir, f"{base_name}_augmentation.png"))
        plt.close()
    
    def augment_dataset(self, annotation_file, data_dir, output_dir, 
                       augment_factor=5, visualize=0):
        """
        增强整个数据集
        
        参数:
            annotation_file: 注释CSV文件路径
            data_dir: 原始数据目录
            output_dir: 输出目录
            augment_factor: 每个文件生成的增强版本数
            visualize: 生成可视化的文件数量
        
        返回:
            更新后的注释DataFrame
        """
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        # 读取原始注释文件
        df = pd.read_csv(annotation_file)
        print(f"原始数据集: {len(df)} 个样本")
        
        # 准备新的注释数据
        new_rows = []
        
        # 随机选择一些样本进行可视化
        vis_indices = random.sample(range(len(df)), min(visualize, len(df)))
        
        # 处理每个音频文件
        for i, row in tqdm(df.iterrows(), total=len(df), desc="增强数据集"):
            file_path = os.path.join(data_dir, row['file_path'])
            
            # 检查文件是否存在
            if not os.path.isfile(file_path):
                warnings.warn(f"文件不存在: {file_path}")
                continue
            
            # 应用增强，对选中的样本进行可视化
            vis = (i in vis_indices)
            augmented_paths = self.augment_file(
                file_path, output_dir, augment_factor, visualize=vis
            )
            
            # 添加增强样本到注释数据
            for aug_path in augmented_paths:
                new_row = row.copy()
                new_row['file_path'] = aug_path
                new_row['is_augmented'] = 1
                new_rows.append(new_row)
        
        # 添加标记字段到原始数据
        df['is_augmented'] = 0
        
        # 合并原始和增强的注释数据
        augmented_df = pd.DataFrame(new_rows)
        combined_df = pd.concat([df, augmented_df], ignore_index=True)
        
        # 保存新的注释文件
        output_annotation_file = os.path.join(output_dir, "augmented_annotations.csv")
        combined_df.to_csv(output_annotation_file, index=False)
        
        print(f"增强后的数据集: {len(combined_df)} 个样本")
        print(f"增强数据: {len(augmented_df)} 个样本")
        print(f"保存增强注释文件到: {output_annotation_file}")
        
        return combined_df


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="音频数据增强工具")
    
    parser.add_argument(
        "--annotation_file", 
        type=str, 
        required=True,
        help="原始数据集的注释CSV文件路径"
    )
    
    parser.add_argument(
        "--data_dir", 
        type=str, 
        default="data",
        help="原始音频文件目录，默认为'data'"
    )
    
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="data_augmented",
        help="增强数据的输出目录，默认为'data_augmented'"
    )
    
    parser.add_argument(
        "--augment_factor", 
        type=int, 
        default=5,
        help="每个原始样本生成的增强版本数量，默认为5"
    )
    
    parser.add_argument(
        "--visualize", 
        type=int, 
        default=5,
        help="生成可视化的样本数量，默认为5"
    )
    
    parser.add_argument(
        "--sample_rate", 
        type=int, 
        default=16000,
        help="音频采样率，默认为16000Hz"
    )
    
    return parser.parse_args()


def main():
    """主函数"""
    args = parse_args()
    
    print("=" * 50)
    print("音频数据增强工具")
    print("=" * 50)
    print(f"注释文件: {args.annotation_file}")
    print(f"数据目录: {args.data_dir}")
    print(f"输出目录: {args.output_dir}")
    print(f"增强因子: {args.augment_factor}")
    print(f"可视化数量: {args.visualize}")
    print(f"采样率: {args.sample_rate}")
    print("=" * 50)
    
    # 创建音频增强器
    augmenter = AudioAugmenter(sample_rate=args.sample_rate)
    
    # 执行数据集增强
    augmenter.augment_dataset(
        args.annotation_file,
        args.data_dir,
        args.output_dir,
        args.augment_factor,
        args.visualize
    )
    
    print("数据增强完成!")


if __name__ == "__main__":
    main() 