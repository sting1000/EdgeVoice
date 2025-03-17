#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
准备ASR训练数据的工具，包括音频处理和标注文件生成
"""

import os
import sys
import argparse
import logging
import json
import time
import pandas as pd
import numpy as np
import librosa
import soundfile as sf
from tqdm import tqdm
import multiprocessing
from pathlib import Path
import re
import random
import shutil

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 导入配置
try:
    from config import (
        TARGET_SAMPLE_RATE, ASR_MAX_LENGTH, ASR_MIN_LENGTH,
        ASR_VAD_ENABLED, ASR_VAD_MODE, ASR_VAD_PADDING_MS
    )
except ImportError:
    # 默认配置
    TARGET_SAMPLE_RATE = 16000
    ASR_MAX_LENGTH = 50000
    ASR_MIN_LENGTH = 10
    ASR_VAD_ENABLED = True
    ASR_VAD_MODE = 3
    ASR_VAD_PADDING_MS = 300

# 设置日志
logging.basicConfig(
    format='%(asctime)s %(levelname)s: %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

def load_audio(file_path, sr=TARGET_SAMPLE_RATE):
    """
    加载音频文件
    
    参数:
        file_path: 音频文件路径
        sr: 目标采样率
        
    返回:
        音频数据和采样率
    """
    try:
        audio, sr = librosa.load(file_path, sr=sr)
        return audio, sr
    except Exception as e:
        logger.error(f"加载音频文件失败 {file_path}: {str(e)}")
        return None, None

def process_audio(audio_file, output_dir, min_length=ASR_MIN_LENGTH, max_length=ASR_MAX_LENGTH, 
                  target_sr=TARGET_SAMPLE_RATE, vad_enabled=ASR_VAD_ENABLED, 
                  vad_mode=ASR_VAD_MODE, vad_padding_ms=ASR_VAD_PADDING_MS):
    """
    处理单个音频文件
    
    参数:
        audio_file: 音频文件路径
        output_dir: 输出目录
        min_length: 最小音频长度（帧）
        max_length: 最大音频长度（帧）
        target_sr: 目标采样率
        vad_enabled: 是否启用VAD
        vad_mode: VAD模式
        vad_padding_ms: VAD填充毫秒数
        
    返回:
        处理后的音频文件路径，如果处理失败则返回None
    """
    try:
        # 加载音频
        audio, sr = load_audio(audio_file, sr=target_sr)
        if audio is None:
            return None
        
        # 检查音频长度
        if len(audio) < min_length:
            logger.warning(f"音频太短，跳过: {audio_file}")
            return None
        
        # 如果音频太长，截断
        if len(audio) > max_length:
            logger.warning(f"音频太长，截断: {audio_file}")
            audio = audio[:max_length]
        
        # 应用VAD（语音活动检测）
        if vad_enabled:
            try:
                import webrtcvad
                vad = webrtcvad.Vad(vad_mode)
                
                # 将音频转换为适合VAD的格式
                frame_duration_ms = 30  # 30ms帧
                frame_size = int(target_sr * frame_duration_ms / 1000)
                frames = []
                
                for i in range(0, len(audio) - frame_size + 1, frame_size):
                    frame = audio[i:i+frame_size]
                    # 转换为16位整数PCM
                    frame_int = (frame * 32768).astype(np.int16)
                    frames.append(frame_int.tobytes())
                
                # 检测语音段
                is_speech = []
                for frame in frames:
                    try:
                        is_speech.append(vad.is_speech(frame, target_sr))
                    except:
                        is_speech.append(False)
                
                # 找出语音段
                speech_segments = []
                in_speech = False
                start = 0
                
                for i, speech in enumerate(is_speech):
                    if speech and not in_speech:
                        # 语音开始
                        start = max(0, i - int(vad_padding_ms / frame_duration_ms))
                        in_speech = True
                    elif not speech and in_speech:
                        # 语音结束
                        end = min(len(is_speech), i + int(vad_padding_ms / frame_duration_ms))
                        speech_segments.append((start, end))
                        in_speech = False
                
                if in_speech:
                    # 处理最后一段语音
                    speech_segments.append((start, len(is_speech)))
                
                # 如果没有检测到语音段，使用整个音频
                if not speech_segments:
                    logger.warning(f"未检测到语音段，使用整个音频: {audio_file}")
                else:
                    # 合并语音段
                    merged_audio = np.zeros_like(audio)
                    for start, end in speech_segments:
                        start_sample = start * frame_size
                        end_sample = min(len(audio), end * frame_size)
                        merged_audio[start_sample:end_sample] = audio[start_sample:end_sample]
                    
                    # 移除静音部分
                    non_zero = np.where(merged_audio != 0)[0]
                    if len(non_zero) > 0:
                        audio = merged_audio[non_zero[0]:non_zero[-1]+1]
                    else:
                        logger.warning(f"VAD移除了所有音频，使用原始音频: {audio_file}")
            
            except ImportError:
                logger.warning("未安装webrtcvad，跳过VAD处理")
            except Exception as e:
                logger.error(f"VAD处理失败: {str(e)}")
        
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        # 生成输出文件名
        file_name = os.path.basename(audio_file)
        output_file = os.path.join(output_dir, file_name)
        
        # 如果输出文件已存在，添加后缀
        if os.path.exists(output_file):
            base, ext = os.path.splitext(file_name)
            output_file = os.path.join(output_dir, f"{base}_{int(time.time())}{ext}")
        
        # 保存处理后的音频
        sf.write(output_file, audio, target_sr)
        
        return output_file
    
    except Exception as e:
        logger.error(f"处理音频文件失败 {audio_file}: {str(e)}")
        return None

def process_audio_files(input_files, output_dir, num_workers=None, **kwargs):
    """
    并行处理多个音频文件
    
    参数:
        input_files: 音频文件路径列表
        output_dir: 输出目录
        num_workers: 工作进程数
        **kwargs: 传递给process_audio的其他参数
        
    返回:
        处理后的音频文件路径列表
    """
    if num_workers is None:
        num_workers = max(1, multiprocessing.cpu_count() - 1)
    
    logger.info(f"使用 {num_workers} 个进程处理 {len(input_files)} 个音频文件")
    
    # 创建进程池
    with multiprocessing.Pool(num_workers) as pool:
        # 准备参数
        args = [(file, output_dir, kwargs.get('min_length', ASR_MIN_LENGTH), 
                 kwargs.get('max_length', ASR_MAX_LENGTH), 
                 kwargs.get('target_sr', TARGET_SAMPLE_RATE),
                 kwargs.get('vad_enabled', ASR_VAD_ENABLED),
                 kwargs.get('vad_mode', ASR_VAD_MODE),
                 kwargs.get('vad_padding_ms', ASR_VAD_PADDING_MS)) 
                for file in input_files]
        
        # 并行处理
        results = list(tqdm(pool.starmap(process_audio, args), total=len(input_files), desc="处理音频文件"))
    
    # 过滤掉处理失败的文件
    processed_files = [f for f in results if f is not None]
    logger.info(f"成功处理 {len(processed_files)}/{len(input_files)} 个音频文件")
    
    return processed_files

def find_audio_files(input_dir, recursive=True, extensions=('.wav', '.mp3', '.flac', '.ogg')):
    """
    查找目录中的音频文件
    
    参数:
        input_dir: 输入目录
        recursive: 是否递归查找子目录
        extensions: 音频文件扩展名
        
    返回:
        音频文件路径列表
    """
    audio_files = []
    
    if recursive:
        for root, _, files in os.walk(input_dir):
            for file in files:
                if file.lower().endswith(extensions):
                    audio_files.append(os.path.join(root, file))
    else:
        for file in os.listdir(input_dir):
            if file.lower().endswith(extensions):
                audio_files.append(os.path.join(input_dir, file))
    
    logger.info(f"在 {input_dir} 中找到 {len(audio_files)} 个音频文件")
    return audio_files

def clean_text(text):
    """
    清理文本，移除标点符号和多余的空格
    
    参数:
        text: 输入文本
        
    返回:
        清理后的文本
    """
    # 移除标点符号
    text = re.sub(r'[^\w\s]', '', text)
    # 移除多余的空格
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def create_annotation_file(audio_files, text_dict, output_file, clean=True, split=None):
    """
    创建标注文件
    
    参数:
        audio_files: 音频文件路径列表
        text_dict: 音频文件到文本的映射字典
        output_file: 输出文件路径
        clean: 是否清理文本
        split: 是否分割数据集，格式为(train_ratio, dev_ratio, test_ratio)
        
    返回:
        创建的标注文件路径
    """
    data = []
    
    for audio_file in audio_files:
        file_name = os.path.basename(audio_file)
        
        # 查找文本
        text = None
        for key in [audio_file, file_name]:
            if key in text_dict:
                text = text_dict[key]
                break
        
        if text is None:
            logger.warning(f"未找到音频文件的文本: {audio_file}")
            continue
        
        # 清理文本
        if clean:
            text = clean_text(text)
        
        # 跳过空文本
        if not text:
            logger.warning(f"清理后文本为空，跳过: {audio_file}")
            continue
        
        data.append({
            'file_path': audio_file,
            'text': text
        })
    
    # 创建DataFrame
    df = pd.DataFrame(data)
    
    # 如果需要分割数据集
    if split:
        train_ratio, dev_ratio, test_ratio = split
        assert abs(train_ratio + dev_ratio + test_ratio - 1.0) < 1e-10, "分割比例之和必须为1"
        
        # 随机打乱数据
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)
        
        # 计算分割点
        n = len(df)
        train_end = int(n * train_ratio)
        dev_end = train_end + int(n * dev_ratio)
        
        # 分割数据集
        train_df = df[:train_end]
        dev_df = df[train_end:dev_end]
        test_df = df[dev_end:]
        
        # 保存各个数据集
        base_name, ext = os.path.splitext(output_file)
        
        train_file = f"{base_name}_train{ext}"
        dev_file = f"{base_name}_dev{ext}"
        test_file = f"{base_name}_test{ext}"
        
        train_df.to_csv(train_file, index=False)
        dev_df.to_csv(dev_file, index=False)
        test_df.to_csv(test_file, index=False)
        
        logger.info(f"训练集: {len(train_df)} 样本，保存到 {train_file}")
        logger.info(f"开发集: {len(dev_df)} 样本，保存到 {dev_file}")
        logger.info(f"测试集: {len(test_df)} 样本，保存到 {test_file}")
        
        return train_file, dev_file, test_file
    else:
        # 保存标注文件
        df.to_csv(output_file, index=False)
        logger.info(f"创建标注文件: {output_file}，包含 {len(df)} 个样本")
        return output_file

def load_text_from_file(text_file, encoding='utf-8'):
    """
    从文本文件加载文本
    
    参数:
        text_file: 文本文件路径
        encoding: 文件编码
        
    返回:
        音频文件到文本的映射字典
    """
    text_dict = {}
    
    try:
        with open(text_file, 'r', encoding=encoding) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                
                # 尝试解析不同格式
                if '\t' in line:
                    # 制表符分隔
                    parts = line.split('\t', 1)
                    if len(parts) == 2:
                        file_name, text = parts
                        text_dict[file_name] = text
                elif ' ' in line and not line.startswith(' '):
                    # 空格分隔，第一个空格前是文件名
                    parts = line.split(' ', 1)
                    if len(parts) == 2:
                        file_name, text = parts
                        text_dict[file_name] = text
                elif ':' in line:
                    # 冒号分隔
                    parts = line.split(':', 1)
                    if len(parts) == 2:
                        file_name, text = parts
                        text_dict[file_name.strip()] = text.strip()
    
    except Exception as e:
        logger.error(f"加载文本文件失败 {text_file}: {str(e)}")
    
    logger.info(f"从 {text_file} 加载了 {len(text_dict)} 条文本记录")
    return text_dict

def load_text_from_json(json_file, encoding='utf-8', key_field='file', text_field='text'):
    """
    从JSON文件加载文本
    
    参数:
        json_file: JSON文件路径
        encoding: 文件编码
        key_field: 文件名字段
        text_field: 文本字段
        
    返回:
        音频文件到文本的映射字典
    """
    text_dict = {}
    
    try:
        with open(json_file, 'r', encoding=encoding) as f:
            data = json.load(f)
            
            # 处理不同的JSON格式
            if isinstance(data, list):
                # 列表格式
                for item in data:
                    if isinstance(item, dict) and key_field in item and text_field in item:
                        text_dict[item[key_field]] = item[text_field]
            elif isinstance(data, dict):
                # 字典格式
                for key, value in data.items():
                    if isinstance(value, dict) and text_field in value:
                        text_dict[key] = value[text_field]
                    elif isinstance(value, str):
                        text_dict[key] = value
    
    except Exception as e:
        logger.error(f"加载JSON文件失败 {json_file}: {str(e)}")
    
    logger.info(f"从 {json_file} 加载了 {len(text_dict)} 条文本记录")
    return text_dict

def load_text_from_csv(csv_file, encoding='utf-8', file_column='file_path', text_column='text'):
    """
    从CSV文件加载文本
    
    参数:
        csv_file: CSV文件路径
        encoding: 文件编码
        file_column: 文件名列
        text_column: 文本列
        
    返回:
        音频文件到文本的映射字典
    """
    text_dict = {}
    
    try:
        df = pd.read_csv(csv_file, encoding=encoding)
        
        if file_column in df.columns and text_column in df.columns:
            for _, row in df.iterrows():
                file_name = row[file_column]
                text = row[text_column]
                if pd.notna(file_name) and pd.notna(text):
                    text_dict[file_name] = str(text)
        else:
            logger.error(f"CSV文件缺少必要的列: {file_column} 或 {text_column}")
    
    except Exception as e:
        logger.error(f"加载CSV文件失败 {csv_file}: {str(e)}")
    
    logger.info(f"从 {csv_file} 加载了 {len(text_dict)} 条文本记录")
    return text_dict

def prepare_wenet_data(processed_files, annotation_file, output_dir, dict_file=None):
    """
    准备WeNet格式的数据
    
    参数:
        processed_files: 处理后的音频文件路径列表
        annotation_file: 标注文件路径
        output_dir: 输出目录
        dict_file: 字典文件路径，如果为None则自动生成
        
    返回:
        WeNet数据目录路径
    """
    # 创建WeNet数据目录
    wenet_data_dir = os.path.join(output_dir, "wenet_data")
    os.makedirs(wenet_data_dir, exist_ok=True)
    
    # 加载标注文件
    df = pd.read_csv(annotation_file)
    
    # 创建wav.scp文件
    wav_scp_file = os.path.join(wenet_data_dir, "wav.scp")
    with open(wav_scp_file, 'w', encoding='utf-8') as f:
        for file_path in processed_files:
            uttid = os.path.splitext(os.path.basename(file_path))[0]
            f.write(f"{uttid} {os.path.abspath(file_path)}\n")
    
    # 创建text文件
    text_file = os.path.join(wenet_data_dir, "text")
    with open(text_file, 'w', encoding='utf-8') as f:
        for _, row in df.iterrows():
            file_path = row['file_path']
            text = row['text']
            uttid = os.path.splitext(os.path.basename(file_path))[0]
            f.write(f"{uttid} {text}\n")
    
    # 如果没有提供字典文件，则自动生成
    if dict_file is None:
        # 收集所有字符
        all_chars = set()
        for text in df['text']:
            all_chars.update(text)
        
        # 排序并添加特殊标记
        chars = sorted(list(all_chars))
        special_tokens = ['<blank>', '<unk>', '<s>', '</s>']
        vocab = special_tokens + chars
        
        # 创建字典文件
        dict_file = os.path.join(wenet_data_dir, "lang_char.txt")
        with open(dict_file, 'w', encoding='utf-8') as f:
            for i, char in enumerate(vocab):
                f.write(f"{char} {i}\n")
    else:
        # 复制字典文件
        shutil.copy(dict_file, os.path.join(wenet_data_dir, "lang_char.txt"))
    
    logger.info(f"WeNet数据准备完成，保存在: {wenet_data_dir}")
    logger.info(f"wav.scp: {wav_scp_file}")
    logger.info(f"text: {text_file}")
    logger.info(f"字典文件: {dict_file}")
    
    return wenet_data_dir

def main():
    parser = argparse.ArgumentParser(description="准备ASR训练数据")
    parser.add_argument("--input", type=str, required=True, help="输入音频目录或文件列表")
    parser.add_argument("--output", type=str, required=True, help="输出目录")
    parser.add_argument("--text", type=str, required=True, help="文本文件路径，支持TXT、JSON、CSV格式")
    parser.add_argument("--recursive", action="store_true", help="递归查找子目录中的音频文件")
    parser.add_argument("--num_workers", type=int, default=None, help="工作进程数")
    parser.add_argument("--vad", action="store_true", help="启用VAD处理")
    parser.add_argument("--vad_mode", type=int, default=3, choices=[0, 1, 2, 3], help="VAD模式")
    parser.add_argument("--min_length", type=int, default=ASR_MIN_LENGTH, help="最小音频长度（帧）")
    parser.add_argument("--max_length", type=int, default=ASR_MAX_LENGTH, help="最大音频长度（帧）")
    parser.add_argument("--sample_rate", type=int, default=TARGET_SAMPLE_RATE, help="目标采样率")
    parser.add_argument("--split", action="store_true", help="分割数据集为训练集、开发集和测试集")
    parser.add_argument("--train_ratio", type=float, default=0.8, help="训练集比例")
    parser.add_argument("--dev_ratio", type=float, default=0.1, help="开发集比例")
    parser.add_argument("--test_ratio", type=float, default=0.1, help="测试集比例")
    parser.add_argument("--wenet_format", action="store_true", help="生成WeNet格式的数据")
    parser.add_argument("--dict_file", type=str, help="字典文件路径（用于WeNet格式）")
    
    args = parser.parse_args()
    
    # 创建输出目录
    os.makedirs(args.output, exist_ok=True)
    
    # 查找音频文件
    if os.path.isdir(args.input):
        audio_files = find_audio_files(args.input, args.recursive)
    else:
        # 假设是文件列表
        with open(args.input, 'r') as f:
            audio_files = [line.strip() for line in f if line.strip()]
    
    # 加载文本
    text_dict = {}
    if args.text.endswith('.json'):
        text_dict = load_text_from_json(args.text)
    elif args.text.endswith('.csv'):
        text_dict = load_text_from_csv(args.text)
    else:
        text_dict = load_text_from_file(args.text)
    
    # 处理音频文件
    processed_dir = os.path.join(args.output, "processed_audio")
    processed_files = process_audio_files(
        audio_files, 
        processed_dir, 
        args.num_workers,
        min_length=args.min_length,
        max_length=args.max_length,
        target_sr=args.sample_rate,
        vad_enabled=args.vad,
        vad_mode=args.vad_mode,
        vad_padding_ms=ASR_VAD_PADDING_MS
    )
    
    # 创建标注文件
    annotation_file = os.path.join(args.output, "annotations.csv")
    
    if args.split:
        split_ratios = (args.train_ratio, args.dev_ratio, args.test_ratio)
        annotation_files = create_annotation_file(
            processed_files, 
            text_dict, 
            annotation_file, 
            clean=True, 
            split=split_ratios
        )
        train_file = annotation_files[0]
    else:
        annotation_file = create_annotation_file(
            processed_files, 
            text_dict, 
            annotation_file, 
            clean=True
        )
        train_file = annotation_file
    
    # 如果需要生成WeNet格式的数据
    if args.wenet_format:
        wenet_data_dir = prepare_wenet_data(
            processed_files, 
            train_file, 
            args.output, 
            args.dict_file
        )
    
    logger.info("数据准备完成!")

if __name__ == "__main__":
    main() 