"""
音频数据增强模块
提供丰富的数据增强功能，用于提高模型泛化能力
"""

import os
import numpy as np
import torch
import librosa
import random
import pandas as pd
import hashlib
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import soundfile as sf
import time
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 默认配置参数
DEFAULT_SAMPLE_RATE = 16000
CACHE_DIR = os.path.join("data", "cache")

class AudioAugmenter:
    """音频增强器，提供多种音频增强方法"""
    
    def __init__(self, sample_rate=DEFAULT_SAMPLE_RATE, seed=None):
        """
        初始化音频增强器
        
        参数:
            sample_rate: 音频采样率
            seed: 随机数种子，用于复现
        """
        self.sample_rate = sample_rate
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
    
    def pitch_shift(self, audio, n_steps=None):
        """
        音高变化增强
        
        参数:
            audio: 音频信号
            n_steps: 音高变化的半音数，范围[-3, 3]
        
        返回:
            增强后的音频
        """
        if n_steps is None:
            n_steps = random.uniform(-3, 3)  # 随机在-3到3个半音之间变化
        
        return librosa.effects.pitch_shift(
            y=audio, 
            sr=self.sample_rate, 
            n_steps=n_steps
        )
    
    def time_stretch(self, audio, rate=None):
        """
        时间伸缩增强
        
        参数:
            audio: 音频信号
            rate: 伸缩比例，范围[0.8, 1.2]
        
        返回:
            增强后的音频
        """
        if rate is None:
            rate = random.uniform(0.8, 1.2)  # 随机在0.8到1.2倍速之间变化
        
        return librosa.effects.time_stretch(audio, rate=rate)
    
    def volume_adjustment(self, audio, factor=None):
        """
        音量调整增强
        
        参数:
            audio: 音频信号
            factor: 音量调整系数，范围[0.5, 1.5]
        
        返回:
            增强后的音频
        """
        if factor is None:
            factor = random.uniform(0.5, 1.5)  # 随机在0.5到1.5倍音量之间变化
        
        return audio * factor
    
    def add_noise(self, audio, noise_level=None):
        """
        添加噪声增强
        
        参数:
            audio: 音频信号
            noise_level: 噪声强度，范围[0.001, 0.01]
        
        返回:
            增强后的音频
        """
        if noise_level is None:
            noise_level = random.uniform(0.001, 0.01)  # 随机在0.001到0.01之间的噪声等级
        
        noise = np.random.randn(len(audio)) * noise_level
        return audio + noise
    
    def combined_augmentation(self, audio):
        """
        组合增强，随机应用多种增强方法
        
        参数:
            audio: 音频信号
        
        返回:
            增强后的音频
        """
        aug_methods = [
            self.pitch_shift,
            self.time_stretch,
            self.volume_adjustment,
            self.add_noise
        ]
        
        # 随机选择1-3种增强方法
        n_augmentations = random.randint(1, 3)
        selected_methods = random.sample(aug_methods, n_augmentations)
        
        augmented_audio = audio.copy()
        for method in selected_methods:
            augmented_audio = method(augmented_audio)
        
        return augmented_audio
    
    def augment(self, audio, method=None, augment_prob=0.5):
        """
        应用音频增强
        
        参数:
            audio: 音频信号
            method: 指定增强方法，默认随机选择
            augment_prob: 应用增强的概率
        
        返回:
            增强后的音频
        """
        # 根据概率决定是否增强
        if random.random() > augment_prob:
            return audio
        
        methods = {
            'pitch_shift': self.pitch_shift,
            'time_stretch': self.time_stretch,
            'volume_adjustment': self.volume_adjustment,
            'add_noise': self.add_noise,
            'combined': self.combined_augmentation
        }
        
        if method is None or method not in methods:
            method = random.choice(list(methods.keys()))
        
        return methods[method](audio)

class AugmentedAudioDataset(Dataset):
    """支持数据增强的音频数据集"""
    
    def __init__(self, annotation_file, data_dir, feature_extractor=None, 
                 augment=True, augment_prob=0.5, use_cache=True, seed=42):
        """
        初始化数据集
        
        参数:
            annotation_file: 标注文件路径
            data_dir: 音频文件目录
            feature_extractor: 特征提取器函数
            augment: 是否启用数据增强
            augment_prob: 应用增强的概率
            use_cache: 是否使用缓存
            seed: 随机数种子
        """
        self.data_dir = data_dir
        self.annotations = pd.read_csv(annotation_file, encoding='utf-8')
        self.feature_extractor = feature_extractor
        self.augment = augment
        self.augment_prob = augment_prob
        self.use_cache = use_cache
        self.sample_rate = DEFAULT_SAMPLE_RATE
        
        # 标签映射
        self.intent_labels = sorted(self.annotations['intent'].unique())
        self.label_to_idx = {label: idx for idx, label in enumerate(self.intent_labels)}
        
        # 初始化增强器
        self.augmenter = AudioAugmenter(sample_rate=self.sample_rate, seed=seed)
        
        # 初始化缓存
        self.cache_dir = CACHE_DIR
        self.cache = {}
        
        if self.use_cache:
            os.makedirs(self.cache_dir, exist_ok=True)
            self._init_cache()
    
    def _init_cache(self):
        """初始化音频缓存"""
        cache_files = [f for f in os.listdir(self.cache_dir) if f.endswith('.npy')]
        
        for cache_file in cache_files:
            file_id = os.path.splitext(cache_file)[0]
            self.cache[file_id] = os.path.join(self.cache_dir, cache_file)
        
        logger.info(f"已加载 {len(self.cache)} 个音频缓存")
    
    def _get_audio_path(self, index):
        """获取音频文件路径"""
        file_path = self.annotations.iloc[index]['file_path']
        return os.path.join(self.data_dir, file_path)
    
    def _get_cache_id(self, index):
        """获取缓存ID"""
        file_path = self.annotations.iloc[index]['file_path']
        return hashlib.md5(file_path.encode()).hexdigest()
    
    def _load_audio(self, index):
        """加载音频文件"""
        if self.use_cache:
            cache_id = self._get_cache_id(index)
            cache_path = os.path.join(self.cache_dir, f"{cache_id}.npy")
            
            if cache_id in self.cache:
                # 从缓存加载
                try:
                    audio = np.load(cache_path)
                    return audio, self.sample_rate
                except Exception as e:
                    logger.warning(f"缓存加载失败: {e}")
                    # 如果缓存加载失败，继续正常加载
        
        # 从文件加载
        audio_path = self._get_audio_path(index)
        try:
            audio, sr = librosa.load(audio_path, sr=self.sample_rate)
            
            # 保存到缓存
            if self.use_cache:
                cache_id = self._get_cache_id(index)
                cache_path = os.path.join(self.cache_dir, f"{cache_id}.npy")
                np.save(cache_path, audio)
                self.cache[cache_id] = cache_path
            
            return audio, sr
        except Exception as e:
            logger.error(f"音频加载失败: {audio_path} - {e}")
            # 返回一个空音频
            return np.zeros(self.sample_rate), self.sample_rate
    
    def __len__(self):
        """返回数据集大小"""
        return len(self.annotations)
    
    def __getitem__(self, index):
        """获取一个样本"""
        # 加载音频
        audio, sr = self._load_audio(index)
        
        # 应用数据增强
        if self.augment:
            audio = self.augmenter.augment(audio, augment_prob=self.augment_prob)
        
        # 获取文本转录（如果有）
        transcript = ""
        if 'transcript' in self.annotations.columns:
            transcript = self.annotations.iloc[index]['transcript']
        
        # 提取特征（如果提供了特征提取器）
        features = audio
        if self.feature_extractor is not None:
            features = self.feature_extractor(audio, sr, transcript=transcript)
        
        # 获取标签
        intent = self.annotations.iloc[index]['intent']
        label_idx = self.label_to_idx[intent]
        
        return {
            'audio': audio,
            'features': features,
            'intent': intent,
            'label': label_idx,
            'transcript': transcript,
            'file_path': self._get_audio_path(index)
        }

def standardize_audio_length(audio, sample_rate, target_length=5.0, min_length=0.5):
    """
    将音频长度标准化
    
    参数:
        audio: 音频数据
        sample_rate: 采样率
        target_length: 目标长度（秒）
        min_length: 最小长度（秒）
    
    返回:
        标准化后的音频
    """
    target_samples = int(target_length * sample_rate)
    current_samples = len(audio)
    
    # 如果音频过短，通过重复填充
    if current_samples < min_length * sample_rate:
        repeats = int(np.ceil((min_length * sample_rate) / current_samples))
        audio = np.tile(audio, repeats)
        current_samples = len(audio)
    
    # 如果音频长度合适，直接返回
    if current_samples == target_samples:
        return audio
    
    # 如果音频过长，进行裁剪（保留中间部分）
    if current_samples > target_samples:
        start = (current_samples - target_samples) // 2
        return audio[start:start + target_samples]
    
    # 如果音频过短，在两端填充0
    padded_audio = np.zeros(target_samples)
    start = (target_samples - current_samples) // 2
    padded_audio[start:start + current_samples] = audio
    
    return padded_audio

def collate_fn(batch):
    """
    数据批次整理函数
    
    参数:
        batch: 数据批次
    
    返回:
        整理后的批次
    """
    audio = []
    features = []
    intents = []
    labels = []
    transcripts = []
    file_paths = []
    
    # 检查特征类型（是否为字典）
    is_dict_features = False
    if batch and isinstance(batch[0]['features'], dict):
        is_dict_features = True
        # 针对字典类型的特征初始化
        dict_features = {k: [] for k in batch[0]['features'].keys()}
    
    for item in batch:
        audio.append(item['audio'])
        
        # 处理字典类型的特征（用于精确分类器）
        if is_dict_features:
            for k, v in item['features'].items():
                dict_features[k].append(v)
        else:
            # 处理普通数组类型的特征（用于快速分类器）
            features.append(item['features'])
            
        intents.append(item['intent'])
        labels.append(item['label'])
        transcripts.append(item['transcript'])
        file_paths.append(item['file_path'])
    
    # 构建返回结果
    result = {
        'audio': audio,
        'intent': intents,
        'label': torch.tensor(labels, dtype=torch.long),
        'transcript': transcripts,
        'file_path': file_paths
    }
    
    # 根据特征类型添加到结果中
    if is_dict_features:
        # 对于字典类型特征，分别进行张量转换
        result['features'] = {
            k: torch.stack([torch.tensor(v_i) for v_i in v]) 
            for k, v in dict_features.items()
        }
    else:
        # 对于普通数组特征，直接转换为张量
        result['features'] = torch.tensor(np.array(features), dtype=torch.float32)
    
    return result

def prepare_augmented_dataloader(annotation_file, data_dir, feature_extractor=None, 
                                batch_size=32, shuffle=True, augment=True, 
                                augment_prob=0.5, use_cache=True, seed=42,
                                num_workers=4):
    """
    准备带数据增强的数据加载器
    
    参数:
        annotation_file: 标注文件路径
        data_dir: 音频文件目录
        feature_extractor: 特征提取器函数
        batch_size: 批次大小
        shuffle: 是否打乱数据
        augment: 是否启用数据增强
        augment_prob: 应用增强的概率
        use_cache: 是否使用缓存
        seed: 随机数种子
        num_workers: 数据加载线程数
    
    返回:
        数据加载器
    """
    import hashlib
    
    dataset = AugmentedAudioDataset(
        annotation_file=annotation_file,
        data_dir=data_dir,
        feature_extractor=feature_extractor,
        augment=augment,
        augment_prob=augment_prob,
        use_cache=use_cache,
        seed=seed
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn,
        num_workers=num_workers,
        drop_last=False
    )
    
    # 输出数据加载器信息
    logger.info(f"数据集大小: {len(dataset)} 样本")
    logger.info(f"意图类别: {dataset.intent_labels}")
    logger.info(f"数据增强: {'启用' if augment else '禁用'} (概率: {augment_prob})")
    logger.info(f"音频缓存: {'启用' if use_cache else '禁用'}")
    
    return dataloader, dataset.intent_labels

# 测试代码
if __name__ == "__main__":
    # 测试数据增强器
    import matplotlib.pyplot as plt
    
    # 生成一个示例音频（正弦波）
    sr = 16000
    t = np.linspace(0, 3, 3 * sr)
    audio = 0.5 * np.sin(2 * np.pi * 440 * t)  # 440Hz音调
    
    # 创建音频增强器
    augmenter = AudioAugmenter(sample_rate=sr)
    
    # 应用各种增强方法
    audio_pitch = augmenter.pitch_shift(audio)
    audio_time = augmenter.time_stretch(audio)
    audio_volume = augmenter.volume_adjustment(audio)
    audio_noise = augmenter.add_noise(audio)
    audio_combined = augmenter.combined_augmentation(audio)
    
    # 绘制原始音频和增强后的音频
    plt.figure(figsize=(12, 10))
    
    plt.subplot(6, 1, 1)
    plt.title("原始音频")
    plt.plot(audio[:sr])
    
    plt.subplot(6, 1, 2)
    plt.title("音高变化增强")
    plt.plot(audio_pitch[:sr])
    
    plt.subplot(6, 1, 3)
    plt.title("时间伸缩增强")
    plt.plot(audio_time[:sr])
    
    plt.subplot(6, 1, 4)
    plt.title("音量调整增强")
    plt.plot(audio_volume[:sr])
    
    plt.subplot(6, 1, 5)
    plt.title("噪声添加增强")
    plt.plot(audio_noise[:sr])
    
    plt.subplot(6, 1, 6)
    plt.title("组合增强")
    plt.plot(audio_combined[:sr])
    
    plt.tight_layout()
    plt.savefig("augmentation_examples.png")
    plt.close()
    
    print("数据增强示例已保存至 augmentation_examples.png")
    
    # 测试数据加载器
    try:
        # 尝试加载一些测试数据
        annotation_file = "data/annotations.csv"
        data_dir = "data"
        
        if os.path.exists(annotation_file):
            print(f"测试数据加载器: {annotation_file}")
            dataloader, intent_labels = prepare_augmented_dataloader(
                annotation_file=annotation_file,
                data_dir=data_dir,
                batch_size=4,
                augment=True,
                augment_prob=0.7
            )
            
            # 获取一个批次
            batch = next(iter(dataloader))
            print(f"批次大小: {len(batch['audio'])}")
            print(f"特征形状: {batch['features'].shape}")
            print(f"意图标签: {batch['intent']}")
            print(f"数值标签: {batch['label']}")
            print(f"意图类别: {intent_labels}")
        else:
            print(f"注释文件不存在: {annotation_file}")
    except Exception as e:
        print(f"测试数据加载器时出错: {e}") 