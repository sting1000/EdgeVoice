# data_utils.py  
import os  
import numpy as np  
import torch  
import librosa  
import pandas as pd  
from torch.utils.data import Dataset, DataLoader  
from transformers import DistilBertTokenizer  
from config import *  
from audio_preprocessing import AudioPreprocessor  
from feature_extraction import FeatureExtractor  

def collate_fn(batch):  
    """自定义collate函数，处理不同长度的特征"""  
    # 分离特征和标签  
    features = [item[0] for item in batch]  
    labels = [item[1] for item in batch]  
    
    # 获取批次中最大的特征长度  
    max_length = max([f.shape[0] for f in features])  
    
    # 填充特征到相同长度  
    padded_features = []  
    for feat in features:  
        # 计算需要填充的长度  
        pad_length = max_length - feat.shape[0]  
        if pad_length > 0:  
            # 创建填充张量并连接  
            padding = torch.zeros((pad_length, feat.shape[1]), dtype=feat.dtype)  
            padded_feat = torch.cat([feat, padding], dim=0)  
        else:  
            padded_feat = feat  
        padded_features.append(padded_feat)  
    
    # 将填充后的特征堆叠成批次  
    features_batch = torch.stack(padded_features)  
    labels_batch = torch.tensor(labels)  
    
    return features_batch, labels_batch  

def precise_collate_fn(batch):  
    """处理precise模式下的数据批次，这里batch中的每个item都是字典"""  
    # 提取各个组件  
    features = [item['features'] for item in batch]  
    input_ids = [item['input_ids'] for item in batch]  
    attention_masks = [item['attention_mask'] for item in batch]  
    labels = [item['label'] for item in batch]  
    
    # 获取批次中最大的特征长度  
    max_length = max([f.shape[0] for f in features])  
    
    # 填充特征到相同长度  
    padded_features = []  
    for feat in features:  
        # 计算需要填充的长度  
        pad_length = max_length - feat.shape[0]  
        if pad_length > 0:  
            # 创建填充张量并连接  
            padding = torch.zeros((pad_length, feat.shape[1]), dtype=feat.dtype)  
            padded_feat = torch.cat([feat, padding], dim=0)  
        else:  
            padded_feat = feat  
        padded_features.append(padded_feat)  
    
    # 将填充后的特征堆叠成批次  
    features_batch = torch.stack(padded_features)  
    input_ids_batch = torch.stack(input_ids)  
    attention_masks_batch = torch.stack(attention_masks)  
    labels_batch = torch.tensor(labels)  
    
    return {  
        'features': features_batch,  
        'input_ids': input_ids_batch,  
        'attention_mask': attention_masks_batch,  
        'labels': labels_batch  
    }  

def load_audio(file_path, sample_rate=SAMPLE_RATE):  
    """加载音频文件"""  
    audio, sr = librosa.load(file_path, sr=sample_rate, mono=True)  
    return audio  

def prepare_audio_features(audio, preprocessor, feature_extractor):  
    """准备音频特征用于模型输入"""  
    # 预处理音频  
    processed_audio = preprocessor.process(audio)  
    
    # 提取特征  
    features = feature_extractor.extract_features(processed_audio)  
    
    return features  

def text_to_input_ids(text, tokenizer, max_length=128):  
    """将文本转换为模型输入ID"""  
    encoding = tokenizer(  
        text,  
        max_length=max_length,  
        padding='max_length',  
        truncation=True,  
        return_tensors='pt'  
    )  
    
    return encoding['input_ids'], encoding['attention_mask']  

class AudioIntentDataset(Dataset):  
    """语音意图数据集"""  
    def __init__(self, data_dir, annotation_file, transform=True, mode='fast'):  
        """  
        data_dir: 音频文件目录  
        annotation_file: 包含文件路径和标签的CSV文件  
        transform: 是否应用预处理和特征提取  
        mode: 'fast'使用纯音频特征，'precise'使用文本+音频特征  
        """  
        self.data_dir = data_dir  
        self.transform = transform  
        self.mode = mode  
        
        # 加载注释文件  
        self.annotations = pd.read_csv(annotation_file)  
        
        # 获取类别列表和类别到索引的映射  
        self.classes = INTENT_CLASSES  
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}  
        
        # 初始化预处理器和特征提取器  
        self.preprocessor = AudioPreprocessor() if transform else None  
        self.feature_extractor = FeatureExtractor() if transform else None  
        
        # 对于精确模式，初始化分词器  
        if mode == 'precise':  
            # 使用绝对路径  
            model_path = os.path.abspath(os.path.join(os.getcwd(), 'models/distilbert-base-uncased'))  
            # 指定为本地目录  
            self.tokenizer = DistilBertTokenizer.from_pretrained(  
                model_path,  
                local_files_only=True  
            ) 
        
    def __len__(self):  
        return len(self.annotations)  
    
    def __getitem__(self, idx):  
        # 获取音频文件路径和标签  
        audio_path = os.path.join(self.data_dir, self.annotations.iloc[idx]['file_path'])  
        intent_label = self.annotations.iloc[idx]['intent']  
        
        # 将标签转换为索引  
        label_idx = self.class_to_idx[intent_label]  
        
        # 加载音频文件  
        audio = load_audio(audio_path)  
        
        # 应用预处理和特征提取  
        if self.transform:  
            features = prepare_audio_features(audio, self.preprocessor, self.feature_extractor)  
            features_tensor = torch.FloatTensor(features)  
        else:  
            features_tensor = torch.FloatTensor(audio)  
        
        # 根据模式返回不同格式的数据  
        if self.mode == 'fast':  
            return features_tensor, label_idx  
        else:  # 'precise'模式  
            # 获取文本转录  
            transcript = self.annotations.iloc[idx]['transcript']  
            
            # 将文本转换为输入ID  
            input_ids, attention_mask = text_to_input_ids(transcript, self.tokenizer)  
            
            return {  
                'features': features_tensor,  
                'input_ids': input_ids.squeeze(0),  
                'attention_mask': attention_mask.squeeze(0),  
                'label': label_idx  
            }  

def prepare_dataloader(data_dir, annotation_file, batch_size=BATCH_SIZE, mode='fast'):  
    """准备数据加载器"""  
    dataset = AudioIntentDataset(data_dir, annotation_file, mode=mode)  
    
    # 根据模式选择不同的collate函数  
    if mode == 'fast':  
        collate = collate_fn  
    else:  # 'precise' 模式  
        collate = precise_collate_fn  
    
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate)  
    return dataloader  
