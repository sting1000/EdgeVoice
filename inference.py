# inference.py  
import torch  
import numpy as np  
import soundfile as sf  
import time  
from config import *  
from audio_preprocessing import AudioPreprocessor  
from feature_extraction import FeatureExtractor  
from models.fast_classifier import FastIntentClassifier  
from models.precise_classifier import PreciseIntentClassifier  
from transformers import DistilBertTokenizer  

class IntentInferenceEngine:  
    def __init__(self, fast_model_path, precise_model_path=None,   
                 fast_confidence_threshold=FAST_CONFIDENCE_THRESHOLD):  
        """  
        初始化推理引擎  
        fast_model_path: 一级分类器模型路径  
        precise_model_path: 二级分类器模型路径(可选)  
        fast_confidence_threshold: 一级分类器的置信度阈值  
        """  
        self.device = DEVICE  
        self.fast_confidence_threshold = fast_confidence_threshold  
        
        # 初始化预处理器和特征提取器  
        self.preprocessor = AudioPreprocessor()  
        self.feature_extractor = FeatureExtractor()  
        
        # 加载一级分类器  
        print("加载一级快速分类器...")  
        self.fast_model = self._load_fast_model(fast_model_path)  
        
        # 如果提供了路径，加载二级分类器  
        self.precise_model = None  
        if precise_model_path:  
            print("加载二级精确分类器...")  
            self.precise_model = self._load_precise_model(precise_model_path)  
            self.tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')  
        
        # 类别名称  
        self.intent_classes = INTENT_CLASSES  
        
    def _load_fast_model(self, model_path):  
        """加载一级快速分类器模型"""  
        # 注意：由于无法预知输入大小，这里使用一个占位值，实际使用时应调整  
        input_size = 39 * (2 * CONTEXT_FRAMES + 1)  # 假设的特征维度(MFCC+Delta+Delta2)*上下文帧数  
        model = FastIntentClassifier(input_size=input_size)  
        model.load_state_dict(torch.load(model_path, map_location=self.device))  
        model = model.to(self.device)  
        model.eval()  
        return model  
    
    def _load_precise_model(self, model_path):  
        """加载二级精确分类器模型"""  
        model = PreciseIntentClassifier()  
        model.load_state_dict(torch.load(model_path, map_location=self.device))  
        model = model.to(self.device)  
        model.eval()  
        return model  
    
    def preprocess_audio(self, audio):  
        """预处理音频并提取特征"""  
        start_time = time.time()  
        
        # 应用预处理  
        processed_audio = self.preprocessor.process(audio)  
        
        # 提取特征  
        features = self.feature_extractor.extract_features(processed_audio)  
        
        preprocess_time = time.time() - start_time  
        
        return features, preprocess_time  
    
    def fast_inference(self, features):  
        """使用一级分类器进行快速推理"""  
        start_time = time.time()  
        
        # 转换为torch张量并添加批次维度  
        features_tensor = torch.FloatTensor(features).unsqueeze(0).to(self.device)  
        
        # 推理  
        with torch.no_grad():  
            intent_idx, confidence = self.fast_model.predict(features_tensor)  
        
        # 获取预测的类别和置信度  
        intent_idx = intent_idx.item()  
        confidence = confidence.item()  
        intent_class = self.intent_classes[intent_idx]  
        
        inference_time = time.time() - start_time  
        
        return intent_class, confidence, inference_time  
    
    def precise_inference(self, audio_text):  
        """使用二级分类器进行精确推理"""  
        if not self.precise_model:  
            return None, 0, 0  
            
        start_time = time.time()  
        
        # 将文本转换为模型输入  
        encoding = self.tokenizer(  
            audio_text,  
            max_length=128,  
            padding='max_length',  
            truncation=True,  
            return_tensors='pt'  
        )  
        
        input_ids = encoding['input_ids'].to(self.device)  
        attention_mask = encoding['attention_mask'].to(self.device)  
        
        # 推理  
        with torch.no_grad():  
            intent_idx, confidence = self.precise_model.predict(input_ids, attention_mask)  
        
        # 获取预测的类别和置信度  
        intent_idx = intent_idx.item()  
        confidence = confidence.item()  
        intent_class = self.intent_classes[intent_idx]  
        
        inference_time = time.time() - start_time  
        
        return intent_class, confidence, inference_time  
    
    def predict_intent(self, audio, audio_text=None):  
        """  
        预测音频的意图  
        audio: 音频数据numpy数组  
        audio_text: 可选的音频文本转录(用于二级分类器)  
        """  
        # 预处理和特征提取  
        features, preprocess_time = self.preprocess_audio(audio)  
        
        # 一级快速分类  
        fast_intent, fast_confidence, fast_time = self.fast_inference(features)  
        
        # 如果一级分类器置信度高于阈值，直接返回结果  
        if fast_confidence >= self.fast_confidence_threshold:  
            total_time = preprocess_time + fast_time  
            return {  
                'intent': fast_intent,  
                'confidence': fast_confidence,  
                'path': 'fast',  
                'times': {  
                    'preprocess': preprocess_time,  
                    'inference': fast_time,  
                    'total': total_time  
                }  
            }  
        
        # 如果有二级分类器且提供了文本，使用二级分类器  
        if self.precise_model and audio_text:  
            precise_intent, precise_confidence, precise_time = self.precise_inference(audio_text)  
            total_time = preprocess_time + fast_time + precise_time  
            
            # 返回置信度更高的结果  
            if precise_confidence > fast_confidence:  
                return {  
                    'intent': precise_intent,  
                    'confidence': precise_confidence,  
                    'path': 'precise',  
                    'times': {  
                        'preprocess': preprocess_time,  
                        'fast_inference': fast_time,  
                        'precise_inference': precise_time,  
                        'total': total_time  
                    }  
                }  
        
        # 默认返回一级分类器结果  
        total_time = preprocess_time + fast_time  
        return {  
            'intent': fast_intent,  
            'confidence': fast_confidence,  
            'path': 'fast',  
            'times': {  
                'preprocess': preprocess_time,  
                'inference': fast_time,  
                'total': total_time  
            }  
        }  
    
    def process_audio_file(self, audio_path, transcript=None):  
        """处理音频文件并返回意图"""  
        # 加载音频文件  
        audio, sample_rate = sf.read(audio_path)  
        
        # 确保音频是单声道  
        if len(audio.shape) > 1:  
            audio = audio[:, 0]  
        
        # 预测意图  
        result = self.predict_intent(audio, transcript)  
        
        return result  
    
    def process_audio_stream(self, audio_stream, transcript=None):  
        """处理音频流并返回意图"""  
        result = self.predict_intent(audio_stream, transcript)  
        return result