# inference.py  
import torch  
import numpy as np  
import soundfile as sf  
import time
import logging
import os
from config import *  
from audio_preprocessing import AudioPreprocessor  
from feature_extraction import FeatureExtractor  
from models.fast_classifier import FastIntentClassifier  
from models.precise_classifier import PreciseIntentClassifier  
from transformers import DistilBertTokenizer  
import torch.nn.functional as F

# 如果存在WeNetASR模型，导入它
try:
    from models.wenet_asr import WeNetASR
except ImportError:
    WeNetASR = None
    print("WeNetASR模型未安装，将使用标准NLU路径")

# 设置日志
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("InferenceEngine")

class IntentInferenceEngine:  
    def __init__(self, fast_model_path, precise_model_path=None, 
                 asr_model_path=None, asr_dict_path=None,
                 fast_confidence_threshold=FAST_CONFIDENCE_THRESHOLD,
                 asr_confidence_threshold=ASR_CONFIDENCE_THRESHOLD,
                 save_asr_results=False):  
        """  
        Initialize inference engine  
        
        Args:
            fast_model_path: First-level classifier model path  
            precise_model_path: Second-level classifier model path (optional)  
            asr_model_path: ASR model path (optional)
            asr_dict_path: ASR dictionary path (optional)
            fast_confidence_threshold: Confidence threshold for first-level classifier  
            asr_confidence_threshold: Confidence threshold for ASR results
            save_asr_results: Whether to save ASR intermediate results
        """  
        self.device = DEVICE  
        self.fast_confidence_threshold = fast_confidence_threshold  
        self.asr_confidence_threshold = asr_confidence_threshold
        self.save_asr_results = save_asr_results
        
        # Initialize preprocessor and feature extractor  
        self.preprocessor = AudioPreprocessor()  
        self.feature_extractor = FeatureExtractor()  
        
        # 加载ASR模型（如果提供）
        self.asr_model = None
        if asr_model_path and asr_dict_path and WeNetASR:
            try:
                logger.info(f"正在加载ASR模型: {asr_model_path}")
                self.asr_model = WeNetASR(
                    model_path=asr_model_path,
                    dict_path=asr_dict_path,
                    device=self.device,
                    save_results=save_asr_results,
                    result_dir=ASR_CACHE_DIR
                )
                logger.info("ASR模型加载完成")
            except Exception as e:
                logger.error(f"加载ASR模型失败: {e}")
                self.asr_model = None
        
        # Load first-level classifier  
        print("Loading first-level fast classifier...")  
        self.fast_model = self._load_fast_model(fast_model_path)  
        
        # Load second-level classifier if provided  
        self.precise_model = None
        if precise_model_path:  
            print("Loading second-level precise classifier...")  
            self.precise_model = self._load_precise_model(precise_model_path)  
            
            # Load DistilBERT tokenizer  
            print("Loading DistilBERT model from local path:", DISTILBERT_MODEL_PATH)  
            try:  
                from transformers import DistilBertModel, DistilBertConfig  
                
                # Load from local path or download  
                config = DistilBertConfig.from_pretrained(DISTILBERT_MODEL_PATH, 
                                                         hidden_size=PRECISE_MODEL_HIDDEN_SIZE,
                                                         num_hidden_layers=3)  
                self.bert_model = DistilBertModel.from_pretrained(DISTILBERT_MODEL_PATH,
                                                                 config=config)  
                print("DistilBERT model loaded from local path")  
            except Exception as e:  
                print(f"Error loading DistilBERT model: {e}")  
            
            # Load tokenizer  
            try:  
                self.tokenizer = DistilBertTokenizer.from_pretrained(DISTILBERT_MODEL_PATH)  
                print("DistilBERT tokenizer loaded from local path:", DISTILBERT_MODEL_PATH)  
            except:  
                print("Failed to load tokenizer from local path, downloading from Hugging Face")  
                self.tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        
        # 类别名称  
        self.intent_classes = INTENT_CLASSES
        
    def _load_fast_model(self, model_path):  
        """Load first-level fast classifier model"""  
        try:
            # 尝试加载包含模型和标签的字典
            checkpoint = torch.load(model_path, map_location=self.device)
            
            # 检查是否是新格式（包含model_state_dict和intent_labels）
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint and 'intent_labels' in checkpoint:
                # 这是新格式，包含标签映射
                self.intent_classes = checkpoint['intent_labels']
                print(f"从模型文件加载意图标签: {self.intent_classes}")
                
                # 如果有input_size，读取它
                input_size = checkpoint.get('input_size', 39)
                
                # 创建与特征维度匹配的模型
                model = FastIntentClassifier(input_size=input_size, num_classes=len(self.intent_classes))
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                # 旧格式，只包含模型权重，使用配置中的标签
                print("使用旧格式加载模型，标签从配置文件获取")
                input_size = 39 # 默认特征维度（MFCC+Delta+Delta2，共39维）
                model = FastIntentClassifier(input_size=input_size)
                
                try:
                    # 尝试加载模型权重
                    model.load_state_dict(checkpoint)
                except Exception as e:
                    print(f"加载模型时出现错误：{e}")
                    print(f"尝试使用兼容模式加载...")
                    # 对于从旧结构迁移到新Conformer结构的模型，使用严格=False
                    model.load_state_dict(checkpoint, strict=False)
                
                self.intent_classes = INTENT_CLASSES
        except Exception as e:
            print(f"加载模型失败：{e}")
            print("使用默认标签和模型结构")
            input_size = 39
            model = FastIntentClassifier(input_size=input_size)
            self.intent_classes = INTENT_CLASSES
        
        model = model.to(self.device)
        model.eval()
        
        print(f"意图类别标签: {self.intent_classes}")
        return model  
    
    def _load_precise_model(self, model_path):  
        """Load second-level precise classifier model"""  
        try:
            # 尝试加载包含模型和标签的字典
            checkpoint = torch.load(model_path, map_location=self.device)
            
            # 检查是否是新格式（包含model_state_dict和intent_labels）
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint and 'intent_labels' in checkpoint:
                # 这是新格式，包含标签映射
                # 注意：我们不覆盖self.intent_classes，因为在这个架构中，两个模型应该共享相同的标签映射
                precise_labels = checkpoint['intent_labels']
                
                # 检查是否与快速分类器标签一致
                if set(precise_labels) != set(self.intent_classes):
                    print(f"警告: 精确分类器标签与快速分类器标签不一致！")
                    print(f"快速分类器标签: {self.intent_classes}")
                    print(f"精确分类器标签: {precise_labels}")
                
                model = PreciseIntentClassifier(num_classes=len(precise_labels))
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                # 旧格式，只包含模型权重
                model = PreciseIntentClassifier()
                model.load_state_dict(checkpoint)
        except Exception as e:
            print(f"加载精确分类器失败：{e}")
            print("使用默认模型结构")
            model = PreciseIntentClassifier()
        
        model = model.to(self.device)
        model.eval()
        return model  
    
    def preprocess_audio(self, audio):  
        """Preprocess audio and extract features"""  
        start_time = time.time()  
        
        # Apply preprocessing  
        processed_audio = self.preprocessor.process(audio)  
        
        # Extract features  
        features = self.feature_extractor.extract_features(processed_audio)  
        
        preprocess_time = time.time() - start_time  
        
        return features, preprocess_time  
    
    def fast_inference(self, features):  
        """Perform fast inference using first-level classifier"""  
        start_time = time.time()  
        
        # Convert to torch tensor and add batch dimension  
        features_tensor = torch.FloatTensor(features).unsqueeze(0).to(self.device)  
        
        # Inference  
        with torch.no_grad():  
            intent_idx, confidence = self.fast_model.predict(features_tensor)  
        
        # Get predicted class and confidence  
        intent_idx = intent_idx.item()  
        confidence = confidence.item()  
        intent_class = self.intent_classes[intent_idx]  
        
        inference_time = time.time() - start_time  
        
        return intent_class, confidence, inference_time
    
    def precise_inference(self, audio_text):  
        """Perform precise inference using second-level classifier"""  
        if self.precise_model is None:  
            return None, 0, 0  
        
        start_time = time.time()  
        
        # Tokenize text input  
        encoding = self.tokenizer(  
            audio_text,  
            max_length=128,  
            padding='max_length',  
            truncation=True,  
            return_tensors='pt'  
        ).to(self.device)  
        
        # Inference  
        with torch.no_grad():  
            input_ids = encoding['input_ids']  
            attention_mask = encoding['attention_mask']  
            outputs = self.precise_model(input_ids, attention_mask)  
            logits = outputs.logits  
            probs = F.softmax(logits, dim=1)  
            confidence, predicted = torch.max(probs, dim=1)  
        
        # Get predicted class and confidence  
        intent_idx = predicted.item()  
        confidence = confidence.item()  
        intent_class = self.intent_classes[intent_idx]  
        
        inference_time = time.time() - start_time  
        
        return intent_class, confidence, inference_time
    
    def asr_inference(self, audio, sample_rate=TARGET_SAMPLE_RATE, audio_id=None):
        """
        使用ASR模型进行语音识别
        
        Args:
            audio: 音频数据
            sample_rate: 采样率
            audio_id: 音频ID，用于保存结果
            
        Returns:
            (转写文本, 置信度, 推理时间)
        """
        if self.asr_model is None:
            return None, 0, 0
        
        start_time = time.time()
        
        # 执行语音识别
        text, confidence = self.asr_model.transcribe(audio, sample_rate)
        
        # 计算耗时
        inference_time = time.time() - start_time
        
        # 保存结果（如果启用）
        if self.save_asr_results and audio_id:
            metadata = {
                "time_ms": inference_time * 1000,
                "sample_rate": sample_rate,
                "audio_length": len(audio) / sample_rate
            }
            self.asr_model.save_result(audio_id, text, confidence, metadata)
        
        return text, confidence, inference_time
    
    def predict(self, features, confidence_threshold=None):
        """
        Predict directly based on features, supporting both fast and precise classifiers
        
        Args:
            features: Features, can be tensor (for fast classifier) or dict (for precise classifier)
            confidence_threshold: Confidence threshold, if None uses default value
            
        Returns:
            (predicted class index, confidence, whether fast classifier was used)
        """
        threshold = confidence_threshold if confidence_threshold is not None else self.fast_confidence_threshold
        
        try:
            # Determine which classifier to use based on feature type
            if isinstance(features, torch.Tensor):
                # Use fast classifier
                predicted_class, confidence = self.fast_model.predict(features)
                intent_idx = predicted_class.item()
                confidence_value = confidence.item()
                
                # If confidence above threshold, return result directly
                if confidence_value >= threshold:
                    return intent_idx, confidence_value, True
                
                # If no precise classifier, still return fast classifier result
                if self.precise_model is None:
                    return intent_idx, confidence_value, True
                    
                # Otherwise will use precise classifier in next step
                
            elif isinstance(features, dict) and 'input_ids' in features and 'attention_mask' in features:
                # Use precise classifier directly
                input_ids = features['input_ids']
                attention_mask = features['attention_mask']
                
                outputs = self.precise_model(input_ids, attention_mask)
                logits = outputs.logits
                probs = F.softmax(logits, dim=1)
                confidence, predicted = torch.max(probs, dim=1)
                
                return predicted.item(), confidence.item(), False
                
            else:
                # Unsupported feature type
                print(f"Unsupported feature type: {type(features)}")
                return 0, 0.0, True  # Return default values
                
        except Exception as e:
            print(f"Error during prediction: {str(e)}")
            return 0, 0.0, True  # Return default values on error
    
    def predict_intent(self, audio, audio_text=None, use_asr=True, audio_id=None):  
        """
        Predict intent from audio data
        
        Args:
            audio: Audio data (numpy array)
            audio_text: Audio transcription text (optional, for secondary classification)
            use_asr: Whether to use ASR for transcription if audio_text is not provided
            audio_id: 音频ID，用于保存ASR结果
            
        Returns:
            包含以下字段的字典:
                - intent: 预测的意图类别
                - confidence: 置信度
                - preprocessing_time: 预处理时间(ms)
                - inference_time: 推理时间(ms)
                - path: 使用的路径 ("fast", "precise", "asr+nlu")
                - transcription: 转写文本 (如果使用了ASR)
                - asr_confidence: ASR置信度 (如果使用了ASR)
                - asr_time: ASR耗时 (如果使用了ASR)
        """
        # 初始化结果字典
        result = {
            "intent": None,
            "confidence": 0.0,
            "preprocessing_time": 0.0,
            "inference_time": 0.0,
            "path": "unknown",
            "transcription": None,
            "asr_confidence": None,
            "asr_time": None
        }
        
        start_time = time.time()
        
        # Preprocess audio and extract features
        features, preprocess_time = self.preprocess_audio(audio)
        result["preprocessing_time"] = preprocess_time * 1000  # 转换为毫秒
        
        # 记录处理开始时间
        processing_start = time.time()
        
        # 首先尝试快速分类器
        intent_class, confidence, inference_time = self.fast_inference(features)
        
        # 更新结果字典
        result["intent"] = intent_class
        result["confidence"] = confidence
        result["inference_time"] = inference_time * 1000  # 转换为毫秒
        result["path"] = "fast"
        
        # 如果置信度足够高，直接返回结果
        if confidence >= self.fast_confidence_threshold:
            result["total_time"] = (time.time() - start_time) * 1000  # 总耗时(毫秒)
            return result
        
        # 如果有提供转写文本，使用精确分类器
        if audio_text:
            # 使用提供的文本进行精确分类
            precise_intent, precise_confidence, precise_time = self.precise_inference(audio_text)
            
            # 更新结果字典
            result["intent"] = precise_intent
            result["confidence"] = precise_confidence
            result["inference_time"] = precise_time * 1000  # 转换为毫秒
            result["path"] = "precise"
            result["transcription"] = audio_text
            
            # 记录总耗时
            result["total_time"] = (time.time() - start_time) * 1000  # 毫秒
            return result
            
        # 如果未提供文本且启用ASR，尝试使用ASR+NLU路径
        if use_asr and self.asr_model and self.precise_model:
            # 使用ASR模型进行转写
            transcription, asr_confidence, asr_time = self.asr_inference(audio, TARGET_SAMPLE_RATE, audio_id)
            
            # 更新ASR相关结果字段
            result["transcription"] = transcription
            result["asr_confidence"] = asr_confidence
            result["asr_time"] = asr_time * 1000  # 转换为毫秒
            
            # 如果ASR成功且置信度足够高，使用精确分类器
            if transcription and asr_confidence >= self.asr_confidence_threshold:
                # 使用ASR转写文本进行精确分类
                precise_intent, precise_confidence, precise_time = self.precise_inference(transcription)
                
                # 更新结果字典
                result["intent"] = precise_intent
                result["confidence"] = precise_confidence
                result["inference_time"] = precise_time * 1000  # 转换为毫秒
                result["path"] = "asr+nlu"
                
                # 记录总耗时
                result["total_time"] = (time.time() - start_time) * 1000  # 毫秒
                return result
        
        # 如果所有其他选项都不可用，返回快速分类器结果
        result["total_time"] = (time.time() - start_time) * 1000  # 毫秒
        return result
    
    def process_audio_file(self, audio_path, transcript=None, use_asr=True):  
        """
        Process audio file and predict intent
        
        Args:
            audio_path: Path to audio file
            transcript: Optional transcript for precise classification
            use_asr: Whether to use ASR if transcript is not provided
            
        Returns:
            Prediction result
        """
        # Load audio file
        audio, sr = sf.read(audio_path)
        
        # If stereo, convert to mono
        if len(audio.shape) > 1:
            audio = np.mean(audio, axis=1)
        
        # Resample if needed
        if sr != TARGET_SAMPLE_RATE:
            import librosa
            audio = librosa.resample(audio, orig_sr=sr, target_sr=TARGET_SAMPLE_RATE)
            sr = TARGET_SAMPLE_RATE
        
        # 获取音频ID（从文件名）
        audio_id = os.path.basename(audio_path)
        
        # Predict intent
        return self.predict_intent(audio, transcript, use_asr, audio_id)
    
    def process_audio_stream(self, audio_stream, transcript=None, use_asr=True, audio_id=None):  
        """
        Process audio stream data and predict intent
        
        Args:
            audio_stream: Audio data as numpy array
            transcript: Optional transcript for precise classification
            use_asr: Whether to use ASR if transcript is not provided
            audio_id: 音频ID，用于保存ASR结果
            
        Returns:
            Prediction result
        """
        # 如果未提供音频ID，使用时间戳
        if audio_id is None:
            audio_id = f"stream_{int(time.time() * 1000)}"
            
        # Predict intent directly from audio stream data
        return self.predict_intent(audio_stream, transcript, use_asr, audio_id)