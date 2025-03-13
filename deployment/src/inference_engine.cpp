/**
 * @file inference_engine.cpp
 * @brief 华为HiAI推理引擎接口实现
 */

#include "../include/inference_engine.h"
#include <iostream>
#include <chrono>
#include <cstring>
#include <utility>
#include <algorithm>
#include <fstream>
#include <stdexcept>
#include <random>

// 注意：此处应包含HiAI的相关头文件
// #include <hiai_api.h>
// #include <hiai_model_manager.h>

namespace edgevoice {

InferenceEngine::InferenceEngine(const std::string& model_path, 
                                float fast_confidence_threshold,
                                const std::vector<std::string>& intent_classes,
                                int sample_rate,
                                float target_audio_length)
    : model_path_(model_path),
      fast_confidence_threshold_(fast_confidence_threshold),
      intent_classes_(intent_classes),
      sample_rate_(sample_rate),
      target_audio_length_(target_audio_length),
      model_manager_(nullptr),
      is_initialized_(false) {
    
    // 如果没有提供意图类别，使用默认类别
    if (intent_classes_.empty()) {
        intent_classes_ = {
            "CAPTURE_AND_DESCRIBE", "CAPTURE_REMEMBER", "CAPTURE_SCAN_QR", 
            "TAKE_PHOTO", "START_RECORDING", "STOP_RECORDING", 
            "GET_BATTERY_LEVEL", "OTHERS"
        };
    }
    
    // 创建音频预处理器和特征提取器
    audio_preprocessor_ = std::make_unique<AudioPreprocessor>(sample_rate_);
    feature_extractor_ = std::make_unique<FeatureExtractor>(sample_rate_);
    
    // 创建HiAI模型管理器
    model_manager_ = new HIAIModelManager();
}

InferenceEngine::~InferenceEngine() {
    // 释放模型资源
    if (model_manager_ != nullptr) {
        if (is_initialized_) {
            model_manager_->UnloadModel();
        }
        delete model_manager_;
        model_manager_ = nullptr;
    }
}

bool InferenceEngine::init() {
    try {
        // 检查模型文件是否存在
        std::ifstream model_file(model_path_, std::ios::binary);
        if (!model_file.is_open()) {
            std::cerr << "无法打开模型文件: " << model_path_ << std::endl;
            return false;
        }
        model_file.close();
        
        // 加载模型
        OH_NN_ReturnCode ret = model_manager_->LoadModelFromBuffer(model_path_);
        if (ret != OH_NN_SUCCESS) {
            std::cerr << "模型加载失败，错误码: " << ret << std::endl;
            return false;
        }
        
        std::cout << "加载模型: " << model_path_ << " 成功" << std::endl;
        is_initialized_ = true;
        
        return true;
    }
    catch (const std::exception& e) {
        std::cerr << "初始化推理引擎失败: " << e.what() << std::endl;
        return false;
    }
}

std::pair<std::vector<std::vector<float>>, float> InferenceEngine::preprocessAudio(
    const std::vector<float>& audio_data, 
    int sample_rate) {
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    // 1. 标准化音频长度
    std::vector<float> standardized_audio = standardizeAudioLength(
        audio_data, 
        sample_rate, 
        target_audio_length_
    );
    
    // 2. 应用预处理
    std::vector<float> processed_audio = audio_preprocessor_->process(standardized_audio, sample_rate);
    
    // 3. 提取特征
    std::vector<std::vector<float>> features = feature_extractor_->extractMFCC(processed_audio);
    
    // 4. 添加上下文信息（可选）
    int context_size = 5;  // 上下文帧数
    std::vector<std::vector<float>> context_features = feature_extractor_->addContext(features, context_size);
    
    auto end_time = std::chrono::high_resolution_clock::now();
    float processing_time = std::chrono::duration<float, std::milli>(end_time - start_time).count();
    
    return std::make_pair(context_features, processing_time);
}

std::vector<InputDataMessage> InferenceEngine::prepareInputData(
    const std::vector<std::vector<float>>& features) {
    
    // 计算特征总大小
    int num_frames = features.size();
    int num_features = (num_frames > 0) ? features[0].size() : 0;
    int total_size = num_frames * num_features;
    
    // 创建连续内存缓冲区
    uint8_t* data_buffer = new uint8_t[total_size * sizeof(float)];
    float* float_buffer = reinterpret_cast<float*>(data_buffer);
    
    // 复制特征数据到缓冲区
    int idx = 0;
    for (const auto& frame : features) {
        for (float val : frame) {
            float_buffer[idx++] = val;
        }
    }
    
    // 创建输入数据消息
    InputDataMessage input_message;
    input_message.data = data_buffer;
    input_message.dims = {1, static_cast<int32_t>(num_frames), static_cast<int32_t>(num_features), 1}; // NHWC格式
    input_message.imageFormat = HiAI_ImageFormat::HIAI_INVALID_FORMAT; // 不是图像数据
    input_message.databyteLen = sizeof(float);
    
    return {input_message};
}

std::pair<int, float> InferenceEngine::runInference(const std::vector<std::vector<float>>& features) {
    if (!is_initialized_) {
        throw std::runtime_error("推理引擎未初始化");
    }
    
    // 1. 准备输入数据
    std::vector<InputDataMessage> input_data = prepareInputData(features);
    
    // 2. 初始化输入输出张量
    OH_NN_ReturnCode ret = model_manager_->InitIOTensors(input_data, AippStatus::NOT_AIPP);
    if (ret != OH_NN_SUCCESS) {
        throw std::runtime_error("初始化输入输出张量失败，错误码: " + std::to_string(ret));
    }
    
    // 3. 执行推理
    ret = model_manager_->RunModel();
    if (ret != OH_NN_SUCCESS) {
        throw std::runtime_error("模型推理失败，错误码: " + std::to_string(ret));
    }
    
    // 4. 准备输出张量
    ret = model_manager_->PrepareOutputTensor();
    if (ret != OH_NN_SUCCESS) {
        throw std::runtime_error("准备输出张量失败，错误码: " + std::to_string(ret));
    }
    
    // 5. 获取推理结果
    std::vector<std::vector<float>> results = model_manager_->GetResult();
    if (results.empty() || results[0].empty()) {
        throw std::runtime_error("获取推理结果失败，结果为空");
    }
    
    // 找出最大值的索引和置信度
    std::vector<float>& output_probs = results[0];
    int max_idx = 0;
    float max_conf = output_probs[0];
    
    for (size_t i = 1; i < output_probs.size(); ++i) {
        if (output_probs[i] > max_conf) {
            max_conf = output_probs[i];
            max_idx = static_cast<int>(i);
        }
    }
    
    // 释放为输入数据分配的内存
    for (auto& input : input_data) {
        delete[] input.data;
    }
    
    return std::make_pair(max_idx, max_conf);
}

IntentResult InferenceEngine::predictIntent(const std::vector<float>& audio_data, int sample_rate) {
    IntentResult result;
    auto start_time = std::chrono::high_resolution_clock::now();
    
    try {
        // 1. 预处理音频并提取特征
        auto [features, preprocess_time] = preprocessAudio(audio_data, sample_rate);
        result.preprocessing_time = preprocess_time;
        
        // 2. 执行推理
        auto inference_start = std::chrono::high_resolution_clock::now();
        auto [intent_idx, confidence] = runInference(features);
        auto inference_end = std::chrono::high_resolution_clock::now();
        
        result.inference_time = std::chrono::duration<float, std::milli>(inference_end - inference_start).count();
        
        // 3. 设置结果
        if (intent_idx >= 0 && intent_idx < static_cast<int>(intent_classes_.size())) {
            result.intent_class = intent_classes_[intent_idx];
        } else {
            result.intent_class = "UNKNOWN";
        }
        
        result.confidence = confidence;
    }
    catch (const std::exception& e) {
        std::cerr << "预测过程中发生错误: " << e.what() << std::endl;
        result.intent_class = "ERROR";
        result.confidence = 0.0f;
    }
    
    auto end_time = std::chrono::high_resolution_clock::now();
    result.total_time = std::chrono::duration<float, std::milli>(end_time - start_time).count();
    
    return result;
}

IntentResult InferenceEngine::predictFromWavFile(const std::string& wav_file_path) {
    try {
        // 加载WAV文件
        WavData wav_data = loadWavFile(wav_file_path);
        
        // 预测意图
        return predictIntent(wav_data.samples, wav_data.sample_rate);
    }
    catch (const std::exception& e) {
        std::cerr << "从WAV文件预测意图失败: " << e.what() << std::endl;
        
        IntentResult result;
        result.intent_class = "ERROR";
        result.confidence = 0.0f;
        result.preprocessing_time = 0.0f;
        result.inference_time = 0.0f;
        result.total_time = 0.0f;
        
        return result;
    }
}

void InferenceEngine::setFastConfidenceThreshold(float threshold) {
    if (threshold < 0.0f || threshold > 1.0f) {
        throw std::invalid_argument("置信度阈值必须在0.0到1.0之间");
    }
    fast_confidence_threshold_ = threshold;
}

float InferenceEngine::getFastConfidenceThreshold() const {
    return fast_confidence_threshold_;
}

std::string InferenceEngine::getModelPath() const {
    return model_path_;
}

} // namespace edgevoice 