/**
 * @file inference_engine.h
 * @brief 华为HiAI推理引擎接口
 */

#ifndef EDGEVOICE_INFERENCE_ENGINE_H
#define EDGEVOICE_INFERENCE_ENGINE_H

#include <string>
#include <vector>
#include <memory>
#include <chrono>
#include <unordered_map>
#include <utility>

#include "audio_processor.h"

namespace edgevoice {

/**
 * @brief 意图识别结果
 */
struct IntentResult {
    std::string intent_class;    ///< 识别出的意图类别
    float confidence;            ///< 置信度
    float preprocessing_time;    ///< 预处理时间（毫秒）
    float inference_time;        ///< 推理时间（毫秒）
    float total_time;            ///< 总时间（毫秒）
};

/**
 * @brief 华为HiAI推理引擎
 * 
 * 该类负责加载OMC模型并进行推理，返回意图识别结果
 */
class InferenceEngine {
public:
    /**
     * @brief 构造函数
     * 
     * @param model_path OMC模型路径
     * @param fast_confidence_threshold 快速模型置信度阈值
     * @param intent_classes 意图类别列表
     * @param sample_rate 音频采样率
     * @param target_audio_length 目标音频长度（秒）
     */
    InferenceEngine(const std::string& model_path, 
                    float fast_confidence_threshold = 0.7f,
                    const std::vector<std::string>& intent_classes = {},
                    int sample_rate = 16000,
                    float target_audio_length = 1.0f);

    /**
     * @brief 析构函数
     */
    ~InferenceEngine();

    /**
     * @brief 初始化推理引擎
     * 
     * @return bool 初始化是否成功
     */
    bool init();

    /**
     * @brief 预处理音频
     * 
     * @param audio_data 原始音频数据
     * @param sample_rate 音频采样率
     * @return std::pair<std::vector<std::vector<float>>, float> 特征矩阵和处理时间（毫秒）
     */
    std::pair<std::vector<std::vector<float>>, float> preprocessAudio(
        const std::vector<float>& audio_data, 
        int sample_rate
    );

    /**
     * @brief 从音频数据预测意图
     * 
     * @param audio_data 音频数据
     * @param sample_rate 音频采样率
     * @return IntentResult 意图识别结果
     */
    IntentResult predictIntent(const std::vector<float>& audio_data, int sample_rate);

    /**
     * @brief 从WAV文件预测意图
     * 
     * @param wav_file_path WAV文件路径
     * @return IntentResult 意图识别结果
     */
    IntentResult predictFromWavFile(const std::string& wav_file_path);

    /**
     * @brief 设置快速模型置信度阈值
     * 
     * @param threshold 新的阈值
     */
    void setFastConfidenceThreshold(float threshold);

    /**
     * @brief 获取当前快速模型置信度阈值
     * 
     * @return 当前阈值
     */
    float getFastConfidenceThreshold() const;

    /**
     * @brief 获取模型路径
     * 
     * @return 模型路径
     */
    std::string getModelPath() const;

private:
    /**
     * @brief 执行推理
     * 
     * @param features 特征矩阵
     * @return std::pair<int, float> 意图索引和置信度
     */
    std::pair<int, float> runInference(const std::vector<std::vector<float>>& features);

    std::string model_path_;
    float fast_confidence_threshold_;
    std::vector<std::string> intent_classes_;

    std::unique_ptr<AudioPreprocessor> preprocessor_;
    std::unique_ptr<FeatureExtractor> feature_extractor_;

    // HiAI相关的成员变量（具体实现时根据HiAI API调整）
    void* model_handle_; // 实际使用时替换为HiAI的模型句柄类型
    bool is_initialized_;
};

} // namespace edgevoice

#endif // INFERENCE_ENGINE_H 