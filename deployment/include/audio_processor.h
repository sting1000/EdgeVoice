/**
 * @file audio_processor.h
 * @brief 音频预处理和特征提取功能
 */

#ifndef EDGEVOICE_AUDIO_PROCESSOR_H
#define EDGEVOICE_AUDIO_PROCESSOR_H

#include <vector>
#include <string>
#include <cstdint>
#include <stdexcept>
#include <memory>
#include <cmath>
#include <complex>

namespace edgevoice {

/**
 * @brief WAV文件加载器
 * 简单结构体用于表示WAV文件内容
 */
struct WavData {
    std::vector<float> samples;  // 归一化到[-1, 1]的采样点
    int sample_rate;             // 采样率
    int bit_depth;               // 比特深度
    int channels;                // 通道数
};

/**
 * @brief 音频预处理器类
 * 
 * 该类负责音频信号的预处理，包括重采样、位深转换、
 * 预加重、静音移除和降噪等操作。
 */
class AudioPreprocessor {
public:
    /**
     * @brief 构造函数
     * 
     * @param sample_rate 原始采样率
     * @param target_sample_rate 目标采样率
     * @param vad_energy_threshold VAD能量阈值
     * @param vad_zcr_threshold VAD过零率阈值
     * @param frame_length_ms 帧长（毫秒）
     * @param frame_shift_ms 帧移（毫秒）
     */
    AudioPreprocessor(
        int sample_rate = 48000,
        int target_sample_rate = 16000,
        float vad_energy_threshold = 0.05f,
        float vad_zcr_threshold = 0.15f,
        int frame_length_ms = 25,
        int frame_shift_ms = 10
    );

    /**
     * @brief 析构函数
     */
    ~AudioPreprocessor();

    /**
     * @brief 重采样
     * 
     * @param audio 输入音频数据
     * @param orig_sample_rate 原始采样率
     * @return std::vector<float> 重采样后的音频数据
     */
    std::vector<float> resample(const std::vector<float>& audio, int orig_sample_rate);

    /**
     * @brief 预加重
     * 
     * @param audio 输入音频数据
     * @param coef 预加重系数
     * @return std::vector<float> 预加重后的音频数据
     */
    std::vector<float> preemphasis(const std::vector<float>& audio, float coef = 0.97f);

    /**
     * @brief 位深转换
     * 
     * @param audio 输入音频数据
     * @param bit_depth 原始位深
     * @return std::vector<float> 位深转换后的音频数据
     */
    std::vector<float> convertBitDepth(const std::vector<uint8_t>& audio, int bit_depth);

    /**
     * @brief 语音活动检测
     * 
     * @param audio 输入音频数据
     * @return std::vector<std::pair<int, int>> 语音片段起始和结束位置（帧索引）
     */
    std::vector<std::pair<int, int>> detectVoiceActivity(const std::vector<float>& audio);

    /**
     * @brief 静音移除
     * 
     * @param audio 输入音频数据
     * @param vad_segments VAD检测到的语音段
     * @return std::vector<float> 移除静音后的音频数据
     */
    std::vector<float> removeSilence(const std::vector<float>& audio, 
                                     const std::vector<std::pair<int, int>>& vad_segments);

    /**
     * @brief 降噪处理
     * 
     * @param audio 输入音频数据
     * @return std::vector<float> 降噪后的音频数据
     */
    std::vector<float> denoise(const std::vector<float>& audio);

    /**
     * @brief 应用所有预处理步骤
     * 
     * @param audio 输入音频数据
     * @param orig_sample_rate 原始采样率
     * @return std::vector<float> 预处理后的音频数据
     */
    std::vector<float> process(const std::vector<float>& audio, int orig_sample_rate);

private:
    int sample_rate_;            // 输入音频采样率
    int target_sample_rate_;     // 目标采样率
    float vad_energy_threshold_; // VAD能量阈值
    float vad_zcr_threshold_;    // VAD过零率阈值
    int frame_length_;           // 帧长（样本数）
    int frame_shift_;            // 帧移（样本数）
    int min_speech_frames_;      // 最小语音帧数
    int min_silence_frames_;     // 最小静音帧数
};

/**
 * @brief 特征提取器类
 * 
 * 该类负责从预处理后的音频数据中提取MFCC和相关特征。
 */
class FeatureExtractor {
public:
    using complex_d = std::complex<double>;

    /**
     * @brief 构造函数
     * 
     * @param sample_rate 采样率
     * @param n_mfcc MFCC特征数量
     * @param n_fft FFT窗口大小
     * @param hop_length 帧移（样本数）
     * @param context_frames 上下文帧数
     */
    FeatureExtractor(
        int sample_rate = 16000,
        int n_mfcc = 13,
        int n_fft = 512,
        int hop_length = 160,
        int context_frames = 5
    );

    /**
     * @brief 析构函数
     */
    ~FeatureExtractor();

    /**
     * @brief 提取MFCC特征
     * 
     * @param audio 输入音频数据
     * @return std::vector<std::vector<float>> MFCC特征（帧数 x 特征数）
     */
    std::vector<std::vector<float>> extractMFCC(const std::vector<float>& audio);

    /**
     * @brief 添加上下文帧信息
     * 
     * @param features 输入特征矩阵
     * @param context_size 上下文大小（单侧）
     * @return std::vector<std::vector<float>> 带上下文的特征矩阵
     */
    std::vector<std::vector<float>> addContext(const std::vector<std::vector<float>>& features, 
                                              int context_size);

    /**
     * @brief 提取所有声学特征
     * 
     * @param audio 输入音频数据
     * @return std::vector<std::vector<float>> 提取的特征矩阵
     */
    std::vector<std::vector<float>> extractFeatures(const std::vector<float>& audio);

private:
    int sample_rate_;        // 采样率
    int n_mfcc_;             // MFCC系数数量
    int n_fft_;              // FFT大小
    int hop_length_;         // 帧移
    int context_frames_;     // 上下文帧数
    int n_filters_;          // Mel滤波器组数量
    int n_dim_;              // FFT输出维度 = n_fft/2 + 1

    // 辅助MFCC提取方法
    std::vector<float> preEmphasis(const std::vector<float>& signal, float coef = 0.97f);
    void applyHammingWindow(std::vector<float>& frame);
    std::vector<float> getHammingWindow(bool periodic = true);
    void fft(std::vector<complex_d>& a, bool invert);
    std::vector<float> computePowerSpec(const std::vector<complex_d>& fft_result);
    std::vector<std::vector<float>> getMelFilterbank();
    double hzToMel(double hz);
    double melToHz(double mel);
    std::vector<double> linspace(double start, double end, int num);
};

/**
 * @brief 标准化音频长度
 * 
 * @param audio 输入音频数据
 * @param sample_rate 采样率
 * @param target_length 目标长度（秒）
 * @param min_length 最小长度（秒）
 * @return std::vector<float> 标准化后的音频数据
 */
std::vector<float> standardizeAudioLength(
    const std::vector<float>& audio,
    int sample_rate,
    float target_length = 5.0f,
    float min_length = 0.5f
);

/**
 * @brief 从WAV文件加载音频数据
 * 
 * @param file_path WAV文件路径
 * @param target_sample_rate 目标采样率（如果需要重采样），默认16000
 * @return std::pair<std::vector<float>, int> 音频数据和原始采样率
 * @throws std::runtime_error 如果文件不存在或格式不支持
 */
std::pair<std::vector<float>, int> loadWavFile(const std::string& file_path, int target_sample_rate = 16000);

/**
 * @brief 从WAV文件加载音频数据（原接口兼容函数）
 * 
 * @param filename WAV文件路径
 * @return WavData WAV文件数据结构
 * @throws std::runtime_error 如果文件不存在或格式不支持
 */
WavData loadWavFile(const std::string& filename);

} // namespace edgevoice

#endif // EDGEVOICE_AUDIO_PROCESSOR_H 