/**
 * @file audio_processor.cpp
 * @brief 音频预处理和特征提取功能的实现
 */

#include "../include/audio_processor.h"
#include <cmath>
#include <algorithm>
#include <fstream>
#include <iostream>
#include <numeric>
#include <cstring>
#include <random>

// 用于WAV文件解析的简单结构体
struct WavHeader {
    char riff[4];                // RIFF标识
    uint32_t file_size;          // 文件大小
    char wave[4];                // WAVE标识
    char fmt[4];                 // fmt子块
    uint32_t fmt_size;           // fmt子块大小
    uint16_t audio_format;       // 音频格式 (1 = PCM)
    uint16_t num_channels;       // 声道数
    uint32_t sample_rate;        // 采样率
    uint32_t byte_rate;          // 字节率
    uint16_t block_align;        // 块对齐
    uint16_t bits_per_sample;    // 每个样本的位数
    // data子块可能不直接跟在fmt子块后，需要搜索
};

namespace edgevoice {

//-----------------------------------------------------------------------------
// AudioPreprocessor实现
//-----------------------------------------------------------------------------
AudioPreprocessor::AudioPreprocessor(
    int sample_rate,
    int target_sample_rate,
    float vad_energy_threshold,
    float vad_zcr_threshold,
    int frame_length_ms,
    int frame_shift_ms
) : sample_rate_(sample_rate),
    target_sample_rate_(target_sample_rate),
    vad_energy_threshold_(vad_energy_threshold),
    vad_zcr_threshold_(vad_zcr_threshold),
    frame_length_ms_(frame_length_ms),
    frame_shift_ms_(frame_shift_ms),
    frame_length_(static_cast<int>(frame_length_ms * target_sample_rate / 1000.0f)),
    frame_shift_(static_cast<int>(frame_shift_ms * target_sample_rate / 1000.0f)),
    min_speech_frames_(static_cast<int>(100 / frame_shift_ms)),   // 100ms 最小语音段
    min_silence_frames_(static_cast<int>(300 / frame_shift_ms))   // 300ms 最小静音段
{
}

AudioPreprocessor::~AudioPreprocessor() {
    // 析构函数，无需特殊操作
}

std::vector<float> AudioPreprocessor::resample(const std::vector<float>& audio, int orig_sample_rate) {
    if (orig_sample_rate == target_sample_rate_) {
        return audio;  // 不需要重采样
    }
    
    // 简单线性插值重采样
    // 注意：在实际部署中，应使用更高质量的重采样方法
    double ratio = static_cast<double>(target_sample_rate_) / orig_sample_rate;
    int output_size = static_cast<int>(audio.size() * ratio);
    std::vector<float> resampled(output_size);
    
    for (int i = 0; i < output_size; ++i) {
        double src_idx = i / ratio;
        int src_idx_floor = static_cast<int>(src_idx);
        int src_idx_ceil = std::min(src_idx_floor + 1, static_cast<int>(audio.size()) - 1);
        double t = src_idx - src_idx_floor;
        
        resampled[i] = (1.0 - t) * audio[src_idx_floor] + t * audio[src_idx_ceil];
    }
    
    return resampled;
}

std::vector<float> AudioPreprocessor::preemphasis(const std::vector<float>& audio, float coef) {
    if (audio.empty()) {
        return {};
    }
    
    std::vector<float> emphasized(audio.size());
    emphasized[0] = audio[0];
    
    for (size_t i = 1; i < audio.size(); ++i) {
        emphasized[i] = audio[i] - coef * audio[i - 1];
    }
    
    return emphasized;
}

std::vector<float> AudioPreprocessor::convertBitDepth(const std::vector<uint8_t>& audio, int bit_depth) {
    std::vector<float> float_audio;
    float_audio.reserve(audio.size() / (bit_depth / 8));
    
    if (bit_depth == 16) {
        // 16位PCM (每样本2字节)
        for (size_t i = 0; i < audio.size(); i += 2) {
            if (i + 1 < audio.size()) {
                int16_t sample = static_cast<int16_t>(audio[i] | (audio[i + 1] << 8));
                float_audio.push_back(sample / 32768.0f);
            }
        }
    } else if (bit_depth == 24) {
        // 24位PCM (每样本3字节)
        for (size_t i = 0; i < audio.size(); i += 3) {
            if (i + 2 < audio.size()) {
                int32_t sample = static_cast<int32_t>(audio[i] | 
                                                     (audio[i + 1] << 8) | 
                                                     (audio[i + 2] << 16));
                // 符号扩展
                if (sample & 0x800000) {
                    sample |= 0xFF000000;
                }
                float_audio.push_back(sample / 8388608.0f); // 2^23
            }
        }
    } else if (bit_depth == 32) {
        // 32位PCM (每样本4字节)
        for (size_t i = 0; i < audio.size(); i += 4) {
            if (i + 3 < audio.size()) {
                int32_t sample = static_cast<int32_t>(audio[i] | 
                                                     (audio[i + 1] << 8) | 
                                                     (audio[i + 2] << 16) | 
                                                     (audio[i + 3] << 24));
                float_audio.push_back(sample / 2147483648.0f); // 2^31
            }
        }
    } else {
        throw std::runtime_error("不支持的位深: " + std::to_string(bit_depth));
    }
    
    return float_audio;
}

std::vector<std::pair<int, int>> AudioPreprocessor::detectVoiceActivity(const std::vector<float>& audio) {
    std::vector<std::pair<int, int>> vad_segments;
    
    if (audio.empty()) {
        return vad_segments;
    }
    
    // 计算帧数
    int num_frames = static_cast<int>((audio.size() - frame_length_) / frame_shift_) + 1;
    
    // 存储每帧的能量和过零率
    std::vector<float> energies(num_frames, 0);
    std::vector<float> zcrs(num_frames, 0);
    
    // 计算每帧的能量和过零率
    for (int i = 0; i < num_frames; ++i) {
        int start = i * frame_shift_;
        int end = std::min(start + frame_length_, static_cast<int>(audio.size()));
        
        // 计算能量
        float energy = 0;
        for (int j = start; j < end; ++j) {
            energy += audio[j] * audio[j];
        }
        energy /= (end - start);
        energies[i] = energy;
        
        // 计算过零率
        int zcr = 0;
        for (int j = start + 1; j < end; ++j) {
            if ((audio[j] >= 0 && audio[j-1] < 0) || (audio[j] < 0 && audio[j-1] >= 0)) {
                zcr++;
            }
        }
        zcrs[i] = static_cast<float>(zcr) / (end - start - 1);
    }
    
    // 根据能量和过零率判断语音段
    bool in_speech = false;
    int speech_start = 0;
    int min_speech_frames = static_cast<int>(0.1 * sample_rate_ / frame_shift_); // 最小语音段长度：100ms
    
    for (int i = 0; i < num_frames; ++i) {
        bool is_speech = (energies[i] > vad_energy_threshold_) || 
                         (zcrs[i] > vad_zcr_threshold_ && energies[i] > vad_energy_threshold_ * 0.5);
        
        if (is_speech && !in_speech) {
            speech_start = i;
            in_speech = true;
        } else if (!is_speech && in_speech) {
            if (i - speech_start >= min_speech_frames) {
                vad_segments.push_back(std::make_pair(speech_start, i));
            }
            in_speech = false;
        }
    }
    
    // 处理最后一个语音段
    if (in_speech && num_frames - speech_start >= min_speech_frames) {
        vad_segments.push_back(std::make_pair(speech_start, num_frames));
    }
    
    return vad_segments;
}

std::vector<float> AudioPreprocessor::removeSilence(const std::vector<float>& audio, 
                                                   const std::vector<std::pair<int, int>>& vad_segments) {
    if (vad_segments.empty() || audio.empty()) {
        return audio;  // 没有检测到语音段或输入为空，返回原始音频
    }
    
    // 根据语音活动段构建无静音的音频
    std::vector<float> speech_audio;
    for (const auto& segment : vad_segments) {
        int start = segment.first * frame_shift_;
        int end = std::min(segment.second * frame_shift_ + frame_length_, static_cast<int>(audio.size()));
        
        speech_audio.insert(speech_audio.end(), audio.begin() + start, audio.begin() + end);
    }
    
    return speech_audio;
}

std::vector<float> AudioPreprocessor::denoise(const std::vector<float>& audio) {
    // 简单的降噪实现
    // 这里使用一个非常简单的降噪方法，在实际项目中应该使用更复杂的算法
    std::vector<float> denoised = audio;
    
    // 简单的阈值降噪
    float noise_threshold = 0.01f;
    for (auto& sample : denoised) {
        if (std::abs(sample) < noise_threshold) {
            sample = 0.0f;
        }
    }
    
    return denoised;
}

std::vector<float> AudioPreprocessor::process(const std::vector<float>& audio, int orig_sample_rate) {
    // 完整的预处理流程
    
    // 1. 重采样
    std::vector<float> resampled = resample(audio, orig_sample_rate);
    
    // 2. 预加重
    std::vector<float> emphasized = preemphasis(resampled);
    
    // 3. 语音活动检测
    std::vector<std::pair<int, int>> vad_segments = detectVoiceActivity(emphasized);
    
    // 4. 去除静音
    std::vector<float> speech_only = removeSilence(emphasized, vad_segments);
    
    // 5. 降噪
    std::vector<float> denoised = denoise(speech_only);
    
    return denoised;
}

//-----------------------------------------------------------------------------
// FeatureExtractor实现
//-----------------------------------------------------------------------------
FeatureExtractor::FeatureExtractor(
    int sample_rate,
    int n_mfcc,
    int n_fft,
    int hop_length,
    int context_frames
) : sample_rate_(sample_rate),
    n_mfcc_(n_mfcc),
    n_fft_(n_fft),
    hop_length_(hop_length),
    context_frames_(context_frames)
{
}

FeatureExtractor::~FeatureExtractor() {
    // 析构函数，无需特殊操作
}

// 注：实际实现中，应该使用专业的MFCC提取库，如KFR、Eigen或HiAI提供的API
// 以下是简化实现，仅展示基本思路
std::vector<std::vector<float>> FeatureExtractor::extractMFCC(const std::vector<float>& audio) {
    if (audio.empty()) {
        return {};
    }
    
    // 计算帧数
    int num_frames = static_cast<int>((audio.size() - n_fft_) / hop_length_) + 1;
    
    // 在实际部署中，应使用真实的MFCC提取算法
    // 这里使用随机值模拟MFCC特征，仅用于演示
    std::vector<std::vector<float>> mfcc_features(num_frames, std::vector<float>(n_mfcc_));
    
    // 创建随机数生成器
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    
    // 生成随机MFCC特征
    for (int i = 0; i < num_frames; ++i) {
        for (int j = 0; j < n_mfcc_; ++j) {
            mfcc_features[i][j] = dist(gen);
        }
    }
    
    // 注意：在实际部署中，此处应使用真实的MFCC提取算法
    // 例如：
    // 1. 分帧
    // 2. 加窗（汉明窗）
    // 3. 快速傅里叶变换（FFT）
    // 4. 梅尔滤波器组
    // 5. 对数运算
    // 6. 离散余弦变换（DCT）
    
    return mfcc_features;
}

std::vector<std::vector<float>> FeatureExtractor::addContext(
    const std::vector<std::vector<float>>& features, 
    int context_size) {
    
    if (features.empty()) {
        return {};
    }
    
    int num_frames = features.size();
    int feature_dim = features[0].size();
    int context_feature_dim = feature_dim * (2 * context_size + 1);
    
    std::vector<std::vector<float>> context_features(num_frames, std::vector<float>(context_feature_dim));
    
    for (int i = 0; i < num_frames; ++i) {
        int feature_index = 0;
        
        // 添加前文上下文
        for (int j = -context_size; j <= context_size; ++j) {
            int frame_idx = i + j;
            
            // 边界处理
            if (frame_idx < 0) {
                frame_idx = 0;
            } else if (frame_idx >= num_frames) {
                frame_idx = num_frames - 1;
            }
            
            // 复制特征
            for (int k = 0; k < feature_dim; ++k) {
                context_features[i][feature_index++] = features[frame_idx][k];
            }
        }
    }
    
    return context_features;
}

std::vector<std::vector<float>> FeatureExtractor::extractFeatures(const std::vector<float>& audio) {
    // 提取基础MFCC特征
    std::vector<std::vector<float>> mfcc_features = extractMFCC(audio);
    
    // 为了与现有模型兼容，此处不添加上下文信息
    // 如果需要上下文特征，可以使用 addContext 方法
    
    return mfcc_features;
}

//-----------------------------------------------------------------------------
// 辅助函数实现
//-----------------------------------------------------------------------------
std::vector<float> standardizeAudioLength(
    const std::vector<float>& audio,
    int sample_rate,
    float target_length,
    float min_length) {
    
    int target_samples = static_cast<int>(target_length * sample_rate);
    int current_samples = audio.size();
    
    // 如果音频过短，通过重复填充
    if (current_samples < static_cast<int>(min_length * sample_rate)) {
        int repeats = static_cast<int>(std::ceil((min_length * sample_rate) / current_samples));
        std::vector<float> repeated;
        repeated.reserve(current_samples * repeats);
        
        for (int i = 0; i < repeats; ++i) {
            repeated.insert(repeated.end(), audio.begin(), audio.end());
        }
        
        std::vector<float> padded_audio(target_samples, 0.0f);
        int to_copy = std::min(static_cast<int>(repeated.size()), target_samples);
        int start = (target_samples - to_copy) / 2;
        
        std::copy(repeated.begin(), repeated.begin() + to_copy, padded_audio.begin() + start);
        return padded_audio;
    }
    
    // 如果音频长度合适，直接返回
    if (current_samples == target_samples) {
        return audio;
    }
    
    // 如果音频过长，进行裁剪（保留中间部分）
    if (current_samples > target_samples) {
        std::vector<float> cropped(target_samples);
        int start = (current_samples - target_samples) / 2;
        std::copy(audio.begin() + start, audio.begin() + start + target_samples, cropped.begin());
        return cropped;
    }
    
    // 如果音频过短，在两端填充0
    std::vector<float> padded_audio(target_samples, 0.0f);
    int start = (target_samples - current_samples) / 2;
    std::copy(audio.begin(), audio.end(), padded_audio.begin() + start);
    
    return padded_audio;
}

std::pair<std::vector<float>, int> loadWavFile(const std::string& file_path, int target_sample_rate) {
    std::ifstream file(file_path, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("无法打开WAV文件: " + file_path);
    }

    WavHeader header;
    file.read(reinterpret_cast<char*>(&header), sizeof(WavHeader));

    // 检查是否有效的WAV文件
    if (std::strncmp(header.riff, "RIFF", 4) != 0 ||
        std::strncmp(header.wave, "WAVE", 4) != 0 ||
        std::strncmp(header.fmt, "fmt ", 4) != 0) {
        throw std::runtime_error("无效的WAV文件格式");
    }

    // 寻找data子块
    char chunk_id[4];
    uint32_t chunk_size;
    bool found_data = false;

    // 跳过fmt子块的剩余部分
    file.seekg(header.fmt_size - 16 + 20, std::ios::beg); // 16 = 已读取的fmt字段, 20 = 已读取的其他字段

    while (file.read(chunk_id, 4) && file.read(reinterpret_cast<char*>(&chunk_size), 4)) {
        if (std::strncmp(chunk_id, "data", 4) == 0) {
            found_data = true;
            break;
        }
        file.seekg(chunk_size, std::ios::cur);
    }

    if (!found_data) {
        throw std::runtime_error("找不到WAV文件的data子块");
    }

    // 分配内存并读取音频数据
    int bytes_per_sample = header.bits_per_sample / 8;
    int num_samples = chunk_size / (bytes_per_sample * header.num_channels);
    std::vector<float> audio_data(num_samples);

    if (header.bits_per_sample == 16) {
        std::vector<int16_t> raw_data(num_samples * header.num_channels);
        file.read(reinterpret_cast<char*>(raw_data.data()), chunk_size);

        // 转换为float并处理多声道（取平均）
        for (int i = 0; i < num_samples; ++i) {
            float sum = 0.0f;
            for (int j = 0; j < header.num_channels; ++j) {
                sum += raw_data[i * header.num_channels + j] / 32768.0f;
            }
            audio_data[i] = sum / header.num_channels;
        }
    } else if (header.bits_per_sample == 24) {
        // 24位音频需要特殊处理
        std::vector<uint8_t> raw_data(chunk_size);
        file.read(reinterpret_cast<char*>(raw_data.data()), chunk_size);

        // 转换为float并处理多声道（取平均）
        for (int i = 0; i < num_samples; ++i) {
            float sum = 0.0f;
            for (int j = 0; j < header.num_channels; ++j) {
                int idx = (i * header.num_channels + j) * 3;
                int32_t sample = static_cast<int8_t>(raw_data[idx + 2]);
                sample = (sample << 8) | raw_data[idx + 1];
                sample = (sample << 8) | raw_data[idx];
                sum += sample / 8388608.0f; // 2^23
            }
            audio_data[i] = sum / header.num_channels;
        }
    } else if (header.bits_per_sample == 32) {
        // 32位浮点音频
        std::vector<float> raw_data(num_samples * header.num_channels);
        file.read(reinterpret_cast<char*>(raw_data.data()), chunk_size);

        // 处理多声道（取平均）
        for (int i = 0; i < num_samples; ++i) {
            float sum = 0.0f;
            for (int j = 0; j < header.num_channels; ++j) {
                sum += raw_data[i * header.num_channels + j];
            }
            audio_data[i] = sum / header.num_channels;
        }
    } else {
        throw std::runtime_error("不支持的位深度: " + std::to_string(header.bits_per_sample));
    }

    // 如果采样率不匹配，执行重采样
    if (header.sample_rate != target_sample_rate) {
        AudioPreprocessor preprocessor(header.sample_rate, target_sample_rate);
        audio_data = preprocessor.resample(audio_data, header.sample_rate);
    }

    return std::make_pair(audio_data, header.sample_rate);
}

} // namespace edgevoice 
} // namespace edgevoice 