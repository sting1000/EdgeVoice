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

// 定义M_PI（如果不存在）
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

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
    int min_speech_frames = min_speech_frames_; // 最小语音段长度（帧数）
    
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
    context_frames_(context_frames),
    n_filters_(40),  // 默认Mel滤波器数量
    n_dim_(n_fft / 2 + 1)  // FFT输出维度
{
}

FeatureExtractor::~FeatureExtractor() {
    // 析构函数，无需特殊操作
}

std::vector<float> FeatureExtractor::preEmphasis(const std::vector<float>& signal, float coef) {
    if (signal.empty()) {
        return {};
    }
    
    std::vector<float> emphasized(signal.size());
    emphasized[0] = signal[0];
    
    for (size_t i = 1; i < signal.size(); ++i) {
        emphasized[i] = signal[i] - coef * signal[i - 1];
    }
    
    return emphasized;
}

std::vector<float> FeatureExtractor::getHammingWindow(bool periodic) {
    int window_length = n_fft_;
    int normSize = periodic ? window_length - 1 : window_length;
    
    std::vector<float> window(window_length);
    for (int i = 0; i < window_length; ++i) {
        float pi = 3.14159f;
        window[i] = 0.54f - (0.46f * std::cos((2.0f * pi * i) / normSize));
    }
    
    return window;
}

void FeatureExtractor::applyHammingWindow(std::vector<float>& frame) {
    std::vector<float> window = getHammingWindow(true);
    
    // 应用窗函数
    for (size_t i = 0; i < frame.size(); ++i) {
        frame[i] *= window[i % window.size()];
    }
}

void FeatureExtractor::fft(std::vector<complex_d>& a, bool invert) {
    int n = a.size();
    if (n == 1) return;

    std::vector<complex_d> a0(n / 2), a1(n / 2);
    for (int i = 0; 2 * i < n; i++) {
        a0[i] = a[2 * i];
        a1[i] = a[2 * i + 1];
    }
    fft(a0, invert);
    fft(a1, invert);

    double ang = 2 * M_PI / n * (invert ? -1 : 1);
    complex_d w(1), wn(std::cos(ang), std::sin(ang));
    for (int i = 0; 2 * i < n; i++) {
        a[i] = a0[i] + w * a1[i];
        a[i + n / 2] = a0[i] - w * a1[i];
        if (invert) {
            a[i] /= 2;
            a[i + n / 2] /= 2;
        }
        w *= wn;
    }
}

std::vector<float> FeatureExtractor::computePowerSpec(const std::vector<complex_d>& fft_result) {
    std::vector<float> powerSpec(n_dim_);
    for (int i = 0; i < n_dim_; ++i) {
        float real = static_cast<float>(fft_result[i].real());
        float imag = static_cast<float>(fft_result[i].imag());
        powerSpec[i] = (real * real + imag * imag) / n_fft_;
    }
    return powerSpec;
}

double FeatureExtractor::hzToMel(double hz) {
    // 将频率从Hz转换为Mel刻度
    // 使用标准公式: m = 1127.01048 * ln(1 + f/700)
    return 1127.01048 * std::log(1.0 + hz / 700.0);
}

double FeatureExtractor::melToHz(double mel) {
    // 将Mel刻度转换回Hz
    // 使用标准公式: f = 700 * (exp(m/1127.01048) - 1)
    return 700.0 * (std::exp(mel / 1127.01048) - 1.0);
}

std::vector<double> FeatureExtractor::linspace(double start, double end, int num) {
    std::vector<double> result(num);
    double step = (end - start) / (num - 1);
    
    for (int i = 0; i < num; ++i) {
        result[i] = start + i * step;
    }
    
    return result;
}

std::vector<std::vector<float>> FeatureExtractor::getMelFilterbank() {
    // 创建Mel滤波器组
    double nyquist_freq = sample_rate_ / 2.0;
    
    // 创建线性频率刻度
    std::vector<double> fft_freqs = linspace(0.0, nyquist_freq, n_dim_);
    
    // 创建Mel刻度边界
    double low_freq_mel = hzToMel(0.0);
    double high_freq_mel = hzToMel(nyquist_freq);
    std::vector<double> mel_points = linspace(low_freq_mel, high_freq_mel, n_filters_ + 2);
    
    // 将Mel刻度边界转换回Hz
    std::vector<double> hz_points(mel_points.size());
    for (size_t i = 0; i < mel_points.size(); ++i) {
        hz_points[i] = melToHz(mel_points[i]);
    }
    
    // 计算滤波器组系数
    std::vector<std::vector<float>> filterbank(n_filters_, std::vector<float>(n_dim_, 0.0f));
    
    for (int i = 0; i < n_filters_; ++i) {
        // 计算三角形滤波器
        double f_m_minus = hz_points[i];     // 左边界
        double f_m = hz_points[i + 1];       // 中心
        double f_m_plus = hz_points[i + 2];  // 右边界
        
        for (int j = 0; j < n_dim_; ++j) {
            double freq = fft_freqs[j];
            
            // 左半三角
            if (freq >= f_m_minus && freq <= f_m) {
                filterbank[i][j] = static_cast<float>((freq - f_m_minus) / (f_m - f_m_minus));
            }
            // 右半三角
            else if (freq >= f_m && freq <= f_m_plus) {
                filterbank[i][j] = static_cast<float>((f_m_plus - freq) / (f_m_plus - f_m));
            }
        }
    }
    
    return filterbank;
}

std::vector<std::vector<float>> FeatureExtractor::extractMFCC(const std::vector<float>& audio) {
    if (audio.empty()) {
        return {};
    }
    
    // 预加重
    std::vector<float> emphasized = preEmphasis(audio);
    
    // 计算帧数
    int num_frames = static_cast<int>((audio.size() - n_fft_) / hop_length_) + 1;
    if (num_frames <= 0) {
        return {};
    }
    
    // 获取Mel滤波器组
    std::vector<std::vector<float>> mel_filterbank = getMelFilterbank();
    
    // 创建MFCC特征矩阵
    std::vector<std::vector<float>> mfcc_features(num_frames, std::vector<float>(n_mfcc_));
    
    // 每帧处理
    for (int i = 0; i < num_frames; ++i) {
        // 帧起始位置
        int start = i * hop_length_;
        int end = std::min(start + n_fft_, static_cast<int>(emphasized.size()));
        
        // 提取当前帧
        std::vector<float> frame(n_fft_, 0.0f);
        std::copy(emphasized.begin() + start, emphasized.begin() + end, frame.begin());
        
        // 应用汉明窗
        applyHammingWindow(frame);
        
        // 执行FFT
        std::vector<complex_d> fft_input(n_fft_);
        for (int j = 0; j < n_fft_; ++j) {
            fft_input[j] = complex_d(frame[j], 0.0);
        }
        fft(fft_input, false);
        
        // 计算功率谱
        std::vector<float> power_spec = computePowerSpec(fft_input);
        
        // 应用Mel滤波器
        std::vector<float> mel_energy(n_filters_, 0.0f);
        for (int j = 0; j < n_filters_; ++j) {
            for (int k = 0; k < n_dim_; ++k) {
                mel_energy[j] += power_spec[k] * mel_filterbank[j][k];
            }
            // 防止对0或负数取对数
            mel_energy[j] = std::max(mel_energy[j], 1e-10f);
            // 取对数
            mel_energy[j] = std::log(mel_energy[j]);
        }
        
        // 应用离散余弦变换(DCT)提取MFCC
        for (int j = 0; j < n_mfcc_; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < n_filters_; ++k) {
                sum += mel_energy[k] * std::cos(M_PI * j * (k + 0.5) / n_filters_);
            }
            mfcc_features[i][j] = sum;
        }
    }
    
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
    
    // 添加上下文特征
    std::vector<std::vector<float>> context_features = addContext(mfcc_features, context_frames_);
    
    return context_features;
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

WavData loadWavFile(const std::string& filename) {
    // 使用新版本的loadWavFile函数，保持默认目标采样率
    auto [samples, sample_rate] = loadWavFile(filename, 16000);
    
    // 创建并返回WavData结构
    WavData wav_data;
    wav_data.samples = samples;
    wav_data.sample_rate = sample_rate;
    wav_data.bit_depth = 16; // 假设16位深度（这是最常见的）
    wav_data.channels = 1;   // 假设单声道（因为我们已经将多声道音频合并为单声道）
    
    return wav_data;
}

} // namespace edgevoice 