/**
 * @file main.cpp
 * @brief 使用HiAI推理框架的EdgeVoice部署示例
 */

#include <iostream>
#include <string>
#include <vector>
#include <chrono>
#include <iomanip>

#include "../include/inference_engine.h"

// 打印识别结果
void printResult(const edgevoice::IntentResult& result, const std::string& wav_file, const std::string& model_path) {
    std::cout << "-------------------------------------------" << std::endl;
    std::cout << "识别结果:" << std::endl;
    std::cout << "文件: " << wav_file << std::endl;
    std::cout << "意图: " << result.intent_class << std::endl;
    std::cout << "置信度: " << result.confidence * 100.0f << "%" << std::endl;
    std::cout << "预处理时间: " << result.preprocessing_time << " ms" << std::endl;
    std::cout << "推理时间: " << result.inference_time << " ms" << std::endl;
    std::cout << "总时间: " << result.total_time << " ms" << std::endl;
    std::cout << "模型: " << model_path << std::endl;
    std::cout << "-------------------------------------------" << std::endl;
}

// 打印使用方法
void printUsage(const char* program_name) {
    std::cout << "使用方法:" << std::endl;
    std::cout << "  " << program_name << " <模型路径> <音频文件路径> [音频文件路径2 ...]" << std::endl;
    std::cout << std::endl;
    std::cout << "参数:" << std::endl;
    std::cout << "  <模型路径>       - OMC模型文件的路径" << std::endl;
    std::cout << "  <音频文件路径>   - 要识别的WAV音频文件路径" << std::endl;
    std::cout << "  [音频文件路径2]  - 可选的其他WAV文件路径，用于批处理" << std::endl;
    std::cout << std::endl;
    std::cout << "示例:" << std::endl;
    std::cout << "  " << program_name << " model.omc test.wav" << std::endl;
    std::cout << "  " << program_name << " model.omc test1.wav test2.wav test3.wav" << std::endl;
}

int main(int argc, char* argv[]) {
    // 检查参数
    if (argc < 3) {
        std::cerr << "错误: 参数不足" << std::endl;
        printUsage(argv[0]);
        return 1;
    }
    
    // 解析参数
    std::string model_path = argv[1];
    std::string audio_file = argv[2];
    
    try {
        // 创建并初始化推理引擎
        edgevoice::InferenceEngine engine(model_path, 0.7f);
        
        if (!engine.init()) {
            std::cerr << "错误: 无法初始化推理引擎" << std::endl;
            return 1;
        }
        
        std::cout << "已加载模型: " << model_path << std::endl;
        
        // 处理第一个音频文件
        std::cout << "正在处理: " << audio_file << std::endl;
        auto result = engine.predictFromWavFile(audio_file);
        printResult(result, audio_file, model_path);
        
        // 如果有更多音频文件，展示批处理能力
        if (argc > 3) {
            std::cout << "\n演示批处理能力...\n" << std::endl;
            
            for (int i = 3; i < argc; ++i) {
                std::string next_audio_file = argv[i];
                std::cout << "正在处理: " << next_audio_file << std::endl;
                
                auto batch_result = engine.predictFromWavFile(next_audio_file);
                printResult(batch_result, next_audio_file, model_path);
            }
        }
        
        std::cout << "处理完成!" << std::endl;
        return 0;
    }
    catch (const std::exception& e) {
        std::cerr << "错误: " << e.what() << std::endl;
        return 1;
    }
} 