#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
性能测试工具 - 测试EdgeVoice模型在设备上的性能
"""

import os
import sys
import time
import argparse
import logging
import subprocess
import numpy as np
from pathlib import Path
from datetime import datetime

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('performance_tester')

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='测试EdgeVoice模型在设备上的性能')
    parser.add_argument('--model', '-m', required=True, help='OMC模型路径')
    parser.add_argument('--test_dir', '-d', required=True, help='测试音频文件目录')
    parser.add_argument('--output', '-o', default='performance_results.csv', 
                        help='输出结果文件路径')
    parser.add_argument('--iterations', '-i', type=int, default=1, 
                        help='每个音频文件测试的迭代次数')
    parser.add_argument('--binary', '-b', default='edgevoice', 
                        help='EdgeVoice可执行文件路径')
    parser.add_argument('--verbose', '-v', action='store_true', help='显示详细日志')
    
    return parser.parse_args()

def find_wav_files(directory):
    """查找目录中的所有WAV文件"""
    wav_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.lower().endswith('.wav'):
                wav_files.append(os.path.join(root, file))
    return wav_files

def run_inference(binary_path, model_path, audio_file):
    """
    运行推理并解析结果
    
    参数:
        binary_path (str): EdgeVoice可执行文件路径
        model_path (str): 模型文件路径
        audio_file (str): 音频文件路径
    
    返回:
        dict: 包含推理结果的字典
    """
    try:
        # 构建命令
        cmd = [binary_path, model_path, audio_file]
        logger.debug(f"执行命令: {' '.join(cmd)}")
        
        # 运行命令并捕获输出
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        output = result.stdout
        
        # 解析输出
        result_dict = {
            'file': audio_file,
            'intent': 'UNKNOWN',
            'confidence': 0.0,
            'preprocessing_time': 0.0,
            'inference_time': 0.0,
            'total_time': 0.0
        }
        
        for line in output.splitlines():
            line = line.strip()
            if line.startswith('意图:'):
                result_dict['intent'] = line.split(':', 1)[1].strip()
            elif line.startswith('置信度:'):
                confidence_str = line.split(':', 1)[1].strip().rstrip('%')
                result_dict['confidence'] = float(confidence_str) / 100.0
            elif line.startswith('预处理时间:'):
                time_str = line.split(':', 1)[1].strip().rstrip(' ms')
                result_dict['preprocessing_time'] = float(time_str)
            elif line.startswith('推理时间:'):
                time_str = line.split(':', 1)[1].strip().rstrip(' ms')
                result_dict['inference_time'] = float(time_str)
            elif line.startswith('总时间:'):
                time_str = line.split(':', 1)[1].strip().rstrip(' ms')
                result_dict['total_time'] = float(time_str)
        
        return result_dict
    
    except subprocess.CalledProcessError as e:
        logger.error(f"推理执行失败: {e}")
        logger.error(f"错误输出: {e.stderr}")
        return None
    except Exception as e:
        logger.error(f"推理过程中发生错误: {str(e)}")
        return None

def run_performance_test(binary_path, model_path, wav_files, iterations=1):
    """
    对多个WAV文件运行性能测试
    
    参数:
        binary_path (str): EdgeVoice可执行文件路径
        model_path (str): 模型文件路径
        wav_files (list): WAV文件路径列表
        iterations (int): 每个文件测试的迭代次数
    
    返回:
        list: 包含所有测试结果的列表
    """
    results = []
    total_files = len(wav_files)
    
    logger.info(f"开始性能测试，共 {total_files} 个文件，每个文件 {iterations} 次迭代")
    
    for i, wav_file in enumerate(wav_files):
        logger.info(f"处理文件 {i+1}/{total_files}: {wav_file}")
        
        file_results = []
        for j in range(iterations):
            if iterations > 1:
                logger.debug(f"  迭代 {j+1}/{iterations}")
            
            result = run_inference(binary_path, model_path, wav_file)
            if result:
                file_results.append(result)
        
        # 如果有多次迭代，计算平均值
        if file_results and iterations > 1:
            avg_result = {
                'file': wav_file,
                'intent': max(set(r['intent'] for r in file_results), key=[r['intent'] for r in file_results].count),
                'confidence': np.mean([r['confidence'] for r in file_results]),
                'preprocessing_time': np.mean([r['preprocessing_time'] for r in file_results]),
                'inference_time': np.mean([r['inference_time'] for r in file_results]),
                'total_time': np.mean([r['total_time'] for r in file_results])
            }
            results.append(avg_result)
        else:
            results.extend(file_results)
    
    return results

def save_results_to_csv(results, output_file):
    """将结果保存到CSV文件"""
    if not results:
        logger.warning("没有结果可保存")
        return False
    
    try:
        # 确保输出目录存在
        output_dir = os.path.dirname(output_file)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # 写入CSV文件
        with open(output_file, 'w') as f:
            # 写入标题行
            f.write("文件,意图,置信度,预处理时间(ms),推理时间(ms),总时间(ms)\n")
            
            # 写入数据行
            for result in results:
                f.write(f"{result['file']},{result['intent']},{result['confidence']:.4f},"
                        f"{result['preprocessing_time']:.2f},{result['inference_time']:.2f},"
                        f"{result['total_time']:.2f}\n")
        
        logger.info(f"结果已保存到: {output_file}")
        return True
    
    except Exception as e:
        logger.error(f"保存结果失败: {str(e)}")
        return False

def print_summary(results):
    """打印性能测试摘要"""
    if not results:
        logger.warning("没有结果可显示")
        return
    
    # 计算统计数据
    preprocessing_times = [r['preprocessing_time'] for r in results]
    inference_times = [r['inference_time'] for r in results]
    total_times = [r['total_time'] for r in results]
    
    # 打印摘要
    logger.info("\n性能测试摘要:")
    logger.info(f"测试文件数: {len(results)}")
    logger.info(f"预处理时间: 平均 {np.mean(preprocessing_times):.2f} ms, "
                f"最小 {np.min(preprocessing_times):.2f} ms, "
                f"最大 {np.max(preprocessing_times):.2f} ms")
    logger.info(f"推理时间: 平均 {np.mean(inference_times):.2f} ms, "
                f"最小 {np.min(inference_times):.2f} ms, "
                f"最大 {np.max(inference_times):.2f} ms")
    logger.info(f"总时间: 平均 {np.mean(total_times):.2f} ms, "
                f"最小 {np.min(total_times):.2f} ms, "
                f"最大 {np.max(total_times):.2f} ms")

def main():
    """主函数"""
    args = parse_args()
    
    # 设置日志级别
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    
    # 检查模型文件是否存在
    if not os.path.exists(args.model):
        logger.error(f"模型文件不存在: {args.model}")
        return 1
    
    # 检查测试目录是否存在
    if not os.path.exists(args.test_dir):
        logger.error(f"测试目录不存在: {args.test_dir}")
        return 1
    
    # 查找WAV文件
    wav_files = find_wav_files(args.test_dir)
    if not wav_files:
        logger.error(f"在目录 {args.test_dir} 中未找到WAV文件")
        return 1
    
    logger.info(f"找到 {len(wav_files)} 个WAV文件")
    
    # 运行性能测试
    start_time = time.time()
    results = run_performance_test(args.binary, args.model, wav_files, args.iterations)
    end_time = time.time()
    
    if not results:
        logger.error("性能测试失败，未获得结果")
        return 1
    
    # 打印摘要
    print_summary(results)
    
    # 保存结果
    if save_results_to_csv(results, args.output):
        logger.info(f"总测试时间: {end_time - start_time:.2f} 秒")
        return 0
    else:
        return 1

if __name__ == "__main__":
    sys.exit(main()) 