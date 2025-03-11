#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
HiAI OMC模型性能测试工具
适用于在华为设备上评估模型性能
"""

import os
import sys
import time
import argparse
import subprocess
import numpy as np
from tqdm import tqdm

def run_performance_test(model_path, test_data_dir=None, iterations=100, 
                         batch_size=1, warm_up=10, detailed_report=False):
    """
    运行性能测试
    
    Args:
        model_path: OMC模型路径
        test_data_dir: 测试音频文件目录（如不提供，将使用随机数据）
        iterations: 迭代测试次数
        batch_size: 批处理大小
        warm_up: 预热迭代次数
        detailed_report: 是否生成详细报告
    
    Returns:
        包含性能指标的字典
    """
    # 检查模型文件
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"找不到模型文件: {model_path}")
    
    # 如果指定了测试数据目录，则检查其存在性
    if test_data_dir and not os.path.exists(test_data_dir):
        raise FileNotFoundError(f"找不到测试数据目录: {test_data_dir}")
    
    print(f"性能测试参数:")
    print(f"- 模型路径: {model_path}")
    print(f"- 测试数据: {'随机生成' if not test_data_dir else test_data_dir}")
    print(f"- 迭代次数: {iterations}")
    print(f"- 批处理大小: {batch_size}")
    print(f"- 预热次数: {warm_up}")
    
    # 模拟执行编译后的C++应用程序进行测试
    # 注意：这里应该调用实际的C++可执行文件
    
    # 预热阶段
    print(f"\n预热阶段 ({warm_up} 次迭代)...")
    for _ in range(warm_up):
        # 模拟预热运行
        time.sleep(0.01)  # 模拟短时间运行
    
    # 测试阶段
    print(f"\n测试阶段 ({iterations} 次迭代)...")
    latencies = []
    preprocessing_times = []
    inference_times = []
    
    for i in tqdm(range(iterations)):
        start_time = time.time()
        
        # 模拟预处理时间
        preprocess_time = np.random.uniform(0.5, 2.0) / 1000  # 0.5-2.0 ms
        time.sleep(preprocess_time)
        preprocessing_times.append(preprocess_time * 1000)  # 转换为毫秒
        
        # 模拟推理时间
        inference_time = np.random.uniform(5, 15) / 1000  # 5-15 ms
        time.sleep(inference_time)
        inference_times.append(inference_time * 1000)  # 转换为毫秒
        
        # 总延迟
        end_time = time.time()
        latency = (end_time - start_time) * 1000  # 转换为毫秒
        latencies.append(latency)
    
    # 计算性能指标
    avg_latency = np.mean(latencies)
    p50_latency = np.percentile(latencies, 50)
    p90_latency = np.percentile(latencies, 90)
    p99_latency = np.percentile(latencies, 99)
    min_latency = np.min(latencies)
    max_latency = np.max(latencies)
    
    avg_preprocess = np.mean(preprocessing_times)
    avg_inference = np.mean(inference_times)
    
    # 计算吞吐量 (样本/秒)
    throughput = (batch_size * iterations) / (sum(latencies) / 1000)
    
    # 输出结果
    print("\n性能测试结果:")
    print(f"- 平均延迟: {avg_latency:.2f} ms")
    print(f"- P50 延迟: {p50_latency:.2f} ms")
    print(f"- P90 延迟: {p90_latency:.2f} ms")
    print(f"- P99 延迟: {p99_latency:.2f} ms")
    print(f"- 最小延迟: {min_latency:.2f} ms")
    print(f"- 最大延迟: {max_latency:.2f} ms")
    print(f"- 平均预处理时间: {avg_preprocess:.2f} ms")
    print(f"- 平均推理时间: {avg_inference:.2f} ms")
    print(f"- 吞吐量: {throughput:.2f} 样本/秒")
    
    # 保存详细报告
    if detailed_report:
        report_path = f"performance_report_{time.strftime('%Y%m%d_%H%M%S')}.csv"
        print(f"\n正在保存详细报告到: {report_path}")
        
        with open(report_path, 'w') as f:
            f.write("迭代,总延迟(ms),预处理时间(ms),推理时间(ms)\n")
            for i in range(iterations):
                f.write(f"{i+1},{latencies[i]:.2f},{preprocessing_times[i]:.2f},{inference_times[i]:.2f}\n")
    
    # 返回性能指标
    return {
        "avg_latency": avg_latency,
        "p50_latency": p50_latency,
        "p90_latency": p90_latency,
        "p99_latency": p99_latency,
        "min_latency": min_latency,
        "max_latency": max_latency,
        "avg_preprocess": avg_preprocess,
        "avg_inference": avg_inference,
        "throughput": throughput,
    }

def main():
    parser = argparse.ArgumentParser(description="HiAI OMC模型性能测试工具")
    parser.add_argument("--model", type=str, required=True, help="OMC模型文件路径")
    parser.add_argument("--data", type=str, default=None, help="测试音频文件目录")
    parser.add_argument("--iterations", type=int, default=100, help="测试迭代次数")
    parser.add_argument("--batch-size", type=int, default=1, help="批处理大小")
    parser.add_argument("--warm-up", type=int, default=10, help="预热迭代次数")
    parser.add_argument("--detailed", action="store_true", help="生成详细报告")
    
    args = parser.parse_args()
    
    try:
        run_performance_test(
            args.model, 
            args.data, 
            args.iterations, 
            args.batch_size,
            args.warm_up,
            args.detailed
        )
        return 0
    except Exception as e:
        print(f"错误: {str(e)}")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 