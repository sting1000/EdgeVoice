#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
评估ASR模型性能的工具，计算字错率(CER)和词错率(WER)
"""

import os
import sys
import argparse
import logging
import json
import time
import pandas as pd
import numpy as np
import jieba
from tqdm import tqdm
from jiwer import wer, cer
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.wenet_asr import WeNetASR
from inference import IntentInferenceEngine

# 设置日志
logging.basicConfig(
    format='%(asctime)s %(levelname)s: %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

def load_test_data(annotation_file):
    """
    加载测试数据
    
    参数:
        annotation_file: 标注文件路径，CSV格式，包含file_path和text列
        
    返回:
        包含音频路径和参考文本的DataFrame
    """
    if not os.path.exists(annotation_file):
        raise FileNotFoundError(f"标注文件不存在: {annotation_file}")
    
    df = pd.read_csv(annotation_file)
    required_columns = ['file_path', 'text']
    
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"标注文件缺少必要的列: {col}")
    
    logger.info(f"加载了 {len(df)} 条测试数据")
    return df

def evaluate_asr_model(model_path, dict_path, test_data, output_dir=None, 
                       batch_size=16, device='cpu', save_results=True):
    """
    评估ASR模型性能
    
    参数:
        model_path: ASR模型路径
        dict_path: 字典文件路径
        test_data: 测试数据DataFrame
        output_dir: 输出目录
        batch_size: 批处理大小
        device: 设备 ('cpu' 或 'cuda')
        save_results: 是否保存结果
        
    返回:
        包含评估结果的字典
    """
    # 初始化ASR模型
    logger.info(f"加载ASR模型: {model_path}")
    asr_model = WeNetASR(
        model_path=model_path,
        dict_path=dict_path,
        device=device,
        save_results=save_results,
        result_dir=output_dir if output_dir else "asr_results"
    )
    
    # 准备结果存储
    results = {
        "predictions": [],
        "references": [],
        "file_paths": [],
        "cer_scores": [],
        "wer_scores": [],
        "processing_times": []
    }
    
    # 处理测试数据
    logger.info("开始评估ASR模型...")
    for i in tqdm(range(0, len(test_data), batch_size), desc="处理测试数据"):
        batch = test_data.iloc[i:i+batch_size]
        
        for _, row in batch.iterrows():
            file_path = row['file_path']
            reference_text = row['text']
            
            # 确保文件存在
            if not os.path.exists(file_path):
                logger.warning(f"文件不存在，跳过: {file_path}")
                continue
            
            # 转写音频
            start_time = time.time()
            try:
                prediction, confidence, _ = asr_model.transcribe(file_path)
                processing_time = time.time() - start_time
                
                # 计算错误率
                cer_score = cer(reference_text, prediction)
                
                # 分词后计算WER
                ref_words = ' '.join(jieba.cut(reference_text))
                pred_words = ' '.join(jieba.cut(prediction))
                wer_score = wer(ref_words, pred_words)
                
                # 存储结果
                results["predictions"].append(prediction)
                results["references"].append(reference_text)
                results["file_paths"].append(file_path)
                results["cer_scores"].append(cer_score)
                results["wer_scores"].append(wer_score)
                results["processing_times"].append(processing_time)
                
            except Exception as e:
                logger.error(f"处理文件时出错 {file_path}: {str(e)}")
    
    # 计算总体指标
    if results["cer_scores"]:
        avg_cer = np.mean(results["cer_scores"])
        avg_wer = np.mean(results["wer_scores"])
        avg_time = np.mean(results["processing_times"])
        
        logger.info(f"评估完成! 平均字错率(CER): {avg_cer:.4f}, 平均词错率(WER): {avg_wer:.4f}")
        logger.info(f"平均处理时间: {avg_time:.4f}秒/样本")
        
        # 保存详细结果
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            
            # 保存详细结果到CSV
            results_df = pd.DataFrame({
                "file_path": results["file_paths"],
                "reference": results["references"],
                "prediction": results["predictions"],
                "cer": results["cer_scores"],
                "wer": results["wer_scores"],
                "processing_time": results["processing_times"]
            })
            
            csv_path = os.path.join(output_dir, "asr_evaluation_results.csv")
            results_df.to_csv(csv_path, index=False, encoding="utf-8")
            logger.info(f"详细结果已保存到: {csv_path}")
            
            # 保存摘要结果
            summary = {
                "model_path": model_path,
                "dict_path": dict_path,
                "num_samples": len(results["cer_scores"]),
                "avg_cer": float(avg_cer),
                "avg_wer": float(avg_wer),
                "avg_processing_time": float(avg_time),
                "device": device,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            }
            
            json_path = os.path.join(output_dir, "asr_evaluation_summary.json")
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(summary, f, ensure_ascii=False, indent=2)
            logger.info(f"摘要结果已保存到: {json_path}")
            
            # 生成错误率分布图
            plt.figure(figsize=(12, 6))
            
            plt.subplot(1, 2, 1)
            sns.histplot(results["cer_scores"], kde=True)
            plt.title("字错率(CER)分布")
            plt.xlabel("CER")
            plt.ylabel("样本数")
            
            plt.subplot(1, 2, 2)
            sns.histplot(results["wer_scores"], kde=True)
            plt.title("词错率(WER)分布")
            plt.xlabel("WER")
            plt.ylabel("样本数")
            
            plt.tight_layout()
            plot_path = os.path.join(output_dir, "error_rate_distribution.png")
            plt.savefig(plot_path)
            logger.info(f"错误率分布图已保存到: {plot_path}")
            
            # 生成错误分析报告
            generate_error_analysis(results, output_dir)
        
        return {
            "avg_cer": avg_cer,
            "avg_wer": avg_wer,
            "avg_processing_time": avg_time,
            "num_samples": len(results["cer_scores"])
        }
    else:
        logger.warning("没有成功处理任何样本!")
        return {
            "avg_cer": None,
            "avg_wer": None,
            "avg_processing_time": None,
            "num_samples": 0
        }

def generate_error_analysis(results, output_dir):
    """
    生成错误分析报告
    
    参数:
        results: 评估结果
        output_dir: 输出目录
    """
    # 找出错误率最高的样本
    results_df = pd.DataFrame({
        "file_path": results["file_paths"],
        "reference": results["references"],
        "prediction": results["predictions"],
        "cer": results["cer_scores"],
        "wer": results["wer_scores"]
    })
    
    # 按错误率排序
    worst_cer = results_df.sort_values("cer", ascending=False).head(10)
    worst_wer = results_df.sort_values("wer", ascending=False).head(10)
    
    # 生成错误分析报告
    report = "# ASR错误分析报告\n\n"
    
    report += "## 字错率(CER)最高的10个样本\n\n"
    report += "| 文件 | 参考文本 | 预测文本 | CER |\n"
    report += "| --- | --- | --- | --- |\n"
    
    for _, row in worst_cer.iterrows():
        file_name = os.path.basename(row["file_path"])
        report += f"| {file_name} | {row['reference']} | {row['prediction']} | {row['cer']:.4f} |\n"
    
    report += "\n## 词错率(WER)最高的10个样本\n\n"
    report += "| 文件 | 参考文本 | 预测文本 | WER |\n"
    report += "| --- | --- | --- | --- |\n"
    
    for _, row in worst_wer.iterrows():
        file_name = os.path.basename(row["file_path"])
        report += f"| {file_name} | {row['reference']} | {row['prediction']} | {row['wer']:.4f} |\n"
    
    # 常见错误分析
    report += "\n## 常见错误模式分析\n\n"
    
    # 分析字符级别错误
    char_errors = analyze_character_errors(results["references"], results["predictions"])
    
    report += "### 最常见的字符混淆\n\n"
    report += "| 参考字符 | 预测字符 | 出现次数 |\n"
    report += "| --- | --- | --- |\n"
    
    for (ref_char, pred_char), count in char_errors[:10]:
        report += f"| {ref_char} | {pred_char} | {count} |\n"
    
    # 保存报告
    report_path = os.path.join(output_dir, "error_analysis.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)
    
    logger.info(f"错误分析报告已保存到: {report_path}")

def analyze_character_errors(references, predictions):
    """
    分析字符级别的错误
    
    参数:
        references: 参考文本列表
        predictions: 预测文本列表
        
    返回:
        按频率排序的字符错误列表
    """
    from difflib import SequenceMatcher
    
    char_errors = {}
    
    for ref, pred in zip(references, predictions):
        # 使用SequenceMatcher找出差异
        matcher = SequenceMatcher(None, ref, pred)
        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag == 'replace':
                # 字符替换错误
                ref_segment = ref[i1:i2]
                pred_segment = pred[j1:j2]
                
                # 如果长度相同，逐字符比较
                if len(ref_segment) == len(pred_segment):
                    for r_char, p_char in zip(ref_segment, pred_segment):
                        if r_char != p_char:
                            error_key = (r_char, p_char)
                            char_errors[error_key] = char_errors.get(error_key, 0) + 1
    
    # 按频率排序
    sorted_errors = sorted(char_errors.items(), key=lambda x: x[1], reverse=True)
    return sorted_errors

def evaluate_inference_engine(engine_config, test_data, output_dir=None):
    """
    评估完整推理引擎的ASR性能
    
    参数:
        engine_config: 推理引擎配置字典
        test_data: 测试数据DataFrame
        output_dir: 输出目录
        
    返回:
        包含评估结果的字典
    """
    # 初始化推理引擎
    logger.info("初始化推理引擎...")
    engine = IntentInferenceEngine(**engine_config)
    
    # 准备结果存储
    results = {
        "predictions": [],
        "references": [],
        "file_paths": [],
        "cer_scores": [],
        "wer_scores": [],
        "processing_times": [],
        "paths": []
    }
    
    # 处理测试数据
    logger.info("开始评估推理引擎的ASR性能...")
    for _, row in tqdm(test_data.iterrows(), total=len(test_data), desc="处理测试数据"):
        file_path = row['file_path']
        reference_text = row['text']
        
        # 确保文件存在
        if not os.path.exists(file_path):
            logger.warning(f"文件不存在，跳过: {file_path}")
            continue
        
        # 处理音频
        start_time = time.time()
        try:
            result = engine.process_audio_file(file_path)
            processing_time = time.time() - start_time
            
            # 检查是否包含转写结果
            if "transcription" in result:
                prediction = result["transcription"]
                path = result["path"]
                
                # 计算错误率
                cer_score = cer(reference_text, prediction)
                
                # 分词后计算WER
                ref_words = ' '.join(jieba.cut(reference_text))
                pred_words = ' '.join(jieba.cut(prediction))
                wer_score = wer(ref_words, pred_words)
                
                # 存储结果
                results["predictions"].append(prediction)
                results["references"].append(reference_text)
                results["file_paths"].append(file_path)
                results["cer_scores"].append(cer_score)
                results["wer_scores"].append(wer_score)
                results["processing_times"].append(processing_time)
                results["paths"].append(path)
            else:
                logger.warning(f"处理 {file_path} 时未生成转写结果")
                
        except Exception as e:
            logger.error(f"处理文件时出错 {file_path}: {str(e)}")
    
    # 计算总体指标
    if results["cer_scores"]:
        avg_cer = np.mean(results["cer_scores"])
        avg_wer = np.mean(results["wer_scores"])
        avg_time = np.mean(results["processing_times"])
        
        # 计算不同路径的性能
        path_results = {}
        for path, cer_score, wer_score in zip(results["paths"], results["cer_scores"], results["wer_scores"]):
            if path not in path_results:
                path_results[path] = {"cer": [], "wer": []}
            path_results[path]["cer"].append(cer_score)
            path_results[path]["wer"].append(wer_score)
        
        for path, scores in path_results.items():
            path_avg_cer = np.mean(scores["cer"])
            path_avg_wer = np.mean(scores["wer"])
            logger.info(f"路径 {path} 的平均字错率(CER): {path_avg_cer:.4f}, 平均词错率(WER): {path_avg_wer:.4f}")
        
        logger.info(f"评估完成! 总体平均字错率(CER): {avg_cer:.4f}, 平均词错率(WER): {avg_wer:.4f}")
        logger.info(f"平均处理时间: {avg_time:.4f}秒/样本")
        
        # 保存详细结果
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            
            # 保存详细结果到CSV
            results_df = pd.DataFrame({
                "file_path": results["file_paths"],
                "reference": results["references"],
                "prediction": results["predictions"],
                "cer": results["cer_scores"],
                "wer": results["wer_scores"],
                "processing_time": results["processing_times"],
                "path": results["paths"]
            })
            
            csv_path = os.path.join(output_dir, "inference_engine_evaluation.csv")
            results_df.to_csv(csv_path, index=False, encoding="utf-8")
            logger.info(f"详细结果已保存到: {csv_path}")
            
            # 保存摘要结果
            summary = {
                "engine_config": engine_config,
                "num_samples": len(results["cer_scores"]),
                "avg_cer": float(avg_cer),
                "avg_wer": float(avg_wer),
                "avg_processing_time": float(avg_time),
                "path_results": {
                    path: {
                        "avg_cer": float(np.mean(scores["cer"])),
                        "avg_wer": float(np.mean(scores["wer"])),
                        "num_samples": len(scores["cer"])
                    }
                    for path, scores in path_results.items()
                },
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            }
            
            json_path = os.path.join(output_dir, "inference_engine_summary.json")
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(summary, f, ensure_ascii=False, indent=2)
            logger.info(f"摘要结果已保存到: {json_path}")
            
            # 生成路径比较图
            plt.figure(figsize=(10, 6))
            
            paths = list(path_results.keys())
            cer_means = [np.mean(path_results[p]["cer"]) for p in paths]
            wer_means = [np.mean(path_results[p]["wer"]) for p in paths]
            
            x = np.arange(len(paths))
            width = 0.35
            
            plt.bar(x - width/2, cer_means, width, label='CER')
            plt.bar(x + width/2, wer_means, width, label='WER')
            
            plt.xlabel('处理路径')
            plt.ylabel('错误率')
            plt.title('不同处理路径的错误率比较')
            plt.xticks(x, paths)
            plt.legend()
            
            plt.tight_layout()
            plot_path = os.path.join(output_dir, "path_comparison.png")
            plt.savefig(plot_path)
            logger.info(f"路径比较图已保存到: {plot_path}")
        
        return {
            "avg_cer": avg_cer,
            "avg_wer": avg_wer,
            "avg_processing_time": avg_time,
            "num_samples": len(results["cer_scores"]),
            "path_results": path_results
        }
    else:
        logger.warning("没有成功处理任何样本!")
        return {
            "avg_cer": None,
            "avg_wer": None,
            "avg_processing_time": None,
            "num_samples": 0,
            "path_results": {}
        }

def main():
    parser = argparse.ArgumentParser(description="评估ASR模型性能")
    parser.add_argument("--model", type=str, required=True, help="ASR模型路径")
    parser.add_argument("--dict", type=str, required=True, help="字典文件路径")
    parser.add_argument("--test_data", type=str, required=True, help="测试数据标注文件路径")
    parser.add_argument("--output", type=str, default="asr_evaluation", help="输出目录")
    parser.add_argument("--batch_size", type=int, default=16, help="批处理大小")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"], help="设备")
    parser.add_argument("--engine_mode", action="store_true", help="使用完整推理引擎进行评估")
    parser.add_argument("--fast_model", type=str, help="快速模型路径 (仅在engine_mode=True时使用)")
    parser.add_argument("--precise_model", type=str, help="精确模型路径 (仅在engine_mode=True时使用)")
    
    args = parser.parse_args()
    
    # 加载测试数据
    test_data = load_test_data(args.test_data)
    
    # 创建输出目录
    os.makedirs(args.output, exist_ok=True)
    
    if args.engine_mode:
        # 检查必要的参数
        if not args.fast_model or not args.precise_model:
            parser.error("使用推理引擎模式时，必须提供fast_model和precise_model参数")
        
        # 配置推理引擎
        engine_config = {
            "fast_model_path": args.fast_model,
            "precise_model_path": args.precise_model,
            "asr_model_path": args.model,
            "asr_dict_path": args.dict,
            "save_asr_results": True
        }
        
        # 评估推理引擎
        results = evaluate_inference_engine(engine_config, test_data, args.output)
    else:
        # 直接评估ASR模型
        results = evaluate_asr_model(
            args.model, 
            args.dict, 
            test_data, 
            args.output,
            args.batch_size,
            args.device
        )
    
    # 打印最终结果
    if results["avg_cer"] is not None:
        print("\n最终评估结果:")
        print(f"平均字错率(CER): {results['avg_cer']:.4f}")
        print(f"平均词错率(WER): {results['avg_wer']:.4f}")
        print(f"平均处理时间: {results['avg_processing_time']:.4f}秒/样本")
        print(f"处理样本数: {results['num_samples']}")
        
        if args.engine_mode and "path_results" in results:
            print("\n不同处理路径的结果:")
            for path, path_results in results["path_results"].items():
                print(f"路径 {path}:")
                print(f"  平均字错率(CER): {np.mean(path_results['cer']):.4f}")
                print(f"  平均词错率(WER): {np.mean(path_results['wer']):.4f}")
                print(f"  样本数: {len(path_results['cer'])}")

if __name__ == "__main__":
    main() 