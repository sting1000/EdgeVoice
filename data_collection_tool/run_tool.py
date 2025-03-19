#!/usr/bin/env python
"""
EdgeVoice 语音数据采集工具启动脚本
"""

import os
import sys
import argparse
import tkinter as tk
from pathlib import Path

def run_data_collection_tool():
    """启动数据采集工具"""
    from data_collection_tool import AudioRecorderApp
    
    root = tk.Tk()
    app = AudioRecorderApp(root)
    root.mainloop()

def run_data_stats():
    """运行数据统计工具"""
    from data_stats import generate_dataset_report
    
    data_dir = "../data"
    annotation_file = "../data/annotations.csv"
    output_dir = "../reports"
    
    generate_dataset_report(annotation_file, data_dir, output_dir)
    print("统计报告已生成")

def run_split_dataset():
    """运行数据集拆分工具"""
    from data_stats import split_dataset
    
    annotation_file = "../data/annotations.csv"
    output_dir = "../data/split"
    
    split_dataset(annotation_file, output_dir=output_dir)
    print("数据集拆分已完成")

def run_prompt_generator():
    """运行提示语生成器"""
    from prompt_generator import enrich_intent_prompts
    
    output_file = "expanded_prompts.json"
    enrich_intent_prompts(output_file)
    print("提示语已扩充")

def show_help():
    """显示帮助信息"""
    print("EdgeVoice 语音数据采集工具套件")
    print("\n可用命令:")
    print("  collect    - 启动语音数据采集工具")
    print("  stats      - 生成数据集统计报告")
    print("  split      - 拆分数据集为训练集、验证集和测试集")
    print("  prompts    - 扩充语音提示语")
    print("  help       - 显示此帮助信息")

def main():
    parser = argparse.ArgumentParser(description='EdgeVoice 语音数据工具套件')
    parser.add_argument('command', nargs='?', default='collect',
                      choices=['collect', 'stats', 'split', 'prompts', 'help'],
                      help='要运行的命令')
    
    args = parser.parse_args()
    
    # 运行指定命令
    if args.command == 'collect':
        run_data_collection_tool()
    elif args.command == 'stats':
        run_data_stats()
    elif args.command == 'split':
        run_split_dataset()
    elif args.command == 'prompts':
        run_prompt_generator()
    else:
        show_help()

if __name__ == "__main__":
    main() 