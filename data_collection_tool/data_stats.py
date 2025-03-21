"""
数据集统计与可视化
提供数据集分析和可视化功能
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.font_manager import FontProperties
import seaborn as sns
import librosa
import librosa.display
from pathlib import Path
from collections import Counter

# 检查系统中可用的中文字体
def get_chinese_font():
    # 尝试寻找可用的中文字体
    chinese_fonts = ['SimHei', 'Microsoft YaHei', 'SimSun', 'NSimSun', 'FangSong', 'KaiTi']
    for font in chinese_fonts:
        try:
            # 检查matplotlib是否找到该字体
            font_path = fm.findfont(FontProperties(family=font), fallback_to_default=True)
            if os.path.exists(font_path) and 'ttf' in font_path.lower():
                return font
        except Exception:
            continue
    # 如果找不到中文字体，返回None
    return None

# 获取可用的中文字体
try:
    chinese_font = get_chinese_font()
    if chinese_font:
        plt.rcParams['font.sans-serif'] = [chinese_font, 'DejaVu Sans', 'sans-serif']
        print(f"使用中文字体: {chinese_font}")
    else:
        # 如果没有中文字体，仅使用英文标签
        print("警告: 没有找到可用的中文字体，图表将使用英文标签")
except Exception as e:
    print(f"设置字体时出错: {e}")
    
plt.rcParams['axes.unicode_minus'] = False

def load_dataset_info(annotation_file, data_dir=None):
    """
    加载数据集信息
    
    参数:
        annotation_file: 注释文件路径
        data_dir: 数据目录路径
    
    返回:
        DataFrame包含数据集信息
    """
    if not os.path.exists(annotation_file):
        print(f"注释文件不存在: {annotation_file}")
        return None
    
    try:
        df = pd.read_csv(annotation_file)
        
        # 添加额外信息
        if data_dir:
            # 添加文件存在检查
            df['file_exists'] = df['file_path'].apply(
                lambda x: os.path.exists(os.path.join(data_dir, x)))
            
            # 添加音频长度信息
            def get_audio_length(file_path):
                try:
                    full_path = os.path.join(data_dir, file_path)
                    if os.path.exists(full_path):
                        audio, sr = librosa.load(full_path, sr=None)
                        return len(audio) / sr
                    return None
                except:
                    return None
            
            df['audio_length'] = df['file_path'].apply(get_audio_length)
        
        return df
    
    except Exception as e:
        print(f"加载数据集信息时出错: {e}")
        return None

def plot_intent_distribution(df, output_path=None):
    """
    绘制意图分布图
    
    参数:
        df: 注释数据DataFrame
        output_path: 输出文件路径
    """
    plt.figure(figsize=(12, 6))
    # 修复 countplot 的使用方式
    sns.countplot(x='intent', hue='intent', data=df, palette='viridis', legend=False)
    plt.title('Intent Distribution')
    plt.xlabel('Intent Class')
    plt.ylabel('Sample Count')
    plt.xticks(rotation=45)
    
    if output_path:
        plt.tight_layout()
        plt.savefig(output_path)
        print(f"图表已保存至: {output_path}")
    else:
        plt.show()
    
    plt.close()

def plot_audio_length_distribution(df, output_path=None):
    """
    绘制音频长度分布图
    
    参数:
        df: 注释数据DataFrame
        output_path: 输出文件路径
    """
    if 'audio_length' not in df.columns:
        print("DataFrame中没有音频长度信息")
        return
    
    plt.figure(figsize=(12, 6))
    sns.histplot(df['audio_length'].dropna(), bins=20, kde=True)
    plt.title('Audio Length Distribution')
    plt.xlabel('Length (seconds)')
    plt.ylabel('Sample Count')
    
    if output_path:
        plt.tight_layout()
        plt.savefig(output_path)
        print(f"图表已保存至: {output_path}")
    else:
        plt.show()
    
    plt.close()

def plot_speaker_demographics(df, output_path=None):
    """
    绘制说话者人口统计数据
    
    参数:
        df: 注释数据DataFrame
        output_path: 输出文件路径
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # 处理中文标签问题 - 将中文标签替换为英文
    # 创建数据副本避免修改原始数据
    plot_df = df.copy()
    
    # 检查是否有中文标签，如果有则替换为英文
    if 'gender' in plot_df.columns:
        # 可能的中文性别标签映射
        gender_mapping = {'男': 'Male', '女': 'Female', '未指定': 'Unspecified'}
        for cn, en in gender_mapping.items():
            if cn in plot_df['gender'].values:
                plot_df['gender'] = plot_df['gender'].replace(cn, en)
    
    # 检查年龄组是否有中文
    if 'age_group' in plot_df.columns and plot_df['age_group'].dtype == 'object':  # 只有字符串类型可能包含中文
        age_group_mapping = {
            '未指定': 'Unspecified',
            '少年': 'Youth',
            '青年': 'Young Adult',
            '中年': 'Middle-aged',
            '老年': 'Senior'
        }
        # 仅替换已知的中文标签
        for cn, en in age_group_mapping.items():
            if cn in plot_df['age_group'].values:
                plot_df['age_group'] = plot_df['age_group'].replace(cn, en)
    
    # 性别分布 - 修复 countplot 的使用方式
    if 'gender' in plot_df.columns:
        sns.countplot(x='gender', hue='gender', data=plot_df, palette='Set2', ax=ax1, legend=False)
        ax1.set_title('Gender Distribution')
        ax1.set_xlabel('Gender')
        ax1.set_ylabel('Sample Count')
    else:
        ax1.text(0.5, 0.5, 'No gender data available', ha='center', va='center')
    
    # 年龄组分布 - 修复 countplot 的使用方式
    if 'age_group' in plot_df.columns:
        sns.countplot(x='age_group', hue='age_group', data=plot_df, palette='Set2', ax=ax2, legend=False)
        ax2.set_title('Age Group Distribution')
        ax2.set_xlabel('Age Group')
        ax2.set_ylabel('Sample Count')
    else:
        ax2.text(0.5, 0.5, 'No age group data available', ha='center', va='center')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path)
        print(f"图表已保存至: {output_path}")
    else:
        plt.show()
    
    plt.close()

def generate_dataset_report(annotation_file, data_dir, output_dir='reports'):
    """
    生成数据集报告
    
    参数:
        annotation_file: 注释文件路径
        data_dir: 数据目录路径
        output_dir: 输出目录路径
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 加载数据集信息
    df = load_dataset_info(annotation_file, data_dir)
    if df is None:
        return
    
    # 生成图表
    plot_intent_distribution(df, os.path.join(output_dir, 'intent_distribution.png'))
    plot_audio_length_distribution(df, os.path.join(output_dir, 'audio_length_distribution.png'))
    plot_speaker_demographics(df, os.path.join(output_dir, 'speaker_demographics.png'))
    
    # 统计信息
    total_samples = len(df)
    intent_counts = df['intent'].value_counts().to_dict()
    gender_counts = df['gender'].value_counts().to_dict() if 'gender' in df.columns else {}
    env_counts = df['environment'].value_counts().to_dict() if 'environment' in df.columns else {}
    
    # 计算音频统计信息
    audio_stats = {}
    if 'audio_length' in df.columns:
        audio_length = df['audio_length'].dropna()
        if not audio_length.empty:
            audio_stats = {
                'mean': audio_length.mean(),
                'median': audio_length.median(),
                'min': audio_length.min(),
                'max': audio_length.max(),
                'std': audio_length.std()
            }
    
    # 保存统计报告
    with open(os.path.join(output_dir, 'dataset_report.txt'), 'w', encoding='utf-8') as f:
        f.write("=== EdgeVoice Speech Dataset Statistics Report ===\n\n")
        f.write(f"Total Samples: {total_samples}\n\n")
        
        f.write("Intent Distribution:\n")
        for intent, count in sorted(intent_counts.items(), key=lambda x: x[1], reverse=True):
            f.write(f"  {intent}: {count} ({count/total_samples*100:.1f}%)\n")
        
        f.write("\nGender Distribution:\n")
        for gender, count in gender_counts.items():
            f.write(f"  {gender}: {count} ({count/total_samples*100:.1f}%)\n")
        
        f.write("\nEnvironment Distribution:\n")
        for env, count in env_counts.items():
            f.write(f"  {env}: {count} ({count/total_samples*100:.1f}%)\n")
        
        f.write("\nAudio Length Statistics (seconds):\n")
        for stat, value in audio_stats.items():
            f.write(f"  {stat}: {value:.2f}\n")
        
        f.write("\nFile Integrity Check:\n")
        if 'file_exists' in df.columns:
            missing_files = df[~df['file_exists']].shape[0]
            f.write(f"  Missing Files: {missing_files}\n")
    
    print(f"Report generated to: {output_dir}")

def split_dataset(annotation_file, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, 
                 stratify=True, output_dir='split_data', random_seed=42):
    """
    拆分数据集为训练集、验证集和测试集
    
    参数:
        annotation_file: 注释文件路径
        train_ratio: 训练集比例
        val_ratio: 验证集比例
        test_ratio: 测试集比例
        stratify: 是否按意图类别分层
        output_dir: 输出目录
        random_seed: 随机种子
    """
    from sklearn.model_selection import train_test_split
    
    # 确保比例之和为1
    total_ratio = train_ratio + val_ratio + test_ratio
    if abs(total_ratio - 1.0) > 1e-10:
        print(f"警告: 比例之和 ({total_ratio}) 不等于1，已自动调整")
        train_ratio /= total_ratio
        val_ratio /= total_ratio
        test_ratio /= total_ratio
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 加载注释文件
    df = pd.read_csv(annotation_file)
    
    # 设置随机种子
    np.random.seed(random_seed)
    
    # 执行拆分
    if stratify:
        # 第一次拆分：训练集 vs (验证集+测试集)
        train_df, temp_df = train_test_split(
            df, 
            train_size=train_ratio, 
            stratify=df['intent'],
            random_state=random_seed
        )
        
        # 第二次拆分：验证集 vs 测试集
        val_ratio_adjusted = val_ratio / (val_ratio + test_ratio)
        val_df, test_df = train_test_split(
            temp_df, 
            train_size=val_ratio_adjusted,
            stratify=temp_df['intent'],
            random_state=random_seed
        )
    else:
        # 第一次拆分
        train_df, temp_df = train_test_split(
            df, 
            train_size=train_ratio,
            random_state=random_seed
        )
        
        # 第二次拆分
        val_ratio_adjusted = val_ratio / (val_ratio + test_ratio)
        val_df, test_df = train_test_split(
            temp_df, 
            train_size=val_ratio_adjusted,
            random_state=random_seed
        )
    
    # 保存拆分的数据集
    train_df.to_csv(os.path.join(output_dir, 'train_annotations.csv'), index=False)
    val_df.to_csv(os.path.join(output_dir, 'val_annotations.csv'), index=False)
    test_df.to_csv(os.path.join(output_dir, 'test_annotations.csv'), index=False)
    
    # 打印统计信息
    print(f"数据集拆分完成:")
    print(f"  训练集: {len(train_df)} 样本 ({len(train_df)/len(df)*100:.1f}%)")
    print(f"  验证集: {len(val_df)} 样本 ({len(val_df)/len(df)*100:.1f}%)")
    print(f"  测试集: {len(test_df)} 样本 ({len(test_df)/len(df)*100:.1f}%)")
    
    # 返回拆分的数据集
    return train_df, val_df, test_df

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='EdgeVoice语音数据集统计工具')
    parser.add_argument('--annotation_file', type=str, default='../data/annotations.csv',
                        help='注释文件路径')
    parser.add_argument('--data_dir', type=str, default='../data',
                        help='数据目录路径')
    parser.add_argument('--output_dir', type=str, default='../reports',
                        help='输出目录路径')
    parser.add_argument('--split', action='store_true',
                        help='是否拆分数据集')
    parser.add_argument('--train_ratio', type=float, default=0.8,
                        help='训练集比例')
    parser.add_argument('--val_ratio', type=float, default=0.1,
                        help='验证集比例')
    parser.add_argument('--test_ratio', type=float, default=0.1,
                        help='测试集比例')
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子')
    
    args = parser.parse_args()
    
    # 生成数据集报告
    generate_dataset_report(args.annotation_file, args.data_dir, args.output_dir)
    
    # 拆分数据集
    if args.split:
        split_dataset(
            args.annotation_file,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
            output_dir=os.path.join(args.output_dir, 'split'),
            random_seed=args.seed
        )

if __name__ == "__main__":
    main() 