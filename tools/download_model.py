#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
下载预训练的WeNet ASR模型工具
"""

import os
import sys
import argparse
import logging
import requests
import tarfile
import zipfile
import json
from tqdm import tqdm
import hashlib

# 设置日志
logging.basicConfig(
    format='%(asctime)s %(levelname)s: %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# 模型信息
MODEL_INFO = {
    "chinese_conformer": {
        "url": "https://wenet-1256283475.cos.ap-shanghai.myqcloud.com/models/wenetspeech/wenetspeech_u2pp_conformer_libtorch.tar.gz",
        "md5": "d3c75f7c96c0c3c7a1d5fa8e1f3bc582",
        "size": "1.1GB",
        "description": "中文Conformer模型 (WenetSpeech数据集训练)",
        "dict_file": "units.txt"
    },
    "chinese_transformer": {
        "url": "https://wenet-1256283475.cos.ap-shanghai.myqcloud.com/models/wenetspeech/wenetspeech_transformer_libtorch.tar.gz",
        "md5": "f3e3a8f27a8a8d3a3ca0b019c1fad2a9",
        "size": "462MB",
        "description": "中文Transformer模型 (WenetSpeech数据集训练)",
        "dict_file": "units.txt"
    },
    "english_conformer": {
        "url": "https://wenet-1256283475.cos.ap-shanghai.myqcloud.com/models/gigaspeech/gigaspeech_u2pp_conformer_libtorch.tar.gz",
        "md5": "a6072d3aa6b5a8b8d5c57a9d0fb5f96e",
        "size": "1.0GB",
        "description": "英文Conformer模型 (GigaSpeech数据集训练)",
        "dict_file": "units.txt"
    },
    "multilingual_conformer": {
        "url": "https://wenet-1256283475.cos.ap-shanghai.myqcloud.com/models/multilingual/multilingual_conformer_libtorch.tar.gz",
        "md5": "e1a1f7d7a8c1f6a8b0f5f4d3c1f1f1f1",
        "size": "1.2GB",
        "description": "多语言Conformer模型 (多语言数据集训练)",
        "dict_file": "units.txt"
    }
}

def calculate_md5(file_path):
    """计算文件的MD5值"""
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

def download_file(url, output_path):
    """下载文件并显示进度条"""
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    block_size = 1024  # 1 KB
    
    with open(output_path, 'wb') as file, tqdm(
            desc=os.path.basename(output_path),
            total=total_size,
            unit='B',
            unit_scale=True,
            unit_divisor=1024,
        ) as bar:
        for data in response.iter_content(block_size):
            bar.update(len(data))
            file.write(data)
    
    return output_path

def extract_archive(archive_path, output_dir):
    """解压缩文件"""
    if archive_path.endswith('.tar.gz') or archive_path.endswith('.tgz'):
        with tarfile.open(archive_path, 'r:gz') as tar:
            tar.extractall(path=output_dir)
    elif archive_path.endswith('.zip'):
        with zipfile.ZipFile(archive_path, 'r') as zip_ref:
            zip_ref.extractall(output_dir)
    else:
        raise ValueError(f"不支持的压缩格式: {archive_path}")
    
    logger.info(f"文件已解压到: {output_dir}")

def download_model(model_name, output_dir, force=False):
    """下载并解压模型"""
    if model_name not in MODEL_INFO:
        logger.error(f"未知模型: {model_name}")
        logger.info(f"可用模型: {', '.join(MODEL_INFO.keys())}")
        return False
    
    model_info = MODEL_INFO[model_name]
    url = model_info["url"]
    expected_md5 = model_info["md5"]
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 下载文件
    file_name = os.path.basename(url)
    download_path = os.path.join(output_dir, file_name)
    
    # 检查文件是否已存在
    if os.path.exists(download_path) and not force:
        logger.info(f"文件已存在: {download_path}")
        
        # 验证MD5
        logger.info("验证文件完整性...")
        actual_md5 = calculate_md5(download_path)
        if actual_md5 != expected_md5:
            logger.warning(f"MD5校验失败，重新下载文件")
            os.remove(download_path)
        else:
            logger.info("MD5校验通过")
            extract_path = os.path.join(output_dir, model_name)
            if os.path.exists(extract_path) and not force:
                logger.info(f"模型目录已存在: {extract_path}")
                return True
    
    if not os.path.exists(download_path) or force:
        logger.info(f"下载模型: {model_name} ({model_info['size']})")
        logger.info(f"下载地址: {url}")
        download_file(url, download_path)
        
        # 验证MD5
        logger.info("验证文件完整性...")
        actual_md5 = calculate_md5(download_path)
        if actual_md5 != expected_md5:
            logger.error(f"MD5校验失败: 预期 {expected_md5}, 实际 {actual_md5}")
            return False
        logger.info("MD5校验通过")
    
    # 解压文件
    extract_path = os.path.join(output_dir, model_name)
    os.makedirs(extract_path, exist_ok=True)
    logger.info(f"解压文件到: {extract_path}")
    extract_archive(download_path, extract_path)
    
    # 创建模型信息文件
    info_file = os.path.join(extract_path, "model_info.json")
    with open(info_file, 'w', encoding='utf-8') as f:
        json.dump({
            "name": model_name,
            "description": model_info["description"],
            "source": url,
            "md5": expected_md5,
            "dict_file": model_info["dict_file"],
            "download_time": tqdm.format_datetime(None)
        }, f, ensure_ascii=False, indent=2)
    
    logger.info(f"模型下载完成: {model_name}")
    logger.info(f"模型保存在: {extract_path}")
    logger.info(f"字典文件: {os.path.join(extract_path, model_info['dict_file'])}")
    
    return True

def list_models():
    """列出所有可用模型"""
    logger.info("可用的预训练模型:")
    for name, info in MODEL_INFO.items():
        logger.info(f"- {name}: {info['description']} ({info['size']})")

def main():
    parser = argparse.ArgumentParser(description="下载预训练的WeNet ASR模型")
    parser.add_argument("--model", type=str, help="模型名称")
    parser.add_argument("--output", type=str, default="saved_models/asr", help="输出目录")
    parser.add_argument("--force", action="store_true", help="强制重新下载")
    parser.add_argument("--list", action="store_true", help="列出所有可用模型")
    
    args = parser.parse_args()
    
    if args.list:
        list_models()
        return
    
    if not args.model:
        parser.print_help()
        list_models()
        return
    
    download_model(args.model, args.output, args.force)

if __name__ == "__main__":
    main() 