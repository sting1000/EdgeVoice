#!/usr/bin/env python
# -*- coding: utf-8 -*-
# models/asr_train.py

"""
WeNet ASR模型训练工具脚本。
用于训练Wenet ASR模型，支持从指定数据目录和标注文件准备数据并训练模型。
"""

import os
import sys
import json
import time
import yaml
import shutil
import logging
import argparse
import subprocess
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path

# 导入工具函数
from utils.asr_utils import prepare_asr_data, cer_calculate
from config import *

# 设置日志
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("ASRTrain")

# Wenet安装路径检查
def check_wenet_install():
    """检查WeNet是否已正确安装
    
    Returns:
        bool: 是否安装
        str: 安装路径
    """
    try:
        import wenet
        wenet_path = os.path.dirname(os.path.dirname(wenet.__file__))
        logger.info(f"找到WeNet安装: {wenet_path}")
        return True, wenet_path
    except ImportError:
        logger.warning("未找到WeNet安装，无法直接训练模型。将尝试从环境变量获取路径。")
        wenet_path = os.environ.get("WENET_DIR", None)
        if wenet_path and os.path.exists(wenet_path):
            logger.info(f"从环境变量找到WeNet路径: {wenet_path}")
            return True, wenet_path
        return False, None

# 准备训练配置
def prepare_training_config(
    data_dir, 
    output_dir,
    model_config=None,
    train_config=None,
    model_type="conformer",
    language="zh"
):
    """准备WeNet训练配置
    
    Args:
        data_dir: 数据目录
        output_dir: 输出目录
        model_config: 模型配置参数
        train_config: 训练配置参数
        model_type: 模型类型，支持"conformer", "transformer"
        language: 语言，支持"zh", "en"
    
    Returns:
        config_path: 配置文件路径
    """
    logger.info(f"准备训练配置，模型类型: {model_type}，语言: {language}")
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 基础配置
    base_config = {
        # 通用配置
        "gpu": 0,
        "cmvn_file": None,
        "cmvn_file_type": "kaldi",
        "lang": language,
        
        # 特征配置
        "filter_conf": {
            "max_length": ASR_MAX_LENGTH,
            "min_length": ASR_MIN_LENGTH,
            "token_max_length": ASR_TOKEN_MAX_LENGTH,
            "token_min_length": ASR_TOKEN_MIN_LENGTH,
            "max_output_input_ratio": 0.1,
            "min_output_input_ratio": 0.05
        },
        "data_type": "sound",
        "data_conf": {
            "max_length": ASR_MAX_LENGTH,
            "min_length": ASR_MIN_LENGTH,
        },
        "dataset_conf": {
            "batch_size": ASR_BATCH_SIZE,
            "batch_type": "static",
            "max_length": ASR_MAX_LENGTH,
            "min_length": ASR_MIN_LENGTH,
            "sort": False
        },
        
        # 特征提取
        "fbank_conf": {
            "num_mel_bins": 80,
            "frame_length": 25,
            "frame_shift": 10,
            "dither": 0.1
        },
        "spec_aug": True,
        "spec_aug_conf": {
            "num_t_mask": 2,
            "num_f_mask": 2,
            "max_t": 50,
            "max_f": 10
        },
        
        # 词表与输入
        "input_dim": 80,
        "output_dim": None,  # 将由词表文件获取
        
        # 训练配置
        "max_epoch": ASR_MAX_EPOCHS,
        "accum_grad": ASR_ACCUM_GRAD,
        "grad_clip": 5.0,
        "lr": ASR_LR,
        "weight_decay": ASR_WEIGHT_DECAY,
        "checkpoint": {
            "kbest_n": 10,
            "save_interval": ASR_SAVE_INTERVAL
        },
        "scheduler_conf": {
            "warmup_steps": 8000,
            "normalize_steps": True
        },
        
        # 解码配置
        "beam_size": 10,
        "decoding_chunk_size": -1,
        "num_decoding_left_chunks": -1,
    }
    
    # 模型特定配置
    if model_type == "conformer":
        model_specific = {
            "model_conf": {
                "use_dynamic_chunk": True,
                "attention_heads": ASR_NUM_HEADS,
                "linear_units": 2048,
                "num_blocks": ASR_NUM_LAYERS,
                "dropout_rate": 0.1,
                "positional_dropout_rate": 0.1,
                "attention_dropout_rate": 0.1,
                "mask_type": "limited",
                "pos_enc_layer_type": "rel_pos",
                "cnn_module_kernel": 15,
            },
            "encoder": "conformer",
            "encoder_conf": {
                "output_size": 256,
                "attention_heads": ASR_NUM_HEADS,
                "linear_units": 2048,
                "num_blocks": ASR_NUM_LAYERS,
                "dropout_rate": 0.1,
                "positional_dropout_rate": 0.1,
                "attention_dropout_rate": 0.1,
                "input_layer": "conv2d",
                "pos_enc_layer_type": "rel_pos",
                "normalize_before": True,
                "cnn_module_kernel": 15,
                "use_dynamic_chunk": True,
                "use_cnn_module": True,
                "cnn_module_norm": "layer_norm",
                "activation_type": "swish"
            },
            "decoder": "transformer",
            "decoder_conf": {
                "attention_heads": ASR_NUM_HEADS,
                "linear_units": 2048,
                "num_blocks": ASR_NUM_DECODER_LAYERS,
                "dropout_rate": 0.1,
                "positional_dropout_rate": 0.1,
                "self_attention_dropout_rate": 0.1,
                "src_attention_dropout_rate": 0.1
            }
        }
    elif model_type == "transformer":
        model_specific = {
            "model_conf": {
                "use_dynamic_chunk": True,
                "attention_heads": ASR_NUM_HEADS,
                "linear_units": 2048,
                "num_blocks": ASR_NUM_LAYERS,
                "dropout_rate": 0.1,
                "positional_dropout_rate": 0.1,
                "attention_dropout_rate": 0.1,
                "mask_type": "limited"
            },
            "encoder": "transformer",
            "encoder_conf": {
                "output_size": 256,
                "attention_heads": ASR_NUM_HEADS,
                "linear_units": 2048,
                "num_blocks": ASR_NUM_LAYERS,
                "dropout_rate": 0.1,
                "positional_dropout_rate": 0.1,
                "attention_dropout_rate": 0.1,
                "input_layer": "conv2d",
                "normalize_before": True
            },
            "decoder": "transformer",
            "decoder_conf": {
                "attention_heads": ASR_NUM_HEADS,
                "linear_units": 2048,
                "num_blocks": ASR_NUM_DECODER_LAYERS,
                "dropout_rate": 0.1,
                "positional_dropout_rate": 0.1,
                "self_attention_dropout_rate": 0.1,
                "src_attention_dropout_rate": 0.1
            }
        }
    else:
        raise ValueError(f"不支持的模型类型: {model_type}")
    
    # 合并配置
    config = {**base_config, **model_specific}
    
    # 如果有额外的模型配置，合并
    if model_config:
        config["model_conf"].update(model_config)
        config["encoder_conf"].update(model_config)
    
    # 如果有额外的训练配置，合并
    if train_config:
        for k, v in train_config.items():
            config[k] = v
    
    # 设置数据和输出目录
    config.update({
        "train_data": f"{data_dir}/train",
        "dev_data": f"{data_dir}/dev",
        "dict": f"{data_dir}/lang_char.txt",
        "dir": output_dir,
    })
    
    # 保存配置
    config_path = os.path.join(output_dir, "train.yaml")
    with open(config_path, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    logger.info(f"配置文件已保存到: {config_path}")
    return config_path

# 运行Wenet训练脚本
def run_wenet_training(wenet_dir, config_path, num_gpus=1):
    """运行WeNet训练脚本
    
    Args:
        wenet_dir: WeNet安装目录
        config_path: 配置文件路径
        num_gpus: GPU数量
    
    Returns:
        bool: 是否成功
    """
    logger.info("开始训练WeNet ASR模型")
    
    # 训练脚本路径
    train_script = os.path.join(wenet_dir, "examples", "aishell", "s0", "run.sh")
    
    # 如果脚本不存在，尝试其他位置
    if not os.path.exists(train_script):
        train_script = os.path.join(wenet_dir, "wenet", "bin", "train.py")
        if not os.path.exists(train_script):
            logger.error(f"无法找到WeNet训练脚本: {train_script}")
            return False
    
    # 构建命令
    if train_script.endswith(".sh"):
        # shell脚本
        cmd = [
            "bash", train_script, 
            "--stage", "3", "--stop_stage", "6",
            "--config", config_path, 
            "--num_gpus", str(num_gpus),
            "--gpu_id", "0,1" if num_gpus > 1 else "0"
        ]
    else:
        # 直接使用Python脚本
        cmd = [
            "python", train_script,
            "--config", config_path,
            "--data_type", "raw",
            "--train_data", os.path.dirname(config_path) + "/train",
            "--dev_data", os.path.dirname(config_path) + "/dev",
            "--gpu", "0,1" if num_gpus > 1 else "0"
        ]
    
    # 运行命令
    logger.info(f"运行命令: {' '.join(cmd)}")
    try:
        process = subprocess.Popen(
            cmd, 
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )
        
        # 实时显示输出
        for line in process.stdout:
            logger.info(line.strip())
        
        # 等待进程完成
        process.wait()
        
        if process.returncode == 0:
            logger.info("训练成功完成")
            return True
        else:
            logger.error(f"训练失败，返回码: {process.returncode}")
            return False
    
    except Exception as e:
        logger.error(f"运行训练命令时出错: {e}")
        return False

# 导出模型为推理格式
def export_model(model_dir, output_dir=None, export_format="onnx"):
    """导出模型为推理格式
    
    Args:
        model_dir: 模型目录
        output_dir: 输出目录，默认为模型目录下的exported
        export_format: 导出格式，支持"onnx", "torchscript"
    
    Returns:
        str: 导出的模型路径
    """
    logger.info(f"导出模型为{export_format}格式")
    
    # 获取最新的checkpoint
    try:
        checkpoints = []
        for file in os.listdir(model_dir):
            if file.startswith("avg") and file.endswith(".pt"):
                checkpoints.append(os.path.join(model_dir, file))
        
        if not checkpoints:
            # 查找单个checkpoint
            for file in os.listdir(model_dir):
                if file.endswith(".pt") and not file.startswith("epoch"):
                    checkpoints.append(os.path.join(model_dir, file))
        
        if not checkpoints:
            logger.error(f"无法在{model_dir}目录中找到checkpoints")
            return None
        
        # 按修改时间排序，取最新的
        checkpoints.sort(key=lambda x: os.path.getmtime(x), reverse=True)
        latest_checkpoint = checkpoints[0]
        logger.info(f"使用最新的checkpoint: {latest_checkpoint}")
        
        # 设置输出目录
        if output_dir is None:
            output_dir = os.path.join(model_dir, "exported")
        os.makedirs(output_dir, exist_ok=True)
        
        # 检查WeNet安装
        has_wenet, wenet_dir = check_wenet_install()
        if not has_wenet:
            logger.error("无法导出模型，因为WeNet未安装")
            return None
        
        # 获取导出脚本路径
        if export_format == "onnx":
            export_script = os.path.join(wenet_dir, "wenet", "bin", "export_onnx.py")
        else:  # torchscript
            export_script = os.path.join(wenet_dir, "wenet", "bin", "export_jit.py")
        
        if not os.path.exists(export_script):
            logger.error(f"导出脚本不存在: {export_script}")
            return None
        
        # 构建命令
        conf_file = os.path.join(model_dir, "train.yaml")
        if not os.path.exists(conf_file):
            logger.error(f"配置文件不存在: {conf_file}")
            return None
        
        # 获取字典文件
        dict_file = None
        with open(conf_file, 'r', encoding='utf-8') as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
            dict_file = config.get("dict", None)
        
        if not dict_file or not os.path.exists(dict_file):
            # 尝试在model_dir下查找
            for file in ["lang_char.txt", "words.txt"]:
                temp_dict = os.path.join(model_dir, file)
                if os.path.exists(temp_dict):
                    dict_file = temp_dict
                    break
            
            if not dict_file:
                logger.error("无法找到字典文件")
                return None
        
        # 输出文件名
        if export_format == "onnx":
            output_file = os.path.join(output_dir, "model.onnx")
        else:
            output_file = os.path.join(output_dir, "model.zip")
        
        # 构建导出命令
        cmd = [
            "python", export_script,
            "--config", conf_file,
            "--checkpoint", latest_checkpoint,
            "--output_file", output_file
        ]
        
        # 对于onnx，可能需要额外参数
        if export_format == "onnx":
            cmd.extend(["--output_quant", "false"])
        
        # 运行导出命令
        logger.info(f"运行命令: {' '.join(cmd)}")
        process = subprocess.run(cmd, capture_output=True, text=True)
        
        if process.returncode == 0:
            logger.info(f"模型导出成功: {output_file}")
            
            # 复制字典文件到导出目录
            dict_output = os.path.join(output_dir, os.path.basename(dict_file))
            shutil.copy(dict_file, dict_output)
            logger.info(f"字典文件已复制到: {dict_output}")
            
            return output_file
        else:
            logger.error(f"模型导出失败: {process.stderr}")
            return None
        
    except Exception as e:
        logger.error(f"导出模型时出错: {e}")
        return None

def train_asr_model(
    data_dir,
    model_save_path,
    annotation_file=None,
    model_type="conformer",
    num_layers=2,
    num_heads=4,
    batch_size=None,
    epochs=None,
    use_gpu=True,
    num_gpus=1,
    seed=42,
    resume_from=None
):
    """训练WeNet ASR模型
    
    Args:
        data_dir: 音频数据目录
        model_save_path: 模型保存路径
        annotation_file: 标注文件
        model_type: 模型类型，支持"conformer", "transformer"
        num_layers: 模型层数
        num_heads: 注意力头数
        batch_size: 批次大小
        epochs: 训练轮数
        use_gpu: 是否使用GPU
        num_gpus: GPU数量
        seed: 随机种子
        resume_from: 恢复训练的模型路径
    
    Returns:
        str: 导出的模型路径
    """
    start_time = time.time()
    logger.info("开始ASR模型训练流程")
    
    # 检查WeNet是否已安装
    has_wenet, wenet_dir = check_wenet_install()
    if not has_wenet:
        logger.error("WeNet未安装，无法训练ASR模型")
        return None
    
    # 设置随机种子
    np.random.seed(seed)
    
    # 准备数据目录
    data_output_dir = os.path.join(model_save_path, "data")
    os.makedirs(data_output_dir, exist_ok=True)
    
    # 准备数据
    if annotation_file:
        logger.info(f"从标注文件准备ASR训练数据: {annotation_file}")
        prepare_asr_data(data_dir, annotation_file, data_output_dir)
    else:
        logger.warning("未提供标注文件，假设数据已准备完毕")
    
    # 检查数据准备情况
    train_data = os.path.join(data_output_dir, "train")
    dev_data = os.path.join(data_output_dir, "dev")
    dict_file = os.path.join(data_output_dir, "lang_char.txt")
    
    if not (os.path.exists(train_data) and os.path.exists(dict_file)):
        logger.error("数据准备不完整，请检查数据准备流程")
        return None
    
    # 准备训练配置
    model_config = {
        "num_blocks": num_layers,
        "attention_heads": num_heads
    }
    
    train_config = {}
    if batch_size:
        train_config["dataset_conf"] = {"batch_size": batch_size}
    if epochs:
        train_config["max_epoch"] = epochs
    
    # 创建模型保存目录
    model_dir = os.path.join(model_save_path, "model")
    os.makedirs(model_dir, exist_ok=True)
    
    # 准备训练配置
    config_path = prepare_training_config(
        data_output_dir, 
        model_dir,
        model_config=model_config,
        train_config=train_config,
        model_type=model_type
    )
    
    # 运行训练
    logger.info("开始WeNet模型训练")
    success = run_wenet_training(wenet_dir, config_path, num_gpus if use_gpu else 0)
    
    if not success:
        logger.error("模型训练失败")
        return None
    
    # 导出模型
    logger.info("导出模型为推理格式")
    export_path = export_model(model_dir)
    
    if export_path:
        logger.info(f"ASR模型训练完成，总耗时: {(time.time() - start_time) / 60:.2f}分钟")
        return export_path
    else:
        logger.error("模型导出失败")
        return None

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="WeNet ASR模型训练工具")
    
    parser.add_argument("--data_dir", type=str, required=True, help="音频数据目录")
    parser.add_argument("--annotation_file", type=str, help="标注文件路径(CSV)")
    parser.add_argument("--model_save_path", type=str, required=True, help="模型保存路径")
    parser.add_argument("--model_type", type=str, default="conformer", choices=["conformer", "transformer"], help="模型类型")
    parser.add_argument("--num_layers", type=int, default=ASR_NUM_LAYERS, help="模型层数")
    parser.add_argument("--num_heads", type=int, default=ASR_NUM_HEADS, help="注意力头数")
    parser.add_argument("--batch_size", type=int, help="批次大小")
    parser.add_argument("--epochs", type=int, help="训练轮数")
    parser.add_argument("--no_gpu", action="store_true", help="不使用GPU")
    parser.add_argument("--num_gpus", type=int, default=1, help="GPU数量")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--resume_from", type=str, help="恢复训练的模型路径")
    
    args = parser.parse_args()
    
    # 训练模型
    model_path = train_asr_model(
        data_dir=args.data_dir,
        model_save_path=args.model_save_path,
        annotation_file=args.annotation_file,
        model_type=args.model_type,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        batch_size=args.batch_size,
        epochs=args.epochs,
        use_gpu=not args.no_gpu,
        num_gpus=args.num_gpus,
        seed=args.seed,
        resume_from=args.resume_from
    )
    
    if model_path:
        print(f"\n训练成功！模型已保存到: {model_path}")
        return 0
    else:
        print("\n训练失败，请检查日志以获取更多信息。")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 