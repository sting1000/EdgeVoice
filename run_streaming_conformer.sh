#!/bin/bash

# 清理缓存
echo "清理特征缓存..."
rm -rf tmp/feature_cache
mkdir -p tmp/feature_cache
mkdir -p saved_models

# 核心参数设置 - 这些参数对模型性能影响最大
ANNOTATION_FILE="data/split/train_annotations.csv"  # 训练数据标注文件
VALID_ANNOTATION_FILE="data/split/val_annotations.csv"  # 验证数据标注文件(不同说话者)
TEST_ANNOTATION_FILE="data/split/test_annotations.csv"  # 测试数据标注文件
DATA_DIR="data"  # 数据目录
MODEL_SAVE_PATH="saved_models/streaming_conformer.pt"  # 模型保存路径
NUM_EPOCHS=30  # 训练轮数，增加可能提高性能，但耗时更长
BATCH_SIZE=16  # 批大小，减小可节省显存但训练更慢
LEARNING_RATE=1e-4  # 学习率，影响收敛速度和最终性能
WEIGHT_DECAY=0.001  # 权重衰减，防止过拟合
CONFIDENCE_THRESHOLD=0.8  # 流式评估的置信度阈值

# 显示核心配置
echo "=====================================
开始训练流式Conformer模型
训练轮数: $NUM_EPOCHS | 批大小: $BATCH_SIZE
学习率: $LEARNING_RATE | 权重衰减: $WEIGHT_DECAY
验证集: $VALID_ANNOTATION_FILE (独立说话者)
======================================="

# 设置环境变量优化性能
export CUDA_VISIBLE_DEVICES=0  # 使用指定GPU
export OMP_NUM_THREADS=8       # 优化OpenMP线程数
export MKL_NUM_THREADS=8       # 优化MKL线程数

# 执行训练脚本
# 各参数说明：
# --use_mixup：启用MixUp增强，提高泛化能力
# --progressive_training：启用渐进式训练，从短序列开始训练
# --evaluate：训练完成后评估模型
python train_streaming.py \
  --data_dir $DATA_DIR \
  --annotation_file $ANNOTATION_FILE \
  --valid_annotation_file $VALID_ANNOTATION_FILE \
  --test_annotation_file $TEST_ANNOTATION_FILE \
  --model_save_path $MODEL_SAVE_PATH \
  --num_epochs $NUM_EPOCHS \
  --batch_size $BATCH_SIZE \
  --learning_rate $LEARNING_RATE \
  --weight_decay $WEIGHT_DECAY \
  --use_mixup \
  --progressive_training \
  --evaluate \
  --confidence_threshold $CONFIDENCE_THRESHOLD

# 检查训练是否成功
if [ $? -eq 0 ]; then
    echo "流式Conformer模型训练完成！模型保存在: $MODEL_SAVE_PATH"
    
    # 导出ONNX模型（可选）
    echo "导出ONNX模型..."
    python export_onnx.py \
      --model_path $MODEL_SAVE_PATH \
      --model_type streaming \
      --onnx_save_path "${MODEL_SAVE_PATH%.pt}.onnx" \
      --dynamic_axes
    
    # 测试实时流式处理（可选）
    # if [ -f "data/test_samples/test_audio.wav" ]; then
    #     echo "测试流式处理..."
    #     python real_time_streaming_demo.py \
    #       --model_path $MODEL_SAVE_PATH \
    #       --buffer_size 1024 \
    #       --chunk_size 20 \
    #       --audio_file data/test_samples/test_audio.wav
    # fi
else
    echo "训练失败，请检查错误信息"
fi 