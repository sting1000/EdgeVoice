#!/bin/bash

# 清理缓存
rm -rf tmp/feature_cache
mkdir -p tmp/feature_cache
mkdir -p saved_models

# 参数设置
ANNOTATION_FILE="data/split/train_annotations.csv"
TEST_ANNOTATION_FILE="data/split/test_annotations.csv"
DATA_DIR="data"
MODEL_SAVE_PATH="saved_models/streaming_conformer.pt"
NUM_EPOCHS=40
BATCH_SIZE=16
LEARNING_RATE=2e-5
WEIGHT_DECAY=0.03

# 显示配置
echo "======================================"
echo "训练流式Conformer模型"
echo "======================================"
echo "训练数据: $ANNOTATION_FILE"
echo "测试数据: $TEST_ANNOTATION_FILE"
echo "模型保存: $MODEL_SAVE_PATH"
echo "批量大小: $BATCH_SIZE"
echo "学习率: $LEARNING_RATE"
echo "训练轮数: $NUM_EPOCHS"
echo "权重衰减: $WEIGHT_DECAY"
echo "======================================"

# 设置环境变量
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8

# 执行训练脚本
python train_streaming.py \
  --data_dir $DATA_DIR \
  --annotation_file $ANNOTATION_FILE \
  --test_annotation_file $TEST_ANNOTATION_FILE \
  --model_save_path $MODEL_SAVE_PATH \
  --num_epochs $NUM_EPOCHS \
  --batch_size $BATCH_SIZE \
  --learning_rate $LEARNING_RATE \
  --weight_decay $WEIGHT_DECAY \
  --use_mixup \
  --use_label_smoothing \
  --progressive_training \
  --evaluate \
  --confidence_threshold 0.75

# 检查训练是否成功
if [ -f "$MODEL_SAVE_PATH" ]; then
  MODEL_SIZE=$(du -h $MODEL_SAVE_PATH | cut -f1)
  echo "训练成功! 模型已保存到 $MODEL_SAVE_PATH (大小: $MODEL_SIZE)"
  
  # 导出ONNX模型和测试流式推理
  echo "导出ONNX模型..."
  python export_onnx.py \
    --model_path $MODEL_SAVE_PATH \
    --model_type streaming \
    --onnx_save_path "saved_models/streaming_conformer.onnx" \
    --dynamic_axes
  
  # 测试实时流式处理
  TEST_AUDIO="data/test_samples/test_command.wav"
  if [ -f "$TEST_AUDIO" ]; then
    echo "测试实时流式处理..."
    python real_time_streaming_demo.py \
      --model_path $MODEL_SAVE_PATH \
      --audio_file $TEST_AUDIO \
      --buffer_size 0.2
  else
    echo "警告: 测试音频文件 $TEST_AUDIO 不存在，跳过实时测试"
  fi
else
  echo "训练失败! 模型文件 $MODEL_SAVE_PATH 不存在"
  exit 1
fi 