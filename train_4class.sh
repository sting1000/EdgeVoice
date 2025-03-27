#!/bin/bash

# 清理缓存
rm -rf tmp/feature_cache
mkdir -p tmp/feature_cache
mkdir -p saved_models

# 设置训练参数
ANNOTATION_FILE="data/split/train_annotations.csv"
TEST_ANNOTATION_FILE="data/split/test_annotations.csv"
DATA_DIR="data"
MODEL_SAVE_PATH="saved_models/model_4class.pt"
ONNX_SAVE_PATH="saved_models/model_4class.onnx"

# 使用全面数据增强训练快速分类器
echo "开始训练4类快速分类器..."
python train.py \
  --model_type fast \
  --data_dir $DATA_DIR \
  --annotation_file $ANNOTATION_FILE \
  --model_save_path $MODEL_SAVE_PATH \
  --num_epochs 30 \
  --augment \
  --augment_prob 0.7 \
  --export_onnx \
  --onnx_save_path $ONNX_SAVE_PATH \
  --clear_cache

# 评估模型
echo "评估模型..."
python streaming_evaluation.py \
  --model_path $MODEL_SAVE_PATH \
  --annotation_file $TEST_ANNOTATION_FILE \
  --data_dir $DATA_DIR \
  --confidence_threshold 0.85

echo "训练和评估完成!"