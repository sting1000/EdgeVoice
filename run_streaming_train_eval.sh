#!/bin/bash

# 创建必要的目录
mkdir -p tmp/feature_cache
mkdir -p saved_models

# 设置训练参数
ANNOTATION_FILE="data/train_annotations.csv"
TEST_ANNOTATION_FILE="data/test_annotations.csv"
DATA_DIR="data"
MODEL_SAVE_PATH="saved_models/streaming_model.pt"
# 恢复正常的训练轮数
PRETRAIN_EPOCHS=5
FINETUNE_EPOCHS=5
TOTAL_EPOCHS=$((PRETRAIN_EPOCHS + FINETUNE_EPOCHS))
ONNX_SAVE_PATH="saved_models/streaming_model.onnx"
echo "========================================"
echo "开始流式模型训练 (预训练 + 流式微调)"
echo "预训练轮数: $PRETRAIN_EPOCHS"
echo "微调轮数: $FINETUNE_EPOCHS"
echo "总轮数: $TOTAL_EPOCHS"
echo "========================================"

# 执行两阶段流式训练
python train.py \
  --model_type streaming \
  --data_dir $DATA_DIR \
  --annotation_file $ANNOTATION_FILE \
  --model_save_path $MODEL_SAVE_PATH \
  --num_epochs $TOTAL_EPOCHS \
  --pre_train_epochs $PRETRAIN_EPOCHS \
  --fine_tune_epochs $FINETUNE_EPOCHS \
  --export_onnx \
  --onnx_save_path $ONNX_SAVE_PATH

# 检查训练是否成功
if [ $? -eq 0 ]; then
    echo "========================================"
    echo "流式模型训练成功!"
    echo "模型保存路径: $MODEL_SAVE_PATH"
    echo "ONNX模型保存路径: $ONNX_SAVE_PATH"
    echo "========================================"
    
    # 验证ONNX文件是否存在
    if [ -f "$ONNX_SAVE_PATH" ]; then
        echo "ONNX文件已成功创建"
        echo "文件大小: $(du -h $ONNX_SAVE_PATH | cut -f1)"
    fi
    
    # 评估流式模型
    echo "开始评估流式模型..."
    python streaming_evaluation.py \
      --model_path $MODEL_SAVE_PATH \
      --annotation_file $TEST_ANNOTATION_FILE \
      --data_dir $DATA_DIR \
      --confidence_threshold 0.85
      
    # 测试实时流式处理demo
    echo "========================================"
    echo "测试实时流式处理demo..."
    python real_time_streaming_demo.py \
      --model_path $MODEL_SAVE_PATH \
      --buffer_size 1024
    
    echo "========================================"
    echo "所有任务完成!"
else
    echo "========================================"
    echo "训练或导出失败，请检查错误信息"
    echo "========================================"
fi 