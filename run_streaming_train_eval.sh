#!/bin/bash

# 清理缓存，避免使用旧的特征缓存
echo "清理特征缓存..."
rm -rf tmp/feature_cache
mkdir -p tmp/feature_cache
mkdir -p saved_models

# 设置训练参数
ANNOTATION_FILE="data/split/train_annotations.csv"
TEST_ANNOTATION_FILE="data/split/test_annotations.csv"
DATA_DIR="data"
MODEL_SAVE_PATH="saved_models/streaming_model.pt"

# 设置训练轮数 - 测试环境使用较少轮数
# 测试模式: 预训练和微调各5轮，生产环境可增加到30+轮
PRETRAIN_EPOCHS=5
FINETUNE_EPOCHS=10
TOTAL_EPOCHS=$((PRETRAIN_EPOCHS + FINETUNE_EPOCHS))
ONNX_SAVE_PATH="saved_models/streaming_model.onnx"
echo "========================================"
echo "开始流式模型训练 (预训练 + 流式微调)"
echo "预训练轮数: $PRETRAIN_EPOCHS"
echo "微调轮数: $FINETUNE_EPOCHS"
echo "总轮数: $TOTAL_EPOCHS"
echo "========================================"

# 设置环境变量以优化性能
export CUDA_VISIBLE_DEVICES=0  # 使用指定GPU
export OMP_NUM_THREADS=8       # 优化OpenMP线程数
export MKL_NUM_THREADS=8       # 优化MKL线程数

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
    if [ -f "data/test_samples/test_audio.wav" ]; then
        # 如果测试音频文件存在，使用文件方式测试
        python real_time_streaming_demo.py \
          --model_path $MODEL_SAVE_PATH \
          --buffer_size 1024 \
          --audio_file data/test_samples/test_audio.wav
    else
        # 如果是无头环境或无法使用麦克风，跳过这一步
        echo "注意: 未找到默认测试音频文件，且未启用麦克风，跳过实时demo测试"
        echo "可以在后续手动测试：python real_time_streaming_demo.py --model_path $MODEL_SAVE_PATH --use_mic"
    fi
    
    echo "========================================"
    echo "所有任务完成!"
else
    echo "========================================"
    echo "训练或导出失败，请检查错误信息"
    echo "========================================"
fi 