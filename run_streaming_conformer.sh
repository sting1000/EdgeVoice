#!/bin/bash

# 清理缓存
echo "清理特征缓存..."
rm -rf tmp/feature_cache
mkdir -p tmp/feature_cache
mkdir -p saved_models

# 设置参数
ANNOTATION_FILE="data/split/train_annotations.csv"
TEST_ANNOTATION_FILE="data/split/test_annotations.csv"
DATA_DIR="data"
MODEL_SAVE_PATH="saved_models/streaming_conformer.pt"
NUM_EPOCHS=30
BATCH_SIZE=32
LEARNING_RATE=0.0002
WEIGHT_DECAY=0.01

# 显示配置
echo "=============================================="
echo "开始训练优化后的流式Conformer模型"
echo "训练轮数: $NUM_EPOCHS"
echo "批大小: $BATCH_SIZE"
echo "学习率: $LEARNING_RATE"
echo "权重衰减: $WEIGHT_DECAY"
echo "特征维度: $(( 20 * 3 ))=60" # N_MFCC * 3
echo "流式参数: CHUNK=15, STEP=5, CACHE=60"
echo "=============================================="

# 设置环境变量以优化性能
export CUDA_VISIBLE_DEVICES=0  # 使用指定GPU
export OMP_NUM_THREADS=8       # 优化OpenMP线程数
export MKL_NUM_THREADS=8       # 优化MKL线程数

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
  --progressive_training \
  --evaluate \
  --confidence_threshold 0.85

# 检查训练是否成功
if [ $? -eq 0 ]; then
    echo "=============================================="
    echo "流式Conformer模型训练和评估成功!"
    echo "模型保存路径: $MODEL_SAVE_PATH"
    
    # 验证模型文件是否存在
    if [ -f "$MODEL_SAVE_PATH" ]; then
        echo "模型文件已成功创建"
        echo "文件大小: $(du -h $MODEL_SAVE_PATH | cut -f1)"
        
        # 导出ONNX模型
        echo "开始导出ONNX模型..."
        python export_onnx.py \
          --model_path $MODEL_SAVE_PATH \
          --onnx_path "${MODEL_SAVE_PATH%.pt}.onnx" \
          --dynamic_axes
          
        # 测试实时流式处理
        echo "=============================================="
        echo "测试实时流式处理demo..."
        if [ -f "data/test_samples/test_audio.wav" ]; then
            # 如果测试音频文件存在，使用文件方式测试
            python real_time_streaming_demo.py \
              --model_path $MODEL_SAVE_PATH \
              --buffer_size 1024 \
              --chunk_size 15 \
              --audio_file data/test_samples/test_audio.wav
        else
            # 如果没有测试文件
            echo "注意: 未找到默认测试音频文件，请准备测试文件或使用麦克风模式"
            echo "可以后续手动测试：python real_time_streaming_demo.py --model_path $MODEL_SAVE_PATH --use_mic --chunk_size 15"
        fi
    else
        echo "警告: 模型文件未创建，请检查错误"
    fi
    
    echo "=============================================="
    echo "所有任务完成!"
else
    echo "=============================================="
    echo "训练或评估失败，请检查错误信息"
    echo "=============================================="
fi 