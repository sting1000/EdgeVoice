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

NUM_EPOCHS=1  # 训练轮数，增加可能提高性能，但耗时更长
BATCH_SIZE=128  # 批大小，减小可节省显存但训练更慢
CONFIDENCE_THRESHOLD=0.8  # 流式评估的置信度阈值


CONFORMER_LAYERS="8 10"
LEARNING_RATE="3e-4"  # 学习率，影响收敛速度和最终性能 (支持grid search)
DROPOUT="0.15"    # Dropout比例，防止过拟合(支持grid search)
KERNEL_SIZE="11"     # 卷积核大小，影响感受野(支持grid search)
WEIGHT_DECAY="0.002"  # 权重衰减，防止过拟合 (支持grid search)
# Loss Function
#  选择损失函数0(Cross Entrop),1(Label Smoothing), 2(Focal Loss) (支持grid search)
LOSS_FUNCTION="2"
# Label smoothing
#  标签平滑强度，推荐范围：0.05-0.2 (支持grid search)
LABEL_SMOOTHING="0.02"
# Focal loss
#  Focal loss 正负样本比例调节参数维度与函数output一致, 此处list需要以string形式输入 (支持grid search)
FOCAL_LOSS_ALPHA="[1,0.1,1,1,1,1,1,1]"
#  Focal loss 聚焦参数gamma维度与函数output一致, 此处list需要以string形式输入 (支持grid search)
FOCAL_LOSS_GAMMA="[1,2.5,1,1,1,1,1,1]"

# 显示核心配置
echo "=====================================
开始训练流式Conformer模型
训练轮数: $NUM_EPOCHS | 批大小: $BATCH_SIZE
学习率: $LEARNING_RATE | 权重衰减: $WEIGHT_DECAY
验证集: $VALID_ANNOTATION_FILE (独立说话者)
======================================="

# 设置环境变量优化性能
export CUDA_VISIBLE_DEVICES=4,5,6,7  # 使用指定GPU
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
  --num_layers $CONFORMER_LAYERS \
  --learning_rate $LEARNING_RATE \
  --dropout $DROPOUT \
  --kernel_size $KERNEL_SIZE \
  --weight_decay $WEIGHT_DECAY \
  --use_mixup \
  --progressive_training \
  --evaluate \
  --confidence_threshold $CONFIDENCE_THRESHOLD \
  --loss_function $LOSS_FUNCTION \
  --label_smoothing $LABEL_SMOOTHING \
  --focal_loss_alpha $FOCAL_LOSS_ALPHA \
  --focal_loss_gamma $FOCAL_LOSS_GAMMA
