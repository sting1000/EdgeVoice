#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
测试修复后的数据集加载
"""

from streaming_dataset import StreamingAudioDataset
import pandas as pd

def test_dataset():
    """测试数据集能否正常加载"""
    print('测试数据集加载...')
    try:
        dataset = StreamingAudioDataset(
            annotation_file='data/split/train_annotations.csv',
            data_dir='data',
            streaming_mode=False
        )
        print(f'数据集大小: {len(dataset)}')
        print(f'意图标签: {dataset.intent_labels}')
        
        # 测试第一个样本
        print('测试第一个样本...')
        sample = dataset[0]
        print(f'样本键: {list(sample.keys())}')
        if 'features' in sample:
            print(f'特征形状: {sample["features"].shape}')
        print('第一个样本测试通过!')
        
        # 测试前几个样本确保没有None
        print('测试前5个样本...')
        for i in range(min(5, len(dataset))):
            sample = dataset[i]
            if 'features' not in sample or sample['features'] is None:
                print(f'样本 {i} 特征为空!')
                return False
            print(f'样本 {i}: 特征形状 {sample["features"].shape}')
        
        print('所有测试通过!')
        return True
    except Exception as e:
        print(f'错误: {e}')
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_dataset() 