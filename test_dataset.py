#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
测试修复后的数据集加载，验证类别映射
"""

from streaming_dataset import StreamingAudioDataset
import pandas as pd
from config import INTENT_CLASSES

def test_dataset():
    """测试数据集能否正常加载并正确映射类别"""
    print('测试数据集加载和类别映射...')
    try:
        dataset = StreamingAudioDataset(
            annotation_file='data/split/train_annotations.csv',
            data_dir='data',
            streaming_mode=False
        )
        print(f'数据集大小: {len(dataset)}')
        print(f'映射后的意图标签: {dataset.intent_labels}')
        print(f'配置文件中的意图标签: {INTENT_CLASSES}')
        
        # 验证标签是否匹配
        if set(dataset.intent_labels) == set(INTENT_CLASSES):
            print('✅ 类别映射成功！数据集标签与配置文件匹配')
        else:
            print('❌ 类别映射失败！数据集标签与配置文件不匹配')
            print(f'差异: {set(dataset.intent_labels) ^ set(INTENT_CLASSES)}')
        
        # 显示原始数据中的类别分布
        df = pd.read_csv('data/split/train_annotations.csv')
        print('\n原始数据类别分布:')
        print(df['intent'].value_counts())
        
        print('\n映射后类别分布:')
        print(df['intent'].map(dataset.intent_mapping).value_counts())
        
        # 测试第一个样本
        print('\n测试第一个样本...')
        sample = dataset[0]
        print(f'样本键: {list(sample.keys())}')
        if 'features' in sample:
            print(f'特征形状: {sample["features"].shape}')
        print(f'标签ID: {sample["label"]}, 对应标签: {dataset.intent_labels[sample["label"]]}')
        
        # 测试前几个样本确保没有None
        print('\n测试前5个样本...')
        for i in range(min(5, len(dataset))):
            sample = dataset[i]
            if 'features' not in sample or sample['features'] is None:
                print(f'样本 {i} 特征为空!')
                return False
            label_name = dataset.intent_labels[sample['label']]
            print(f'样本 {i}: 特征形状 {sample["features"].shape}, 标签: {label_name}')
        
        print('\n✅ 所有测试通过!')
        return True
    except Exception as e:
        print(f'❌ 错误: {e}')
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_dataset() 