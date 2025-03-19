#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
为了保持向后兼容性，从utils.feature_extraction导入所有功能
这个文件将被弃用，请直接使用utils.feature_extraction
"""

from utils.feature_extraction import (
    FeatureExtractor,
    extract_features,
    extract_features_streaming,
    standardize_features,
    random_crop_audio,
    streaming_feature_extractor
) 