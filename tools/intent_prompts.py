"""
意图和提示语列表
提供8种意图类别的常用口语表达方式
"""

import random

# 意图类别
INTENT_CLASSES = [
    "TAKE_PHOTO",
    "START_RECORDING",
    "STOP_RECORDING",
    "CAPTURE_AND_DESCRIBE",
    "CAPTURE_AND_REMEMBER",
    "CAPTURE_SCAN_QR",
    "GET_BATTERY_LEVEL",
    "OTHERS",
]

# 每个意图类别的提示语列表
INTENT_PROMPTS = {
    "TAKE_PHOTO": [
        "拍照",
        "拍张照片",
        "拍个照吧",
        "给我拍照",
        "拍一张照",
        "帮我拍照片",
        "照相",
        "拍个照",
        "来张照片",
    ],
    
    "START_RECORDING": [
        "开始录像",
        "开始录视频",
        "录制视频",
        "开始拍视频",
        "开始录制",
        "开录像",
        "录视频",
        "开始拍摄",
        "录制",
    ],
    
    "STOP_RECORDING": [
        "停止录像",
        "结束录制",
        "不录了",
        "停止录制",
        "停止拍摄",
        "结束拍摄",
        "停录",
        "结束录像",
    ],
    
    "CAPTURE_AND_DESCRIBE": [
        "拍下这个并告诉我是什么",
        "这是什么东西",
        "识别一下这个",
        "帮我看看这是什么",
        "这个是什么",
        "告诉我这是什么",
        "拍下来并分析一下",
        "帮我识别这个物体",
    ],
    
    "CAPTURE_AND_REMEMBER": [
        "记住这个场景",
        "把这个保存下来",
        "记住这个",
        "记录这个画面",
        "保存这个场景",
        "记下这个",
        "把这个记下来",
        "保存这一刻",
    ],
    
    "CAPTURE_SCAN_QR": [
        "扫描二维码",
        "扫一下这个码",
        "扫码",
        "识别这个二维码",
        "读取这个码",
        "扫描这个二维码",
        "帮我扫一下码",
        "扫描这个",
    ],
    
    "GET_BATTERY_LEVEL": [
        "还剩多少电",
        "电量怎么样",
        "电池电量",
        "还有多少电",
        "电量还有多少",
        "剩余电量",
        "电池状态",
        "查看电量",
    ],
    
    "OTHERS": [
        "你好",
        "今天天气怎么样",
        "几点了",
        "谢谢",
        "取消",
        "退出",
        "返回",
        "不用了",
    ],
}

def get_random_prompt(intent):
    """获取指定意图的随机提示语"""
    if intent in INTENT_PROMPTS:
        return random.choice(INTENT_PROMPTS[intent])
    return "未知意图"

def get_all_prompts_for_intent(intent):
    """获取指定意图的所有提示语"""
    if intent in INTENT_PROMPTS:
        return INTENT_PROMPTS[intent]
    return [] 