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
    "PLAY_MUSIC",
    "PAUSE_MUSIC",
    "NEXT_SONG",
    "PREVIOUS_SONG",
    "VOLUME_UP",
    "VOLUME_DOWN",
]

# 每个意图类别的提示语列表
INTENT_PROMPTS = {
    "TAKE_PHOTO": [
        "拍照",
        "拍张照片",
        "给我拍照",
        "拍一张照",
        "帮我拍照片",
        "照相",
        "拍个照",
        "来张照片",
    ],
    
    "START_RECORDING": [
        "开始录像",
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
    
    "PLAY_MUSIC": [
        "播放音乐",
        "开始播放",
        "播放歌曲",
        "放音乐",
        "来点音乐",
        "播放",
        "放首歌",
        "开始音乐",
    ],
    
    "PAUSE_MUSIC": [
        "暂停播放",
        "暂停音乐",
        "停止播放",
        "暂停",
        "别放了",
        "停一下音乐",
        "暂停一下",
        "不放了",
    ],
    
    "NEXT_SONG": [
        "下一首",
        "下一曲",
        "切歌",
        "换一首",
        "播放下一首",
        "下一个",
        "下一首歌",
        "跳过这首",
    ],
    
    "PREVIOUS_SONG": [
        "上一首",
        "上一曲",
        "返回上一首",
        "回到上一首",
        "上一个",
        "上一首歌",
        "播放上一首",
        "回到前一首",
    ],
    
    "VOLUME_UP": [
        "增大音量",
        "提高音量",
        "声音大点",
        "音量大一点",
        "调高音量",
        "音量高一点",
        "大声一点",
        "声音调大",
    ],
    
    "VOLUME_DOWN": [
        "减小音量",
        "降低音量",
        "声音小点",
        "音量小一点",
        "调低音量",
        "音量低一点",
        "小声一点",
        "声音调小",
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