import os
import pandas as pd
import numpy as np
import soundfile as sf
import uuid
import random
import glob
import librosa
import csv
import scipy.signal
from scipy.signal import fftconvolve
from tqdm import tqdm
import math
from config import *


# for f in *.{mp3,flac,aac,m4a,ogg,wma,ape,aiff,alac,amr}; do     [ -f "$f" ] || continue;     ffmpeg -y -i "$f" "${f%.*}.wav"; done

ENERGY_THRESHOLD = 0.008  # 能量阈值
TARGET_RMS = 0.012        # 增益后希望的RMS能量

# 动态获取配置参数，如果不存在则使用默认值
data_dir = globals().get('data_dir', 'data')
output_base = globals().get('output_base', 'data_augmented')  
base_dir = globals().get('base_dir', '.')
features_base = globals().get('features_base', 'features')
csv_name = globals().get('csv_name', 'annotations.csv')
aug_num = globals().get('aug_num', 2)

os.makedirs(output_base, exist_ok=True)


# ------------------文件处理相关-------------------------

# ==== 只保留音频存在的行 ====
def wav_exists(row):
    input_wav = os.path.join(data_dir, row['file_path'])
    exists = os.path.isfile(input_wav)
    # if not exists:
        # print(f"缺失音频文件: {input_wav}，已剔除")
    return exists

# ==== 处理后的数据生成对应描述的文件 ====
def save_wav(row, y, description, sr):
    # 新的文件名和路径
    basename = os.path.splitext(os.path.basename(row['file_path']))[0]
    dirname  = os.path.dirname(row['file_path'])
    output_dir = os.path.join(output_base, dirname)
    os.makedirs(output_dir, exist_ok=True)

    new_basename = f"{basename}_{description}"
    new_filename = f"{new_basename}.wav"
    output_wav = os.path.join(output_dir, new_filename)
    
    # 保存增强后的音频
    sf.write(output_wav, y, sr)
    # print(f"新文件 {output_wav}")
    
    # 构造新的一行元数据
    new_row = row.copy()
    new_row['file_path'] = os.path.relpath(output_wav, base_dir).replace('\\', '/')
    # 若有session_id可生成新id，时间戳可更新
    new_row['session_id'] = str(uuid.uuid4())
#     new_row['timestamp'] = datetime.now().strftime("%Y/%m/%d %H:%M")
    # 你也可以在user那一列添加"_aug"做区分
    new_row['user'] = str(row['user'])
    new_rows.append(new_row)


def save_new_df(new_rows):
    df_new = pd.DataFrame(new_rows)
    df_new.to_csv('annotations_augmented.csv', index=False, encoding='utf-8-sig')
    with open('annotations_augmented.csv', 'r', encoding='utf-8') as fin, \
         open('task_list.csv', 'w', encoding='utf-8', newline='') as fout:
        reader = csv.reader(fin)
        writer = csv.writer(fout)
        header = next(reader)
        writer.writerow(['wav_path', 'txt_path'])
        _output_base = output_base.replace('\\', '/')
        _features_base = features_base.replace('\\', '/')
        for row in reader:
            wav = row[0].strip().replace('\\', '/')
            if wav.startswith(_output_base + '/'):
                txt_path = wav.replace(_output_base + '/', _features_base + '/')
            else:
                txt_path = wav
            txt_path = os.path.splitext(txt_path)[0] + ".txt"
            writer.writerow([wav, txt_path])


# ------------------真实环境混响处理-------------------------

def apply_real_reverb(y, sr=16000):
    """
    使用真实的房间冲激响应(RIR)来添加混响效果
    
    Args:
        y (np.ndarray): 输入音频信号
        sr (int): 采样率
    
    Returns:
        np.ndarray: 添加混响后的音频信号
    """
    # 获取所有RIR文件
    rir_files = glob.glob("AIR_wav_files/*.wav")
    
    if not rir_files:
        # 如果没有RIR文件，返回原始音频
        print("警告：未找到RIR文件，跳过混响处理")
        return y
    
    # 随机选择一个RIR文件
    rir_path = random.choice(rir_files)
    
    try:
        # 加载RIR音频数据
        rir, rir_sr = librosa.load(rir_path, sr=sr)
        
        # 确保RIR是一维数组
        if rir.ndim > 1:
            rir = rir.mean(axis=1)
        
        # 归一化RIR能量，防止过度放大
        rir = rir / (np.sqrt(np.sum(rir**2)) + 1e-12)
        
        # 处理输入音频维度
        if y.ndim == 2 and y.shape[1] == 1:
            y = y[:, 0]
        
        if y.ndim == 1:
            # 单声道处理
            out = fftconvolve(y, rir, mode='full')
            # 截取到原始长度
            out = out[:len(y)]
        elif y.ndim == 2:
            # 多声道处理
            out = np.stack([fftconvolve(y[:, i], rir, mode='full')[:len(y)] 
                           for i in range(y.shape[1])], axis=1)
        else:
            raise ValueError(f"不支持的输入音频维度：{y.shape}")
        
        # 防止音量过大，保持相对音量比例
        peak_original = np.max(np.abs(y)) + 1e-12
        peak_reverb = np.max(np.abs(out)) + 1e-12
        out = out / peak_reverb * peak_original
        
        return out
        
    except Exception as e:
        print(f"应用混响时出错：{e}，返回原始音频")
        return y


# ------------------数据增强-------------------------

# 改变音速
def time_stretch(y, rate=1.0):
    """
    时域拉伸: rate>1变快(音频变短); rate<1变慢(音频变长)
    """
    return librosa.effects.time_stretch(y, rate=rate)

# 改变语调
def pitch_shift(y, sr, n_steps=0):
    """
    变调: n_steps为半音数, 正数升调, 负数降调
    """
    return librosa.effects.pitch_shift(y, sr=sr, n_steps=n_steps)

# 改变音量
def change_volume(y, gain_db=0):
    """
    音量放大或缩小/gain_db为dB变换量
    正数增大, 负数减小
    """
    factor = 10 ** (gain_db / 20)
    return y * factor


def mix_wav_with_noise(
    speech, noise_path, target_snr_db,
    frame_length=1024, hop_length=512, energy_db_threshold=None,
    sr=16000
):
    # 1. 读取噪声
    noise, _ = librosa.load(noise_path, sr=sr)
    speech_len = len(speech)
    
    if len(noise) < speech_len:
        # 若噪声长度不足，循环填充
        noise = np.tile(noise, int(np.ceil(speech_len / len(noise))))
        noise = noise[:speech_len]
    else:
        # 重点修改：随机起始点截取
        start = np.random.randint(0, len(noise) - speech_len + 1)
        noise = noise[start:start + speech_len]

    # 2. 计算人声能量
    frames = librosa.util.frame(
        speech, frame_length=frame_length, hop_length=hop_length
    ).T
    frame_energies = np.mean(frames**2, axis=1)
    frame_energies_db = 10 * np.log10(frame_energies + 1e-12)
    if energy_db_threshold is None:
        energy_db_threshold = np.max(frame_energies_db) - 10

    voiced_frames = frame_energies_db > energy_db_threshold
    if np.sum(voiced_frames) == 0:
        return speech

    Es = np.mean(frame_energies[voiced_frames])
    En = np.mean(noise ** 2)
    En_target = Es / (10 ** (target_snr_db / 10))
    scaling = np.sqrt(En_target / (En + 1e-12))
    noise_scaled = noise * scaling

    # 3. 混合
    mixed = speech + noise_scaled
    # 防止溢出
    max_amp = np.max(np.abs(mixed))
    if max_amp > 1:
        mixed = mixed / max_amp
    return mixed


NOISE_CATEGORIES = {
    "bike_ride": [        # 骑行风噪+偶有车流
        '1bike-ride.mp3', '6traffic1.mp3'
    ],
    "busy_road": [        # 市区、郊区公路、来往车流
        '6traffic1.mp3', '6traffic2.mp3'
    ],
    "meeting": [   # 办公室会议
        '2meeting1.mp3', '2meeting2.mp3'
    ],
    "cafe": [             # 餐厅嘈杂声、交谈杯碟
        '3cafe1.mp3', '3cafe2.mp3'
    ],
    "market": [           # 市场/菜市/小商品市场人士叫卖
        '4market1.mp3'
    ],
    "subway": [  # 地铁站台/进出站/到发报站
        '5subway2.mp3', '5subway1.mp3'
    ],
    "walk": [ # 脚步
        '7walk1.mp3', '7walk2.mp3', '7walk3.mp3'
    ],
    "city": [   # 跨场景融合-都市感 (多种urban噪音混)
        '3cafe1.mp3', '3cafe2.mp3', '4market1.mp3', '6traffic2.mp3'
    ],
    "rain": [   # 下雨场景
        '8rain1.mp3', '8rain2.mp3'
    ],
    "sea": [   # 海浪、海边场景
        '9sea1.mp3', '9sea2.mp3'
    ],
    "night": [   # 夜晚蛐蛐声
        '10night.mp3'
    ],
    "forest": [   # 森林小溪，鸟鸣等
        '11forest1.mp3', "11forest2.mp3"
    ],
    "kid": [   # 幼儿吵闹声
        '12kid1.mp3', "12kid2.mp3"
    ],
    "concert": [   # 演唱会欢呼鼓掌等
        '13concert1.mp3', "13concert2.mp3", "13concert3.mp3"
    ],
    "sports": [   # 运动：篮球、桌上足球、乒乓球
        '14sports1.mp3', "14sports2.mp3", "14sports3.mp3"
    ],
}
# 全部转为完整路径
_all_noise_files = sorted(glob.glob('noise/*.mp3'))
NOISE_CATEGORIES_FULLPATH = {}
for k, lst in NOISE_CATEGORIES.items():
    NOISE_CATEGORIES_FULLPATH[k] = [
        f for f in _all_noise_files if any(f.endswith('/'+fn) or f.endswith('\\'+fn) for fn in lst)
    ]
NOISE_CATEGORY_NAMES = list(NOISE_CATEGORIES_FULLPATH.keys()) # eg. ["outdoor","human_indoor","subway"]
# 2. Preset更改为"噪声类别名"引用
ENHANCE_PRESETS = {
    "quiet_office": {  # 办公室安静环境
        "possible_ops": ["speed", "pitch", "volume", "reverb", "noise"],
        "noise_categories": ["meeting"],
        "snr_db_range": (15, 20),
        "speed_range": (0.98, 1.02),
        "min_ops": 2,
        "max_ops": 5,
        "force_ops": ["noise","volume"],
    },
    "meeting": {  # 小型会议室
        "possible_ops": ["speed", "pitch", "volume", "reverb", "noise"],
        "noise_categories": ["meeting"],
        "snr_db_range": (13, 20),
        "speed_range": (0.97, 1.03),
        "min_ops": 2,
        "max_ops": 5,
        "force_ops": ["noise","volume"],
    },
    "cafe": {
        "possible_ops": ["speed", "pitch", "volume", "reverb", "noise"],
        "noise_categories": ["cafe"],
        "snr_db_range": (8, 15),
        "speed_range": (0.95, 1.05),
        "min_ops": 2,
        "max_ops": 5,
        "force_ops": ["noise","volume"],
    },
    "market": {
        "possible_ops": ["speed", "pitch", "volume", "reverb", "noise"],
        "noise_categories": ["market"],
        "snr_db_range": (6, 13),
        "speed_range": (0.94, 1.06),
        "min_ops": 2,
        "max_ops": 5,
        "force_ops": ["noise","volume"],
    },
    "busy_road": {
        "possible_ops": ["speed", "pitch", "volume", "noise"],
        "noise_categories": ["busy_road"],
        "snr_db_range": (5, 13),
        "speed_range": (0.96, 1.04),
        "min_ops": 2,
        "max_ops": 4,
        "force_ops": ["noise","volume"],
    },
    "bike_ride": {
        "possible_ops": ["speed", "pitch", "volume", "noise"],
        "noise_categories": ["bike_ride"],
        "snr_db_range": (5, 10),  # 降低SNR使其更具挑战性
        "speed_range": (0.95, 1.06),
        "min_ops": 2,
        "max_ops": 4,
        "force_ops": ["noise","volume"],
    },
    "subway": {
        "possible_ops": ["speed", "pitch", "volume", "reverb", "noise"],
        "noise_categories": ["subway"],
        "snr_db_range": (5, 13),
        "speed_range": (0.91, 1.09),
        "min_ops": 2,
        "max_ops": 5,
        "force_ops": ["noise","volume"],
    },
    "city": {
        "possible_ops": ["speed", "pitch", "volume", "noise"],
        "noise_categories": ["city"],
        "snr_db_range": (5, 16),
        "speed_range": (0.95, 1.07),
        "min_ops": 2,
        "max_ops": 4,
        "force_ops": ["noise","volume"],
    },
    "rain": {
        "possible_ops": ["speed", "pitch", "volume", "noise"],
        "noise_categories": ["rain"],
        "snr_db_range": (9, 15),
        "speed_range": (0.97, 1.03),
        "min_ops": 2,
        "max_ops": 4,
        "force_ops": ["noise","volume"],
    },
    "sea": {
        "possible_ops": ["speed", "pitch", "volume", "noise"],
        "noise_categories": ["sea"],
        "snr_db_range": (9, 18),
        "speed_range": (0.96, 1.04),
        "min_ops": 2,
        "max_ops": 4,
        "force_ops": ["noise","volume"],
    },
    "night": {
        "possible_ops": ["speed", "pitch", "volume", "noise"],
        "noise_categories": ["night"],
        "snr_db_range": (12, 20),
        "speed_range": (0.97, 1.01),
        "min_ops": 2,
        "max_ops": 3,
        "force_ops": ["noise","volume"],
    },
    "forest": {
        "possible_ops": ["speed", "pitch", "volume", "noise"],
        "noise_categories": ["forest"],
        "snr_db_range": (11, 18),
        "speed_range": (0.97, 1.03),
        "min_ops": 2,
        "max_ops": 4,
        "force_ops": ["noise","volume"],
    },
    "walk": {
        "possible_ops": ["speed", "pitch", "volume", "noise"],
        "noise_categories": ["walk"],
        "snr_db_range": (11, 18),
        "speed_range": (0.98, 1.03),
        "min_ops": 2,
        "max_ops": 4,
        "force_ops": ["noise","volume"],
    },
    "kid": {
        "possible_ops": ["speed", "pitch", "volume", "reverb", "noise"],
        "noise_categories": ["kid"],
        "snr_db_range": (9, 14),
        "speed_range": (0.93, 1.08),
        "min_ops": 2,
        "max_ops": 5,
        "force_ops": ["noise","volume"],
    },
    "concert": {
        "possible_ops": ["speed", "pitch", "volume", "reverb", "noise"],
        "noise_categories": ["concert"],
        "snr_db_range": (5, 10),
        "speed_range": (0.90, 1.12),
        "min_ops": 3,
        "max_ops": 5,
        "force_ops": ["noise","volume"],
    },
    "sports": {
        "possible_ops": ["speed", "pitch", "volume", "reverb", "noise"],
        "noise_categories": ["sports"],
        "snr_db_range": (8, 15),
        "speed_range": (0.95, 1.09),
        "min_ops": 2,
        "max_ops": 5,
        "force_ops": ["noise","volume"],
    },
    "clean": {
        "possible_ops": ["speed", "pitch", "volume"],
        "speed_range": (0.95, 1.09),
        "min_ops": 2,
        "max_ops": 3,
    },
}

# --- 用于根据噪声类别获取真实noise文件 ---
def get_noise_file(category):
    files = NOISE_CATEGORIES_FULLPATH.get(category, [])
    if not files:
        # 如果没有找到噪声文件，返回None
        print(f"警告：未找到类别 '{category}' 的噪声文件，跳过噪声处理")
        return None
    return random.choice(files)

# ---- 主增强方法 - 按照"Real-World First"原则重构 ----
def random_augment(
    y, sr,
    preset=None,         # 可传特定Preset名 (str) 或直接None
    global_aug_prob=0.7  # 叠加增强几率（提高多样性，部分样本无增广）
):
    """
    按照"Real-World First"原则进行音频增强：
    1. 源变换 (速度、音调、音量)
    2. 环境传播 (混响)
    3. 噪声叠加 (环境噪声)
    4. 最终裁剪 (防止溢出)
    """
    # 1. 有一定概率不做增强
    if random.random() > global_aug_prob:
        return y
    
    # 2. Preset选取
    if preset is None:
        preset = random.choice(list(ENHANCE_PRESETS.keys()))
    cfg = ENHANCE_PRESETS[preset]
    
    force_ops = set(cfg.get("force_ops", []))
    all_ops = set(cfg["possible_ops"])
    available_ops = list(all_ops - force_ops)
    
    num_force = len(force_ops)
    min_ops = cfg.get("min_ops", 1)
    max_ops = cfg.get("max_ops", min(len(cfg["possible_ops"]), 3))
    actual_min_ops = max(min_ops, num_force)
    actual_max_ops = max(max_ops, num_force)
    total_ops = random.randint(actual_min_ops, actual_max_ops)
    num_random_ops = max(total_ops - num_force, 0)
    random_ops = set(random.sample(available_ops, min(num_random_ops, len(available_ops))))
    ops = list(force_ops | random_ops)
    
    # 3. 按照"Real-World First"原则排序操作
    # 源变换操作
    source_ops = ["speed", "pitch", "volume"]
    # 环境传播操作
    env_ops = ["reverb"]
    # 噪声叠加操作
    noise_ops = ["noise"]
    
    # 分离不同类型的操作
    source_transforms = [op for op in ops if op in source_ops]
    env_transforms = [op for op in ops if op in env_ops]
    noise_transforms = [op for op in ops if op in noise_ops]
    
    # 4. 执行源变换（第一阶段）
    y_transformed = y.copy()
    
    for op in source_transforms:
        if op == "speed":
            rate = random.uniform(*cfg["speed_range"])
            y_transformed = time_stretch(y_transformed, rate)
        elif op == "pitch":
            n_steps = random.uniform(-1.2, 1.2)
            y_transformed = pitch_shift(y_transformed, sr, n_steps)
        elif op == "volume":
            gain_db = random.uniform(-6, 3)
            y_transformed = change_volume(y_transformed, gain_db)
    
    # 5. 执行环境传播（第二阶段）
    y_reverb = y_transformed
    for op in env_transforms:
        if op == "reverb":
            y_reverb = apply_real_reverb(y_reverb, sr)
    
    # 6. 执行噪声叠加（第三阶段）
    y_final = y_reverb
    for op in noise_transforms:
        if op == "noise":
            if "noise_categories" not in cfg: 
                continue
            cat = random.choice(cfg["noise_categories"])
            snr_db = random.uniform(*cfg["snr_db_range"])
            noise_file = get_noise_file(cat)
            if noise_file is not None:  # 只有在找到噪声文件时才应用
                y_final = mix_wav_with_noise(y_final, noise_file, snr_db, sr=sr)
    
    # 7. 最终裁剪防止溢出（第四阶段）
    y_aug = y_final / (np.max(np.abs(y_final)) + 1e-8)
    
    return y_aug


if __name__ == "__main__":
    csv_base = os.path.join(data_dir, csv_name)
    df = pd.read_csv(csv_base, encoding='utf-8')
    df = df[df.apply(wav_exists, axis=1)].reset_index(drop=True)
    print(f"原始csv总行数: {len(pd.read_csv(csv_base, encoding='utf-8'))}")
    print(f"过滤后df行数: {len(df)}")
    # 下面使用过滤后的df即可
    df_all = df
    new_rows = []

    for idx, row in tqdm(df_all.iterrows(), total=len(df_all), desc='Processing audio files'):
        input_wav = os.path.join(data_dir, row['file_path'])
        y, sr = librosa.load(input_wav, sr=16000)
        
        # 保存原始音频（不进行任何预处理）
        save_wav(row, y, 'origin', sr)

        # 根据intent决定增强倍数
        intent = str(row['intent'])
        if intent == 'TAKE_PHOTO':
            cur_aug = math.ceil(aug_num * 1.5)
        elif intent == 'CAPTURE_AND_DESCRIBE':
            cur_aug = math.ceil(aug_num * 3)
        else:
            cur_aug = aug_num
            
        # 对原始清洁音频进行增强
        for i in range(cur_aug):
            y_aug = random_augment(y, sr)
            save_wav(row, y_aug, f'aug{i}', sr)
    
    save_new_df(new_rows)

    input_csv = csv_base
    output_csv = os.path.join(data_dir, 'result.csv')
    df = pd.read_csv(input_csv, encoding='utf-8')
    rows = []
    for idx, row in df.iterrows():
        file_path = row['file_path'].replace('\\', '/')
        others = row.tolist()[1:]  # 其它列

        # 根据intent决定增强倍数
        intent = str(row['intent'])
        if intent == 'TAKE_PHOTO':
            cur_aug = math.ceil(aug_num * 1.5)
        elif intent == 'CAPTURE_AND_DESCRIBE':
            cur_aug = math.ceil(aug_num * 3)
        else:
            cur_aug = aug_num
        # 原始（origin）
        if file_path.endswith('.txt'):
            new_path = file_path.replace('.txt', '_origin.txt')
        elif file_path.endswith('.wav'):
            new_path = file_path.replace('.wav', '_origin.txt')
        else:
            raise ValueError(f"Unknown file_path suffix: {file_path}")
        rows.append([new_path] + others)

        # 数据增强
        for i in range(cur_aug):
            if file_path.endswith('.txt'):
                aug_path = file_path.replace('.txt', f'_aug{i}.txt')
            elif file_path.endswith('.wav'):
                aug_path = file_path.replace('.wav', f'_aug{i}.txt')
            else:
                raise ValueError(f"Unknown file_path suffix: {file_path}")
            rows.append([aug_path] + others)
    out_df = pd.DataFrame(rows, columns=df.columns)
    out_df.to_csv(output_csv, index=False, encoding='utf-8')

    # ---
    # NOTE FOR MODEL TRAINING:
    # This script prepares the time-domain audio data. For optimal Conformer model
    # performance, remember to apply SpecAugment in the training DataLoader.
    # SpecAugment operates on the frequency-domain features (e.g., Mel Spectrograms)
    # after they are computed from these generated .wav files.
    # ---
