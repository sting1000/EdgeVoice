"""
EdgeVoice 语音数据采集工具
提供用户友好的界面采集语音样本和标注
"""

import os
import sys
import time
import tkinter as tk
from tkinter import ttk, messagebox, StringVar
import threading
import pandas as pd
import datetime
import uuid
from pathlib import Path

# 导入自定义模块
from intent_prompts import INTENT_CLASSES, get_random_prompt, get_all_prompts_for_intent
from audio_utilities import AudioRecorder, generate_unique_filename

# 常量定义
DATA_DIR = "../data"
ANNOTATION_FILE = os.path.join(DATA_DIR, "annotations.csv")
RECORDINGS_PER_SESSION = 10  # 每个会话的录音数量
DEFAULT_FONT = ('Microsoft YaHei', 10)  # 默认字体
TITLE_FONT = ('Microsoft YaHei', 12, 'bold')  # 标题字体
PROMPT_FONT = ('Microsoft YaHei', 14, 'bold')  # 提示语字体

# 性别选项
GENDER_OPTIONS = ["男", "女", "其他", "未指定"]

# 年龄组选项
AGE_GROUP_OPTIONS = ["儿童(0-12)", "青少年(13-19)", "青年(20-35)", "中年(36-50)", "老年(51+)", "未指定"]

# 录音环境选项
ENVIRONMENT_OPTIONS = ["安静室内", "嘈杂室内", "室外", "移动中", "其他", "未指定"]

class AudioRecorderApp:
    """音频录制应用GUI"""
    
    def __init__(self, root):
        """初始化应用"""
        self.root = root
        self.root.title("EdgeVoice 语音数据采集工具")
        self.root.geometry("800x600")
        self.root.minsize(800, 600)
        
        # 设置样式
        style = ttk.Style()
        style.configure('TLabel', font=DEFAULT_FONT)
        style.configure('TButton', font=DEFAULT_FONT)
        style.configure('TFrame', background='#f0f0f0')
        
        # 音频录制器
        self.recorder = AudioRecorder()
        
        # 当前会话ID
        self.session_id = str(uuid.uuid4())
        
        # 当前会话记录数
        self.session_recordings = 0
        
        # 创建界面组件
        self.create_widgets()
        
        # 确保数据目录存在
        os.makedirs(DATA_DIR, exist_ok=True)
        
        # 初始化注释文件
        self.init_annotation_file()
        
        # 选择默认意图
        if INTENT_CLASSES:
            self.intent_var.set(INTENT_CLASSES[0])
            self.update_prompt()
        
    def create_widgets(self):
        """创建界面组件"""
        # 主框架
        main_frame = ttk.Frame(self.root, padding="20")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # 标题
        title_label = ttk.Label(main_frame, text="EdgeVoice 语音数据采集工具", font=TITLE_FONT)
        title_label.grid(row=0, column=0, columnspan=2, sticky="w", pady=(0, 20))
        
        # 左侧面板 - 用户信息
        left_frame = ttk.LabelFrame(main_frame, text="用户信息", padding="10")
        left_frame.grid(row=1, column=0, sticky="nsew", padx=(0, 10))
        
        # 用户名
        ttk.Label(left_frame, text="用户名:").grid(row=0, column=0, sticky="w", pady=5)
        self.username_var = StringVar()
        username_entry = ttk.Entry(left_frame, textvariable=self.username_var)
        username_entry.grid(row=0, column=1, sticky="ew", pady=5)
        
        # 性别
        ttk.Label(left_frame, text="性别:").grid(row=1, column=0, sticky="w", pady=5)
        self.gender_var = StringVar(value=GENDER_OPTIONS[-1])  # 默认"未指定"
        gender_combo = ttk.Combobox(left_frame, textvariable=self.gender_var, values=GENDER_OPTIONS)
        gender_combo.grid(row=1, column=1, sticky="ew", pady=5)
        
        # 年龄组
        ttk.Label(left_frame, text="年龄组:").grid(row=2, column=0, sticky="w", pady=5)
        self.age_group_var = StringVar(value=AGE_GROUP_OPTIONS[-1])  # 默认"未指定"
        age_group_combo = ttk.Combobox(left_frame, textvariable=self.age_group_var, values=AGE_GROUP_OPTIONS)
        age_group_combo.grid(row=2, column=1, sticky="ew", pady=5)
        
        # 录音环境
        ttk.Label(left_frame, text="录音环境:").grid(row=3, column=0, sticky="w", pady=5)
        self.environment_var = StringVar(value=ENVIRONMENT_OPTIONS[0])  # 默认"安静室内"
        environment_combo = ttk.Combobox(left_frame, textvariable=self.environment_var, values=ENVIRONMENT_OPTIONS)
        environment_combo.grid(row=3, column=1, sticky="ew", pady=5)
        
        # 会话信息
        ttk.Label(left_frame, text="会话ID:").grid(row=4, column=0, sticky="w", pady=5)
        session_label = ttk.Label(left_frame, text=self.session_id[:8] + "...")
        session_label.grid(row=4, column=1, sticky="w", pady=5)
        
        ttk.Label(left_frame, text="已录制:").grid(row=5, column=0, sticky="w", pady=5)
        self.recordings_label = ttk.Label(left_frame, text=f"0 / {RECORDINGS_PER_SESSION}")
        self.recordings_label.grid(row=5, column=1, sticky="w", pady=5)
        
        # 意图选择
        ttk.Label(left_frame, text="意图:").grid(row=6, column=0, sticky="w", pady=5)
        self.intent_var = StringVar()
        intent_combo = ttk.Combobox(left_frame, textvariable=self.intent_var, values=INTENT_CLASSES)
        intent_combo.grid(row=6, column=1, sticky="ew", pady=5)
        intent_combo.bind("<<ComboboxSelected>>", lambda e: self.update_prompt())
        
        # 随机提示按钮
        random_prompt_btn = ttk.Button(left_frame, text="随机提示", command=self.update_prompt)
        random_prompt_btn.grid(row=7, column=0, columnspan=2, sticky="ew", pady=10)
        
        # 说明
        instructions_frame = ttk.LabelFrame(left_frame, text="使用说明", padding="10")
        instructions_frame.grid(row=8, column=0, columnspan=2, sticky="nsew", pady=(10, 0))
        
        instructions_text = """
1. 填写用户信息
2. 选择意图类别
3. 点击"随机提示"获取提示语
4. 点击"开始录音"并朗读提示语
5. 录音完成后点击"停止录音"
6. 检查录音质量，满意后保存
7. 每个会话将录制10个样本
        """
        ttk.Label(instructions_frame, text=instructions_text, justify="left").pack(fill="both")
        
        # 右侧面板 - 录音控制
        right_frame = ttk.LabelFrame(main_frame, text="录音控制", padding="10")
        right_frame.grid(row=1, column=1, sticky="nsew")
        
        # 提示语显示
        prompt_frame = ttk.LabelFrame(right_frame, text="请朗读下面的文本", padding="10")
        prompt_frame.pack(fill="both", expand=True, pady=(0, 10))
        
        self.prompt_var = StringVar(value="请先选择意图类别")
        prompt_label = ttk.Label(prompt_frame, textvariable=self.prompt_var, font=PROMPT_FONT, wraplength=350)
        prompt_label.pack(fill="both", expand=True, padx=10, pady=10)
        
        # 录音状态
        self.status_var = StringVar(value="就绪")
        status_label = ttk.Label(right_frame, textvariable=self.status_var)
        status_label.pack(fill="x", pady=(0, 10))
        
        # 录音时长
        self.duration_var = StringVar(value="时长: 0.0 秒")
        duration_label = ttk.Label(right_frame, textvariable=self.duration_var)
        duration_label.pack(fill="x", pady=(0, 20))
        
        # 录音控制按钮
        buttons_frame = ttk.Frame(right_frame)
        buttons_frame.pack(fill="x")
        
        self.record_btn = ttk.Button(buttons_frame, text="开始录音", command=self.start_recording)
        self.record_btn.pack(side="left", fill="x", expand=True, padx=5)
        
        self.stop_btn = ttk.Button(buttons_frame, text="停止录音", command=self.stop_recording, state="disabled")
        self.stop_btn.pack(side="left", fill="x", expand=True, padx=5)
        
        play_btn = ttk.Button(buttons_frame, text="播放录音", command=self.play_audio)
        play_btn.pack(side="left", fill="x", expand=True, padx=5)
        
        save_btn = ttk.Button(buttons_frame, text="保存录音", command=self.save_audio)
        save_btn.pack(side="left", fill="x", expand=True, padx=5)
        
        # 设置网格权重
        main_frame.columnconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=2)
        main_frame.rowconfigure(1, weight=1)
        
        left_frame.columnconfigure(1, weight=1)
        
    def init_annotation_file(self):
        """初始化注释文件"""
        # 检查是否已存在
        if not os.path.exists(ANNOTATION_FILE):
            # 创建空的数据框
            columns = ["file_path", "intent", "transcript", "gender", "age_group", 
                      "environment", "session_id", "timestamp"]
            df = pd.DataFrame(columns=columns)
            
            # 保存到CSV，使用UTF-8-SIG编码使Excel能正确显示中文
            df.to_csv(ANNOTATION_FILE, index=False, encoding='utf-8-sig')
            print(f"已创建注释文件: {ANNOTATION_FILE}")
    
    def update_prompt(self):
        """更新提示语"""
        intent = self.intent_var.get()
        if intent:
            prompt = get_random_prompt(intent)
            self.prompt_var.set(prompt)
        else:
            self.prompt_var.set("请先选择意图类别")
    
    def start_recording(self):
        """开始录音"""
        # 更新UI状态
        self.record_btn.config(state="disabled")
        self.stop_btn.config(state="normal")
        self.status_var.set("正在录音...")
        
        # 在新线程中开始录音
        self.recorder.recording_thread = threading.Thread(target=self._record)
        self.recorder.recording_thread.daemon = True
        self.recorder.recording_thread.start()
    
    def _record(self):
        """录音线程"""
        try:
            self.recorder.start_recording()
        except Exception as e:
            print(f"录音线程出错: {e}")
            # 确保在主线程更新UI
            self.root.after(0, lambda: self.status_var.set("录音出错"))
            self.root.after(0, lambda: self.record_btn.config(state="normal"))
            self.root.after(0, lambda: self.stop_btn.config(state="disabled"))
    
    def stop_recording(self):
        """停止录音"""
        # 停止录音
        self.recorder.stop_recording()
        
        # 更新UI状态
        self.record_btn.config(state="normal")
        self.stop_btn.config(state="disabled")
        self.status_var.set("录音已停止")
        
        # 更新时长
        duration = self.recorder.get_audio_length()
        self.duration_var.set(f"时长: {duration:.1f} 秒")
    
    def play_audio(self):
        """播放录音"""
        if not hasattr(self.recorder, 'audio_data') or self.recorder.audio_data is None:
            messagebox.showinfo("提示", "没有可播放的录音")
            return
        
        # 在新线程中播放音频
        self.status_var.set("正在播放...")
        threading.Thread(target=self._play_audio).start()
    
    def _play_audio(self):
        """音频播放线程"""
        self.recorder.play_audio()
        self.root.after(0, lambda: self.status_var.set("就绪"))
    
    def save_audio(self):
        """保存录音和注释"""
        if not hasattr(self.recorder, 'frames') or not self.recorder.frames:
            messagebox.showinfo("提示", "没有可保存的录音")
            return
        
        intent = self.intent_var.get()
        if not intent:
            messagebox.showwarning("警告", "请先选择意图类别")
            return
        
        # 生成文件名
        file_path = generate_unique_filename(DATA_DIR, intent)
        
        # 保存音频文件
        if self.recorder.save_audio(file_path):
            # 准备注释数据
            data = {
                "file_path": os.path.relpath(file_path, DATA_DIR),
                "intent": intent,
                "transcript": self.prompt_var.get(),
                "gender": self.gender_var.get(),
                "age_group": self.age_group_var.get(),
                "environment": self.environment_var.get(),
                "session_id": self.session_id,
                "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            
            try:
                # 添加到注释文件 - 使用UTF-8-SIG编码以兼容Excel
                if os.path.exists(ANNOTATION_FILE) and os.path.getsize(ANNOTATION_FILE) > 0:
                    try:
                        # 先尝试读取现有文件
                        df = pd.read_csv(ANNOTATION_FILE, encoding='utf-8-sig')
                    except UnicodeDecodeError:
                        # 如果读取失败，尝试其他编码
                        df = pd.read_csv(ANNOTATION_FILE, encoding='utf-8')
                else:
                    # 创建新的空DataFrame
                    columns = ["file_path", "intent", "transcript", "gender", "age_group", 
                              "environment", "session_id", "timestamp"]
                    df = pd.DataFrame(columns=columns)
                
                # 添加新行
                df = pd.concat([df, pd.DataFrame([data])], ignore_index=True)
                
                # 保存为UTF-8-SIG格式，使Excel能正确显示中文
                df.to_csv(ANNOTATION_FILE, index=False, encoding='utf-8-sig')
                
                # 更新会话计数
                self.session_recordings += 1
                self.recordings_label.config(text=f"{self.session_recordings} / {RECORDINGS_PER_SESSION}")
                
                # 更新UI
                self.status_var.set("录音已保存")
                messagebox.showinfo("成功", f"录音已保存: {os.path.basename(file_path)}")
                
                # 检查是否需要创建新会话
                if self.session_recordings >= RECORDINGS_PER_SESSION:
                    self.create_new_session()
                
                # 更新提示语
                self.update_prompt()
            except Exception as e:
                print(f"保存注释数据时出错: {e}")
                messagebox.showerror("错误", f"保存注释数据失败: {str(e)}")
        else:
            messagebox.showerror("错误", "保存录音失败")
    
    def create_new_session(self):
        """创建新会话"""
        self.session_id = str(uuid.uuid4())
        self.session_recordings = 0
        
        # 更新UI
        messagebox.showinfo("提示", "当前会话已完成，已创建新会话")
        
        # 更新显示
        self.recordings_label.config(text=f"0 / {RECORDINGS_PER_SESSION}")
        self.root.title(f"EdgeVoice 语音数据采集工具 - 会话: {self.session_id[:8]}...")

def main():
    """主函数"""
    root = tk.Tk()
    app = AudioRecorderApp(root)
    root.mainloop()

if __name__ == "__main__":
    main() 