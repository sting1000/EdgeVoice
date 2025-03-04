# demo.py  
import argparse  
import sounddevice as sd  
import numpy as np  
import queue  
import threading  
import time  
import sys  
import os
import soundfile as sf  
from inference import IntentInferenceEngine  
from config import *  
import librosa

class AudioStreamer:  
    def __init__(self, sample_rate=SAMPLE_RATE, buffer_size=MAX_COMMAND_DURATION_S*SAMPLE_RATE):  
        self.sample_rate = sample_rate  
        self.buffer_size = buffer_size  
        self.audio_queue = queue.Queue()  
        self.audio_buffer = np.zeros(buffer_size, dtype=np.float32)  
        self.is_active = False  
        self.stream = None  
        self.file_mode = False
        
    def callback(self, indata, frames, time, status):  
        """音频流回调函数"""  
        if status:  
            print(f"Stream callback status: {status}")  
        
        # 将新的音频数据添加到队列  
        self.audio_queue.put(indata.copy())  
    
    def update_buffer(self):  
        """从队列更新缓冲区"""  
        while self.is_active:  
            try:  
                # 非阻塞获取  
                new_audio = self.audio_queue.get(block=False)  
                
                # 移动缓冲区并添加新音频  
                new_size = new_audio.shape[0]  
                self.audio_buffer[:-new_size] = self.audio_buffer[new_size:]  
                self.audio_buffer[-new_size:] = new_audio.flatten()  
                
            except queue.Empty:  
                # 队列为空，短暂休眠  
                time.sleep(0.01)  
    
    def get_audio(self):  
        """获取当前缓冲区中的音频"""  
        return self.audio_buffer.copy()  
    
    def load_from_file(self, file_path):
        """从音频文件加载数据"""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"音频文件未找到: {file_path}")
            
        try:
            print(f"正在从文件加载音频: {file_path}")
            audio_data, file_sample_rate = sf.read(file_path)
            
            # 处理多声道情况，转换为单声道
            if len(audio_data.shape) > 1:
                audio_data = np.mean(audio_data, axis=1)
                
            # 如果采样率不匹配，进行重采样
            if file_sample_rate != self.sample_rate:
                print(f"警告: 文件采样率 ({file_sample_rate}Hz) 与系统设置 ({self.sample_rate}Hz) 不匹配，进行重采样")
                # 使用librosa进行重采样
                audio_data = librosa.resample(audio_data, orig_sr=file_sample_rate, target_sr=self.sample_rate)
            
            # 调整数据类型为float32
            if audio_data.dtype != np.float32:
                audio_data = audio_data.astype(np.float32)
            
            # 添加音频长度验证
            audio_duration = len(audio_data) / self.sample_rate
            if audio_duration > MAX_COMMAND_DURATION_S:
                print(f"警告: 音频长度 ({audio_duration:.2f}s) 超过最大命令长度 ({MAX_COMMAND_DURATION_S}s)")
            elif audio_duration < 0.5:  # 小于0.5秒的音频可能太短
                print(f"警告: 音频过短 ({audio_duration:.2f}s)，可能不足以包含完整命令")
            
            # 使用音频预处理器获取有效语音段
            from audio_preprocessing import AudioPreprocessor
            preprocessor = AudioPreprocessor(sample_rate=self.sample_rate)
            
            # 检测语音活动区域
            speech_segments = preprocessor.detect_voice_activity(audio_data)
            
            if speech_segments:
                # 如果检测到语音段，合并所有语音段
                processed_audio = np.array([])
                for start, end in speech_segments:
                    # 转换帧索引到样本索引
                    frame_shift = int(FRAME_SHIFT_MS * self.sample_rate / 1000)
                    frame_length = int(FRAME_LENGTH_MS * self.sample_rate / 1000)
                    sample_start = start * frame_shift
                    sample_end = min(end * frame_shift + frame_length, len(audio_data))
                    processed_audio = np.append(processed_audio, audio_data[sample_start:sample_end])
                
                print(f"检测到语音段，长度从 {len(audio_data)} 减少到 {len(processed_audio)} 采样点")
                audio_data = processed_audio
            else:
                print("未检测到明显的语音段，使用原始音频")
            
            # 智能处理音频长度
            if len(audio_data) > self.buffer_size:
                # 如果音频仍然太长，优先保留中间部分
                # 这样比简单截取末尾更合理，因为语音命令通常在中间部分
                start = (len(audio_data) - self.buffer_size) // 2
                self.audio_buffer = audio_data[start:start+self.buffer_size]
                print(f"音频超出缓冲区大小，截取中间 {self.buffer_size} 采样点")
            else:
                # 居中放置音频，而不是放在末尾
                # 这对于特征提取和VAD更为合理
                self.audio_buffer = np.zeros(self.buffer_size, dtype=np.float32)
                start_pos = (self.buffer_size - len(audio_data)) // 2
                self.audio_buffer[start_pos:start_pos+len(audio_data)] = audio_data
                print(f"音频小于缓冲区大小，居中放置")
            
            print(f"音频文件加载成功，原始长度: {len(audio_data)} 采样点，持续时间: {audio_duration:.2f}s")
            return True
            
        except Exception as e:
            print(f"加载音频文件时出错: {e}")
            return False
    
    def start(self, file_path=None):  
        """启动音频流或加载文件"""  
        self.is_active = True
        
        if file_path:
            # 文件模式
            self.file_mode = True
            return self.load_from_file(file_path)
        else:
            # 麦克风流模式
            self.file_mode = False
            
            # 启动更新缓冲区的线程  
            self.buffer_thread = threading.Thread(target=self.update_buffer)  
            self.buffer_thread.daemon = True  
            self.buffer_thread.start()  
            
            # 启动音频流  
            self.stream = sd.InputStream(  
                samplerate=self.sample_rate,  
                channels=1,  
                callback=self.callback  
            )  
            self.stream.start()
            return True
    
    def stop(self):  
        """停止音频流"""  
        if self.is_active and not self.file_mode:  
            self.is_active = False  
            
            if self.stream:  
                self.stream.stop()  
                self.stream.close()  
            
            # 等待缓冲区线程结束  
            if hasattr(self, 'buffer_thread') and self.buffer_thread.is_alive():  
                self.buffer_thread.join(timeout=1.0)  

class IntentDemo:  
    def __init__(self, fast_model_path, precise_model_path=None):  
        # 初始化推理引擎  
        self.engine = IntentInferenceEngine(fast_model_path, precise_model_path)  
        
        # 初始化音频流  
        self.audio_streamer = AudioStreamer()  
        
        # 状态标志  
        self.running = False  
        self.wakeword_detected = False  
        self.last_intent_time = 0  
        self.cooldown_period = 2.0  # 冷却时间，避免连续识别  
    
    def simulate_wakeword_detection(self):  
        """模拟唤醒词检测(在实际场景中应替换为真实的唤醒词检测)"""  
        while self.running:  
            user_input = input("输入'wake'来模拟唤醒，或'q'退出: ")  
            if user_input.lower() == 'wake':  
                print("检测到唤醒词! 请说出你的命令...")  
                self.wakeword_detected = True  
                self.last_intent_time = time.time()  
            elif user_input.lower() == 'q':  
                self.running = False  
            time.sleep(0.1)  
    
    def process_intent(self):  
        """意图处理循环"""  
        while self.running:  
            # 如果检测到唤醒词且冷却期已过  
            if self.wakeword_detected and (time.time() - self.last_intent_time) > self.cooldown_period:  
                # 获取当前音频  
                audio = self.audio_streamer.get_audio()  
                
                # 预测意图  
                result = self.engine.process_audio_stream(audio)  
                
                # 输出结果  
                intent = result['intent']  
                confidence = result['confidence']  
                path = result['path']  
                total_time = result['times']['total']  
                
                print(f"\n检测到意图: {intent}")  
                print(f"置信度: {confidence:.4f}")  
                print(f"使用路径: {path}")  
                print(f"处理时间: {total_time*1000:.2f}ms")  
                
                # 根据意图类型执行相应操作  
                if intent in ['TAKE_PHOTO', 'START_RECORDING', 'STOP_RECORDING', 'CAPTURE_SCAN_QR']:  
                    print(f"执行相机操作: {intent}")  
                
                # 重置状态  
                self.wakeword_detected = False  
                self.last_intent_time = time.time()  
            
            time.sleep(0.1)  
    
    def process_file(self, file_path):
        """处理单个音频文件"""
        print(f"正在处理文件: {file_path}")
        
        # 加载音频文件
        if not self.audio_streamer.start(file_path):
            print("文件加载失败，跳过处理")
            return False
        
        # 获取音频数据
        audio = self.audio_streamer.get_audio()
        
        # 预测意图
        print("正在处理音频...")
        start_time = time.time()
        result = self.engine.process_audio_stream(audio)
        
        # 输出结果
        intent = result['intent']
        confidence = result['confidence']
        path = result['path']
        total_time = result['times']['total']
        
        print(f"\n文件: {os.path.basename(file_path)}")
        print(f"检测到意图: {intent}")
        print(f"置信度: {confidence:.4f}")
        print(f"使用路径: {path}")
        print(f"处理时间: {total_time*1000:.2f}ms")
        print(f"总处理时间: {(time.time() - start_time)*1000:.2f}ms")
        
        return True
    
    def batch_process(self, directory):
        """批量处理目录中的所有音频文件"""
        if not os.path.isdir(directory):
            print(f"错误: '{directory}' 不是有效目录")
            return
        
        # 支持的音频格式
        audio_extensions = ['.wav', '.mp3', '.flac', '.ogg']
        
        # 查找目录中的所有音频文件
        audio_files = []
        for root, _, files in os.walk(directory):
            for file in files:
                if any(file.lower().endswith(ext) for ext in audio_extensions):
                    audio_files.append(os.path.join(root, file))
        
        if not audio_files:
            print(f"在 '{directory}' 中没有找到音频文件")
            return
        
        print(f"找到 {len(audio_files)} 个音频文件进行处理")
        
        # 处理每个文件
        success_count = 0
        for i, file_path in enumerate(audio_files, 1):
            print(f"\n[{i}/{len(audio_files)}] 处理文件: {file_path}")
            if self.process_file(file_path):
                success_count += 1
        
        print(f"\n批处理完成: 成功处理 {success_count}/{len(audio_files)} 个文件")
    
    def run(self, use_file=False, file_path=None, batch_mode=False):  
        """运行演示"""  
        self.running = True  
        
        if use_file:
            if batch_mode:
                # 批处理模式
                print(f"开始批处理目录: {file_path}")
                self.batch_process(file_path)
            else:
                # 单文件模式
                self.process_file(file_path)
        else:
            # 实时麦克风模式
            print("启动音频流...")  
            self.audio_streamer.start()  
            
            try:  
                # 启动唤醒词检测线程  
                wakeword_thread = threading.Thread(target=self.simulate_wakeword_detection)  
                wakeword_thread.daemon = True  
                wakeword_thread.start()  
                
                # 启动意图处理  
                self.process_intent()  
                
            except KeyboardInterrupt:  
                print("\n用户中断，正在关闭...")  
            finally:  
                # 清理资源  
                self.running = False  
                self.audio_streamer.stop()  
                
                # 等待线程结束  
                if 'wakeword_thread' in locals() and wakeword_thread.is_alive():  
                    wakeword_thread.join(timeout=1.0)  
        
        print("演示已结束")  

if __name__ == "__main__":  
    parser = argparse.ArgumentParser(description='语音意图识别演示')  
    parser.add_argument('--fast_model', type=str, required=True, help='一级快速分类器路径')  
    parser.add_argument('--precise_model', type=str, help='二级精确分类器路径(可选)')  
    parser.add_argument('--use_file', action='store_true', help='使用音频文件代替麦克风')
    parser.add_argument('--file_path', type=str, help='音频文件或目录路径')
    parser.add_argument('--batch_mode', action='store_true', help='批处理模式，处理目录中的所有音频文件')
    args = parser.parse_args()  
    
    # 如果指定了使用文件但未提供文件路径
    if args.use_file and not args.file_path:
        parser.error("使用--use_file时必须指定--file_path")
    
    # 如果指定了批处理模式但未提供目录
    if args.batch_mode and not args.file_path:
        parser.error("使用--batch_mode时必须指定--file_path为有效目录")
    
    # 创建并运行演示  
    demo = IntentDemo(args.fast_model, args.precise_model)  
    demo.run(use_file=args.use_file, file_path=args.file_path, batch_mode=args.batch_mode)