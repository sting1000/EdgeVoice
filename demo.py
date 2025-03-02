# demo.py  
import argparse  
import sounddevice as sd  
import numpy as np  
import queue  
import threading  
import time  
import sys  
from inference import IntentInferenceEngine  
from config import *  

class AudioStreamer:  
    def __init__(self, sample_rate=SAMPLE_RATE, buffer_size=MAX_COMMAND_DURATION_S*SAMPLE_RATE):  
        self.sample_rate = sample_rate  
        self.buffer_size = buffer_size  
        self.audio_queue = queue.Queue()  
        self.audio_buffer = np.zeros(buffer_size, dtype=np.float32)  
        self.is_active = False  
        self.stream = None  
        
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
    
    def start(self):  
        """启动音频流"""  
        self.is_active = True  
        
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
    
    def stop(self):  
        """停止音频流"""  
        if self.is_active:  
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
    
    def run(self):  
        """运行演示"""  
        self.running = True  
        
        # 启动音频流  
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
            if wakeword_thread.is_alive():  
                wakeword_thread.join(timeout=1.0)  
            
            print("演示已结束")  

if __name__ == "__main__":  
    parser = argparse.ArgumentParser(description='语音意图识别演示')  
    parser.add_argument('--fast_model', type=str, required=True, help='一级快速分类器路径')  
    parser.add_argument('--precise_model', type=str, help='二级精确分类器路径(可选)')  
    args = parser.parse_args()  
    
    # 创建并运行演示  
    demo = IntentDemo(args.fast_model, args.precise_model)  
    demo.run()