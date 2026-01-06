"""
音频处理模块
用于音频格式转换、质量验证等
"""

import os
import librosa
import soundfile as sf
import numpy as np
from pathlib import Path
from typing import Tuple, Optional, List
import logging

logger = logging.getLogger(__name__)


class AudioProcessor:
    """音频处理器"""
    
    def __init__(
        self,
        target_sample_rate: int = 22050,
        target_channels: int = 1,
        min_duration: float = 1.0,
        max_duration: float = 30.0
    ):
        """
        初始化音频处理器
        
        Args:
            target_sample_rate: 目标采样率
            target_channels: 目标声道数 (1=单声道)
            min_duration: 最小时长（秒）
            max_duration: 最大时长（秒）
        """
        self.target_sample_rate = target_sample_rate
        self.target_channels = target_channels
        self.min_duration = min_duration
        self.max_duration = max_duration
    
    def load_audio(self, file_path: str) -> Tuple[np.ndarray, int]:
        """
        加载音频文件
        
        Args:
            file_path: 音频文件路径
            
        Returns:
            (音频数据, 采样率)
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"音频文件不存在: {file_path}")
        
        # 检查是否是m4a文件，尝试使用pydub（如果可用）
        if file_path.lower().endswith('.m4a'):
            try:
                from pydub import AudioSegment
                # 使用pydub加载m4a
                audio_seg = AudioSegment.from_file(file_path, format="m4a")
                # 转换为numpy数组
                samples = audio_seg.get_array_of_samples()
                audio = np.array(samples, dtype=np.float32)
                # 如果是立体声，重新整形
                if audio_seg.channels == 2:
                    audio = audio.reshape((-1, 2)).T
                else:
                    audio = audio.reshape((1, -1))
                # 归一化到[-1, 1]
                audio = audio / (1 << (audio_seg.sample_width * 8 - 1))
                sr = audio_seg.frame_rate
                logger.info(f"使用pydub成功加载m4a文件: {file_path}")
                return audio, sr
            except ImportError:
                # pydub未安装，静默使用librosa
                pass
            except Exception as e:
                # pydub加载失败，静默使用librosa
                pass
        
        # 使用librosa加载（支持大多数格式）
        try:
            audio, sr = librosa.load(file_path, sr=None, mono=False)
            return audio, sr
        except Exception as e:
            error_msg = str(e)
            if file_path.lower().endswith('.m4a'):
                # 检查是否是ffmpeg缺失
                if "ffmpeg" in error_msg.lower() or "WinError 2" in error_msg or "系统找不到指定的文件" in error_msg:
                    raise ValueError(
                        f"无法加载m4a文件 {file_path}\n"
                        f"\n原因：缺少ffmpeg（m4a格式需要ffmpeg支持）\n"
                        f"\n解决方案（选择其一）：\n"
                        f"  方案1：安装ffmpeg（推荐）\n"
                        f"    Windows: 下载 https://ffmpeg.org/download.html\n"
                        f"    或使用chocolatey: choco install ffmpeg\n"
                        f"    或使用scoop: scoop install ffmpeg\n"
                        f"    安装后需要将ffmpeg添加到系统PATH环境变量\n"
                        f"\n  方案2：使用在线工具或VLC等软件先将m4a转换为wav格式\n"
                        f"    然后将wav文件放入 data/origdata 目录\n"
                        f"\n  方案3：使用ffmpeg命令行手动转换\n"
                        f"    ffmpeg -i input.m4a -ar 22050 -ac 1 output.wav"
                    )
                raise ValueError(
                    f"无法加载m4a文件 {file_path}: {error_msg}\n"
                    f"请检查文件是否损坏，或安装ffmpeg: https://ffmpeg.org/download.html"
                )
            raise ValueError(f"无法加载音频文件 {file_path}: {error_msg}")
    
    def validate_audio(
        self,
        audio: np.ndarray,
        sample_rate: int
    ) -> Tuple[bool, Optional[str]]:
        """
        验证音频质量
        
        Args:
            audio: 音频数据
            sample_rate: 采样率
            
        Returns:
            (是否有效, 错误信息)
        """
        # 计算时长
        if len(audio.shape) > 1:
            duration = len(audio[0]) / sample_rate
        else:
            duration = len(audio) / sample_rate
        
        if duration < self.min_duration:
            return False, f"音频时长过短 ({duration:.2f}秒)，需要至少 {self.min_duration}秒"
        
        if duration > self.max_duration:
            return False, f"音频时长过长 ({duration:.2f}秒)，最多 {self.max_duration}秒"
        
        # 检查是否为静音
        max_amplitude = np.max(np.abs(audio))
        if max_amplitude < 0.01:
            return False, "音频可能为静音或音量过小"
        
        return True, None
    
    def convert_to_mono(self, audio: np.ndarray) -> np.ndarray:
        """转换为单声道"""
        if len(audio.shape) > 1:
            audio = np.mean(audio, axis=0)
        return audio
    
    def resample(
        self,
        audio: np.ndarray,
        original_sr: int,
        target_sr: int
    ) -> np.ndarray:
        """重采样音频"""
        if original_sr == target_sr:
            return audio
        return librosa.resample(audio, orig_sr=original_sr, target_sr=target_sr)
    
    def normalize(self, audio: np.ndarray) -> np.ndarray:
        """音频归一化"""
        max_val = np.max(np.abs(audio))
        if max_val > 0:
            audio = audio / max_val * 0.95
        return audio
    
    def process_audio(
        self,
        input_path: str,
        output_path: Optional[str] = None,
        normalize: bool = True
    ) -> Tuple[np.ndarray, int]:
        """
        完整的音频处理流程
        
        Args:
            input_path: 输入音频路径
            output_path: 输出音频路径（可选）
            normalize: 是否归一化
            
        Returns:
            (处理后的音频数据, 采样率)
        """
        # 加载音频
        audio, sr = self.load_audio(input_path)
        
        # 转换为单声道
        if self.target_channels == 1:
            audio = self.convert_to_mono(audio)
        
        # 验证音频
        is_valid, error_msg = self.validate_audio(audio, sr)
        if not is_valid:
            raise ValueError(error_msg)
        
        # 重采样
        audio = self.resample(audio, sr, self.target_sample_rate)
        
        # 归一化
        if normalize:
            audio = self.normalize(audio)
        
        # 保存
        if output_path:
            os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
            sf.write(output_path, audio, self.target_sample_rate)
            logger.info(f"音频已保存到: {output_path}")
        
        return audio, self.target_sample_rate
    
    def get_audio_info(self, file_path: str) -> dict:
        """获取音频文件信息"""
        audio, sr = self.load_audio(file_path)
        duration = len(audio) / sr if len(audio.shape) == 1 else len(audio[0]) / sr
        channels = 1 if len(audio.shape) == 1 else audio.shape[0]
        
        return {
            "file_path": file_path,
            "sample_rate": sr,
            "duration": duration,
            "channels": channels,
            "samples": len(audio) if len(audio.shape) == 1 else len(audio[0]),
            "max_amplitude": float(np.max(np.abs(audio)))
        }
    
    def batch_process(
        self,
        input_dir: str,
        output_dir: str,
        extensions: List[str] = [".wav", ".mp3", ".flac", ".m4a"]
    ) -> List[str]:
        """
        批量处理音频文件
        
        Args:
            input_dir: 输入目录
            output_dir: 输出目录
            extensions: 支持的音频格式
            
        Returns:
            成功处理的文件列表
        """
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        processed_files = []
        
        for ext in extensions:
            for audio_file in input_path.glob(f"*{ext}"):
                try:
                    output_file = output_path / f"{audio_file.stem}.wav"
                    self.process_audio(str(audio_file), str(output_file))
                    processed_files.append(str(output_file))
                    logger.info(f"处理成功: {audio_file.name}")
                except Exception as e:
                    logger.error(f"处理失败 {audio_file.name}: {str(e)}")
        
        return processed_files

