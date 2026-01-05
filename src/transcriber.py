"""
音频转文本模块
使用Whisper进行音频转录
"""

import os
import whisper
from pathlib import Path
from typing import Optional, List
import logging

logger = logging.getLogger(__name__)


class Transcriber:
    """音频转录器（使用Whisper）"""
    
    def __init__(self, model_size: str = "base", language: str = "zh"):
        """
        初始化转录器
        
        Args:
            model_size: Whisper模型大小 (tiny, base, small, medium, large)
            language: 语言代码 (zh=中文, en=英文等)
        """
        self.model_size = model_size
        self.language = language
        self.model = None
    
    def load_model(self):
        """加载Whisper模型"""
        if self.model is None:
            logger.info(f"正在加载Whisper模型: {self.model_size}")
            self.model = whisper.load_model(self.model_size)
            logger.info("模型加载完成")
        return self.model
    
    def transcribe_file(
        self,
        audio_path: str,
        output_path: Optional[str] = None
    ) -> str:
        """
        转录单个音频文件
        
        Args:
            audio_path: 音频文件路径
            output_path: 输出文本文件路径（可选）
            
        Returns:
            转录文本
        """
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"音频文件不存在: {audio_path}")
        
        model = self.load_model()
        
        logger.info(f"正在转录: {audio_path}")
        result = model.transcribe(audio_path, language=self.language)
        text = result["text"].strip()
        
        # 保存文本文件
        if output_path:
            os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(text)
            logger.info(f"文本已保存到: {output_path}")
        
        return text
    
    def batch_transcribe(
        self,
        audio_dir: str,
        output_dir: Optional[str] = None,
        extensions: List[str] = [".wav", ".mp3", ".flac", ".m4a"]
    ) -> List[tuple]:
        """
        批量转录音频文件
        
        Args:
            audio_dir: 音频文件目录
            output_dir: 输出文本文件目录（可选，默认与音频同目录）
            extensions: 支持的音频格式
            
        Returns:
            [(音频路径, 文本路径, 文本内容)] 列表
        """
        audio_path = Path(audio_dir)
        if not audio_path.exists():
            raise ValueError(f"音频目录不存在: {audio_dir}")
        
        if output_dir:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
        else:
            output_path = audio_path
        
        model = self.load_model()
        results = []
        
        # 遍历音频文件
        for ext in extensions:
            for audio_file in sorted(audio_path.glob(f"*{ext}")):
                try:
                    logger.info(f"正在转录: {audio_file.name}")
                    result = model.transcribe(str(audio_file), language=self.language)
                    text = result["text"].strip()
                    
                    # 保存文本文件
                    text_file = output_path / f"{audio_file.stem}.txt"
                    with open(text_file, "w", encoding="utf-8") as f:
                        f.write(text)
                    
                    results.append((str(audio_file), str(text_file), text))
                    logger.info(f"完成: {audio_file.name}")
                except Exception as e:
                    logger.error(f"转录失败 {audio_file.name}: {str(e)}")
        
        logger.info(f"批量转录完成，共处理 {len(results)} 个文件")
        return results

