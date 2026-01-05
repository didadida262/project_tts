"""
模型加载和使用模块
用于在其他项目中使用训练好的模型
"""

import os
from pathlib import Path
from typing import Optional, List
import logging

logger = logging.getLogger(__name__)


class ModelLoader:
    """模型加载器"""
    
    def __init__(
        self,
        model_path: str,
        config_path: Optional[str] = None
    ):
        """
        初始化模型加载器
        
        Args:
            model_path: 模型文件路径（.pth文件或模型目录）
            config_path: 配置文件路径（可选）
        """
        self.model_path = Path(model_path)
        self.config_path = Path(config_path) if config_path else None
        self.tts = None
    
    def load(self):
        """加载TTS模型"""
        try:
            from TTS.api import TTS
        except ImportError:
            raise ImportError("TTS未安装，请运行: pip install TTS")
        
        logger.info(f"正在加载模型: {self.model_path}")
        
        if self.config_path and self.config_path.exists():
            self.tts = TTS(
                model_path=str(self.model_path),
                config_path=str(self.config_path)
            )
        else:
            # 尝试直接加载模型目录
            if self.model_path.is_dir():
                self.tts = TTS(model_path=str(self.model_path))
            else:
                # 假设是模型文件，尝试加载
                self.tts = TTS(model_path=str(self.model_path))
        
        logger.info("模型加载完成")
        return self.tts
    
    def synthesize(
        self,
        text: str,
        output_path: str,
        language: Optional[str] = None
    ):
        """
        合成语音
        
        Args:
            text: 要合成的文本
            output_path: 输出音频文件路径
            language: 语言代码（可选）
        """
        if self.tts is None:
            self.load()
        
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        
        logger.info(f"正在合成: {text[:50]}...")
        
        try:
            if language:
                self.tts.tts_to_file(text=text, file_path=output_path, language=language)
            else:
                self.tts.tts_to_file(text=text, file_path=output_path)
            
            logger.info(f"语音已生成: {output_path}")
        except Exception as e:
            logger.error(f"合成失败: {str(e)}")
            raise
    
    def batch_synthesize(
        self,
        texts: List[str],
        output_dir: str,
        prefix: str = "output",
        language: Optional[str] = None
    ) -> List[str]:
        """
        批量合成语音
        
        Args:
            texts: 文本列表
            output_dir: 输出目录
            prefix: 文件名前缀
            language: 语言代码（可选）
            
        Returns:
            生成的音频文件路径列表
        """
        if self.tts is None:
            self.load()
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        generated_files = []
        
        for i, text in enumerate(texts, 1):
            output_file = output_path / f"{prefix}_{i:03d}.wav"
            try:
                self.synthesize(text, str(output_file), language)
                generated_files.append(str(output_file))
            except Exception as e:
                logger.error(f"合成失败 [{i}]: {str(e)}")
        
        logger.info(f"批量合成完成，共生成 {len(generated_files)} 个文件")
        return generated_files

