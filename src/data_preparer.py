"""
数据准备模块
用于生成metadata.csv和处理训练数据
"""

import os
import csv
import pandas as pd
from pathlib import Path
from typing import List, Optional, Dict
import logging

logger = logging.getLogger(__name__)


class DataPreparer:
    """数据准备器"""
    
    def __init__(self, data_dir: str = "data"):
        """
        初始化数据准备器
        
        Args:
            data_dir: 数据目录
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
    
    def create_metadata(
        self,
        audio_dir: str,
        output_file: str = "metadata.csv",
        delimiter: str = "|",
        text_dir: Optional[str] = None
    ) -> str:
        """
        创建metadata.csv文件
        
        Args:
            audio_dir: 音频文件目录
            output_file: 输出metadata文件路径
            delimiter: CSV分隔符
            text_dir: 文本文件目录（如果文本文件与音频文件分开存放）
            
        Returns:
            metadata文件路径
        """
        audio_path = Path(audio_dir)
        if not audio_path.exists():
            raise ValueError(f"音频目录不存在: {audio_dir}")
        
        # 支持的音频格式
        audio_extensions = [".wav", ".mp3", ".flac", ".m4a"]
        
        metadata_path = self.data_dir / output_file
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        
        rows = []
        
        # 遍历音频文件
        for ext in audio_extensions:
            for audio_file in sorted(audio_path.glob(f"*{ext}")):
                # 查找对应的文本文件
                text_file = None
                
                if text_dir:
                    # 在指定文本目录查找
                    text_path = Path(text_dir) / f"{audio_file.stem}.txt"
                    if text_path.exists():
                        text_file = text_path
                else:
                    # 在音频目录查找同名txt文件
                    text_path = audio_path / f"{audio_file.stem}.txt"
                    if text_path.exists():
                        text_file = text_path
                
                if text_file and text_file.exists():
                    # 读取文本内容
                    try:
                        with open(text_file, "r", encoding="utf-8") as f:
                            text = f.read().strip()
                        
                        if text:
                            # 使用相对路径
                            audio_rel_path = str(audio_file.relative_to(self.data_dir.parent))
                            rows.append({
                                "audio_path": audio_rel_path,
                                "text": text
                            })
                            logger.debug(f"添加: {audio_file.name} -> {text[:50]}...")
                        else:
                            logger.warning(f"文本文件为空: {text_file}")
                    except Exception as e:
                        logger.error(f"读取文本文件失败 {text_file}: {str(e)}")
                else:
                    logger.warning(f"未找到文本文件: {audio_file.stem}.txt")
        
        if not rows:
            raise ValueError("未找到任何有效的音频-文本对")
        
        # 写入CSV文件
        with open(metadata_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["audio_path", "text"], delimiter=delimiter)
            writer.writeheader()
            writer.writerows(rows)
        
        logger.info(f"metadata.csv已创建: {metadata_path}")
        logger.info(f"共 {len(rows)} 条记录")
        
        return str(metadata_path)
    
    def validate_metadata(self, metadata_file: str) -> Dict:
        """
        验证metadata文件
        
        Args:
            metadata_file: metadata文件路径
            
        Returns:
            验证结果字典
        """
        metadata_path = Path(metadata_file)
        if not metadata_path.exists():
            raise FileNotFoundError(f"metadata文件不存在: {metadata_file}")
        
        result = {
            "total": 0,
            "valid": 0,
            "invalid": 0,
            "errors": []
        }
        
        # 读取metadata
        try:
            df = pd.read_csv(metadata_path, delimiter="|", encoding="utf-8")
        except Exception as e:
            result["errors"].append(f"读取文件失败: {str(e)}")
            return result
        
        result["total"] = len(df)
        
        # 验证每一行
        for idx, row in df.iterrows():
            audio_path = row.get("audio_path", "")
            text = row.get("text", "")
            
            # 检查音频文件是否存在
            full_audio_path = self.data_dir.parent / audio_path
            if not full_audio_path.exists():
                result["invalid"] += 1
                result["errors"].append(f"第{idx+1}行: 音频文件不存在 - {audio_path}")
                continue
            
            # 检查文本是否为空
            if not text or not text.strip():
                result["invalid"] += 1
                result["errors"].append(f"第{idx+1}行: 文本为空")
                continue
            
            result["valid"] += 1
        
        return result
    
    def get_statistics(self, metadata_file: str) -> Dict:
        """
        获取数据统计信息
        
        Args:
            metadata_file: metadata文件路径
            
        Returns:
            统计信息字典
        """
        metadata_path = Path(metadata_file)
        if not metadata_path.exists():
            raise FileNotFoundError(f"metadata文件不存在: {metadata_file}")
        
        df = pd.read_csv(metadata_path, delimiter="|", encoding="utf-8")
        
        # 计算总时长（需要加载音频文件）
        total_duration = 0.0
        import librosa
        
        for audio_path in df["audio_path"]:
            try:
                full_path = self.data_dir.parent / audio_path
                if full_path.exists():
                    audio, sr = librosa.load(str(full_path), sr=None)
                    duration = len(audio) / sr if len(audio.shape) == 1 else len(audio[0]) / sr
                    total_duration += duration
            except Exception as e:
                logger.warning(f"无法获取时长 {audio_path}: {str(e)}")
        
        # 计算文本统计
        text_lengths = df["text"].str.len()
        
        return {
            "total_files": len(df),
            "total_duration_minutes": total_duration / 60.0,
            "total_duration_seconds": total_duration,
            "avg_text_length": text_lengths.mean(),
            "min_text_length": text_lengths.min(),
            "max_text_length": text_lengths.max(),
            "total_text_chars": text_lengths.sum()
        }

