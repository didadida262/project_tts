"""
批量将 m4a 音频文件转换为 wav 格式
"""

import sys
import os
import argparse
import logging
from pathlib import Path
from typing import List

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.audio_processor import AudioProcessor

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class AudioConverter:
    """音频格式转换器"""
    
    def __init__(
        self,
        input_dir: str = "data/origdata",
        output_dir: str = "data/audios",
        target_sample_rate: int = 22050,
        target_channels: int = 1
    ):
        """
        初始化转换器
        
        Args:
            input_dir: 输入目录（m4a文件所在目录）
            output_dir: 输出目录（wav文件保存目录）
            target_sample_rate: 目标采样率
            target_channels: 目标声道数（1=单声道）
        """
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.target_sample_rate = target_sample_rate
        self.target_channels = target_channels
        
        # 创建输出目录
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 初始化音频处理器
        self.processor = AudioProcessor(
            target_sample_rate=target_sample_rate,
            target_channels=target_channels
        )
    
    def scan_m4a_files(self) -> List[Path]:
        """
        扫描输入目录下的所有 m4a 文件
        
        Returns:
            m4a文件路径列表
        """
        if not self.input_dir.exists():
            raise FileNotFoundError(f"输入目录不存在: {self.input_dir}")
        
        m4a_files = list(self.input_dir.glob("*.m4a"))
        m4a_files.extend(self.input_dir.glob("*.M4A"))
        m4a_files = sorted(m4a_files)
        
        logger.info(f"扫描到 {len(m4a_files)} 个 m4a 文件")
        return m4a_files
    
    def convert_file(
        self,
        input_file: Path,
        overwrite: bool = False
    ) -> bool:
        """
        转换单个文件
        
        Args:
            input_file: 输入文件路径
            overwrite: 如果输出文件已存在，是否覆盖
            
        Returns:
            是否转换成功
        """
        # 生成输出文件路径
        output_file = self.output_dir / f"{input_file.stem}.wav"
        
        # 检查输出文件是否已存在
        if output_file.exists() and not overwrite:
            logger.info(f"跳过（已存在）: {input_file.name} -> {output_file.name}")
            return True
        
        try:
            logger.info(f"正在转换: {input_file.name}")
            
            # 处理音频（转换格式、重采样、归一化）
            self.processor.process_audio(
                input_path=str(input_file),
                output_path=str(output_file),
                normalize=True
            )
            
            # 获取文件大小信息
            input_size = input_file.stat().st_size / (1024 * 1024)  # MB
            output_size = output_file.stat().st_size / (1024 * 1024)  # MB
            
            logger.info(f"完成: {input_file.name} -> {output_file.name}")
            logger.info(f"  大小: {input_size:.2f} MB -> {output_size:.2f} MB")
            
            return True
            
        except Exception as e:
            logger.error(f"转换失败 {input_file.name}: {str(e)}")
            return False
    
    def batch_convert(
        self,
        overwrite: bool = False,
        skip_existing: bool = True
    ) -> dict:
        """
        批量转换文件
        
        Args:
            overwrite: 是否覆盖已存在的文件
            skip_existing: 是否跳过已存在的文件（如果overwrite=False）
            
        Returns:
            转换结果统计
        """
        # 扫描文件
        m4a_files = self.scan_m4a_files()
        
        if not m4a_files:
            logger.warning(f"在 {self.input_dir} 中未找到 m4a 文件")
            return {
                "total": 0,
                "success": 0,
                "failed": 0,
                "skipped": 0
            }
        
        # 统计信息
        stats = {
            "total": len(m4a_files),
            "success": 0,
            "failed": 0,
            "skipped": 0
        }
        
        logger.info(f"开始批量转换，共 {len(m4a_files)} 个文件...")
        logger.info(f"输出目录: {self.output_dir}")
        logger.info(f"目标采样率: {self.target_sample_rate} Hz")
        logger.info(f"目标声道: {self.target_channels} ({'单声道' if self.target_channels == 1 else '立体声'})")
        logger.info("=" * 60)
        
        # 转换每个文件
        for i, m4a_file in enumerate(m4a_files, 1):
            logger.info(f"\n[{i}/{len(m4a_files)}] {m4a_file.name}")
            
            # 检查输出文件是否已存在
            output_file = self.output_dir / f"{m4a_file.stem}.wav"
            if output_file.exists() and not overwrite:
                if skip_existing:
                    logger.info(f"  跳过（已存在）: {output_file.name}")
                    stats["skipped"] += 1
                    continue
            
            # 转换文件
            if self.convert_file(m4a_file, overwrite=overwrite):
                stats["success"] += 1
            else:
                stats["failed"] += 1
        
        # 打印统计信息
        logger.info("\n" + "=" * 60)
        logger.info("批量转换完成！")
        logger.info(f"总计: {stats['total']} 个文件")
        logger.info(f"成功: {stats['success']} 个")
        logger.info(f"失败: {stats['failed']} 个")
        logger.info(f"跳过: {stats['skipped']} 个")
        logger.info("=" * 60)
        
        return stats


def main():
    parser = argparse.ArgumentParser(
        description="批量将 m4a 音频文件转换为 wav 格式"
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        default="data/origdata",
        help="输入目录（m4a文件所在目录，默认: data/origdata）"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/audios",
        help="输出目录（wav文件保存目录，默认: data/audios）"
    )
    parser.add_argument(
        "--sample_rate",
        type=int,
        default=22050,
        help="目标采样率（默认: 22050 Hz）"
    )
    parser.add_argument(
        "--channels",
        type=int,
        default=1,
        choices=[1, 2],
        help="目标声道数（1=单声道, 2=立体声，默认: 1）"
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="覆盖已存在的输出文件"
    )
    parser.add_argument(
        "--no_skip",
        action="store_true",
        help="不跳过已存在的文件（会尝试转换，但可能因文件已存在而失败）"
    )
    
    args = parser.parse_args()
    
    # 创建转换器
    converter = AudioConverter(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        target_sample_rate=args.sample_rate,
        target_channels=args.channels
    )
    
    # 执行批量转换
    try:
        stats = converter.batch_convert(
            overwrite=args.overwrite,
            skip_existing=not args.no_skip
        )
        
        if stats["failed"] > 0:
            logger.warning(f"有 {stats['failed']} 个文件转换失败，请检查日志")
            sys.exit(1)
        else:
            logger.info("所有文件转换成功！")
            
    except Exception as e:
        logger.error(f"转换过程出错: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()

