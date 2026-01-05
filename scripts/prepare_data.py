"""
数据准备脚本
用于处理音频文件和生成metadata.csv
"""

import sys
import os
import argparse
import logging
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.audio_processor import AudioProcessor
from src.data_preparer import DataPreparer
from src.transcriber import Transcriber

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="准备训练数据")
    parser.add_argument(
        "--audio_dir",
        type=str,
        default="data/audio",
        help="音频文件目录"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/processed",
        help="处理后的音频输出目录"
    )
    parser.add_argument(
        "--metadata_file",
        type=str,
        default="data/metadata.csv",
        help="metadata.csv输出路径"
    )
    parser.add_argument(
        "--transcribe",
        action="store_true",
        help="使用Whisper自动转录音频（如果文本文件不存在）"
    )
    parser.add_argument(
        "--whisper_model",
        type=str,
        default="base",
        help="Whisper模型大小 (tiny, base, small, medium, large)"
    )
    parser.add_argument(
        "--language",
        type=str,
        default="zh",
        help="音频语言代码"
    )
    parser.add_argument(
        "--process_audio",
        action="store_true",
        help="处理音频文件（格式转换、重采样等）"
    )
    
    args = parser.parse_args()
    
    # 1. 处理音频文件（可选）
    if args.process_audio:
        logger.info("开始处理音频文件...")
        processor = AudioProcessor(target_sample_rate=22050)
        processed_files = processor.batch_process(
            args.audio_dir,
            args.output_dir
        )
        logger.info(f"已处理 {len(processed_files)} 个音频文件")
        audio_dir = args.output_dir
    else:
        audio_dir = args.audio_dir
    
    # 2. 转录音频（如果需要）
    if args.transcribe:
        logger.info("开始转录音频文件...")
        transcriber = Transcriber(
            model_size=args.whisper_model,
            language=args.language
        )
        transcriber.batch_transcribe(audio_dir)
        logger.info("转录完成")
    
    # 3. 生成metadata.csv
    logger.info("生成metadata.csv...")
    preparer = DataPreparer()
    metadata_path = preparer.create_metadata(
        audio_dir=audio_dir,
        output_file=args.metadata_file
    )
    
    # 4. 验证metadata
    logger.info("验证metadata...")
    validation = preparer.validate_metadata(metadata_path)
    logger.info(f"验证结果: 总计={validation['total']}, "
                f"有效={validation['valid']}, "
                f"无效={validation['invalid']}")
    
    if validation['errors']:
        logger.warning("发现以下错误:")
        for error in validation['errors'][:10]:  # 只显示前10个错误
            logger.warning(f"  - {error}")
    
    # 5. 显示统计信息
    logger.info("数据统计:")
    stats = preparer.get_statistics(metadata_path)
    logger.info(f"  总文件数: {stats['total_files']}")
    logger.info(f"  总时长: {stats['total_duration_minutes']:.2f} 分钟")
    logger.info(f"  平均文本长度: {stats['avg_text_length']:.1f} 字符")
    
    logger.info(f"\n数据准备完成！metadata文件: {metadata_path}")


if __name__ == "__main__":
    main()

