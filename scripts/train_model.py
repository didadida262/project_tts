"""
模型训练脚本
"""

import sys
import argparse
import logging
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.trainer import ModelTrainer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="训练TTS模型")
    parser.add_argument(
        "--metadata",
        type=str,
        required=True,
        help="metadata.csv文件路径"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./models/trained_model",
        help="模型输出路径"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="配置文件路径（可选）"
    )
    parser.add_argument(
        "--base_model",
        type=str,
        default="tts_models/multilingual/multi-dataset/xtts_v2",
        help="基础模型（用于fine-tuning）"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="训练轮数"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="批次大小"
    )
    parser.add_argument(
        "--fine_tune",
        action="store_true",
        help="使用fine-tuning模式（快速训练）"
    )
    
    args = parser.parse_args()
    
    trainer = ModelTrainer(
        config_path=args.config,
        output_path=args.output
    )
    
    if args.fine_tune:
        # 使用fine-tuning模式
        logger.info("使用fine-tuning模式训练...")
        trainer.fine_tune_xtts(
            metadata_path=args.metadata,
            base_model=args.base_model,
            epochs=args.epochs,
            batch_size=args.batch_size
        )
    else:
        # 标准训练模式
        if not args.config:
            raise ValueError("标准训练模式需要提供 --config 参数")
        
        trainer.train(
            metadata_path=args.metadata,
            config_path=args.config,
            restore_path=args.base_model if args.base_model else None,
            epochs=args.epochs,
            batch_size=args.batch_size
        )
    
    logger.info("训练完成！")


if __name__ == "__main__":
    main()

