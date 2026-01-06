"""
模型训练脚本

运行方式（bash）：
python scripts/train_model.py --metadata data/metadata.csv \
    --config config.json --output ./models/trained_model \
    --base_model tts_models/multilingual/multi-dataset/xtts_v2 \
    --epochs 100 --batch_size 4
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
        default="data/metadata.csv",
        help="metadata.csv文件路径（默认: data/metadata.csv）"
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
    
    args = parser.parse_args()

    # 校验 metadata 是否存在
    metadata_path = Path(args.metadata)
    if not metadata_path.exists():
        raise FileNotFoundError(f"未找到 metadata 文件: {metadata_path}")

    # 确保输出目录存在
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    trainer = ModelTrainer(
        config_path=args.config,
        output_path=args.output
    )
    
    # 标准训练模式
    if not args.config:
        raise ValueError("需要提供 --config 参数")
    
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

