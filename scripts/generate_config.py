"""
生成配置文件脚本

运行方式：
python scripts/generate_config.py --base_model tts_models/multilingual/multi-dataset/xtts_v2 --output config.json
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
    parser = argparse.ArgumentParser(description="从基础模型生成配置文件")
    parser.add_argument(
        "--base_model",
        type=str,
        default="tts_models/multilingual/multi-dataset/xtts_v2",
        help="基础模型名称（默认: tts_models/multilingual/multi-dataset/xtts_v2）"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="config.json",
        help="输出配置文件路径（默认: config.json）"
    )
    
    args = parser.parse_args()
    
    trainer = ModelTrainer()
    
    try:
        config_path = trainer.create_config_from_model(
            model_name=args.base_model,
            output_config=args.output
        )
        logger.info(f"\n✓ 配置文件已生成: {config_path}")
        logger.info(f"\n接下来可以：")
        logger.info(f"1. 编辑 {config_path}，设置数据集路径和训练参数")
        logger.info(f"2. 使用标准训练模式：")
        logger.info(f"   python scripts/train_model.py --metadata data/metadata.csv \\")
        logger.info(f"       --config {config_path} --output ./models/trained_model \\")
        logger.info(f"       --base_model {args.base_model} --epochs 100 --batch_size 4")
    except Exception as e:
        logger.error(f"生成配置文件失败: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()

