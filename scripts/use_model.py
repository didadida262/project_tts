"""
使用训练好的模型生成语音
"""

import sys
import argparse
import logging
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model_loader import ModelLoader

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="使用训练好的模型生成语音")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="模型路径（.pth文件或模型目录）"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="配置文件路径（可选）"
    )
    parser.add_argument(
        "--text",
        type=str,
        default=None,
        help="要合成的文本"
    )
    parser.add_argument(
        "--text_file",
        type=str,
        default=None,
        help="包含文本的文件（每行一段文本）"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="output",
        help="输出目录或文件路径"
    )
    parser.add_argument(
        "--language",
        type=str,
        default=None,
        help="语言代码（可选）"
    )
    
    args = parser.parse_args()
    
    # 加载模型
    loader = ModelLoader(
        model_path=args.model,
        config_path=args.config
    )
    loader.load()
    
    # 处理文本
    if args.text:
        # 单个文本
        output_path = args.output
        if not output_path.endswith(".wav"):
            output_path = Path(output_path) / "output.wav"
        
        loader.synthesize(
            text=args.text,
            output_path=str(output_path),
            language=args.language
        )
    elif args.text_file:
        # 从文件读取文本
        with open(args.text_file, "r", encoding="utf-8") as f:
            texts = [line.strip() for line in f if line.strip()]
        
        loader.batch_synthesize(
            texts=texts,
            output_dir=args.output,
            language=args.language
        )
    else:
        parser.error("必须提供 --text 或 --text_file 参数")
    
    logger.info("完成！")


if __name__ == "__main__":
    main()

