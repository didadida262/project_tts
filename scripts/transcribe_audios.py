"""
音频转文本脚本
读取data/audios文件夹下的音频文件，使用Whisper转录音频生成文本，
生成对应的txt文件，命名与音频文件名一致
"""

import sys
import os
import argparse
import logging
import re
from pathlib import Path
from typing import List, Tuple, Optional

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))


def check_environment() -> Tuple[bool, str]:
    """检查运行环境"""
    errors = []
    
    # 检查Python版本
    if sys.version_info < (3, 8):
        errors.append(f"Python版本过低: {sys.version_info.major}.{sys.version_info.minor}，需要Python 3.8+")
    
    # 检查Whisper
    try:
        import whisper
    except ImportError:
        errors.append("缺少依赖库: openai-whisper")
    
    if errors:
        error_msg = "\n".join([f"  [{i+1}] {err}" for i, err in enumerate(errors)])
        return False, error_msg
    
    return True, ""


def print_environment_error(error_msg: str):
    """打印环境错误信息"""
    import platform
    is_windows = platform.system() == "Windows"
    
    print("=" * 70)
    print("环境检查失败！")
    print("=" * 70)
    print("\n发现以下问题：")
    print(error_msg)
    print("\n" + "=" * 70)
    print("解决方案：")
    print("=" * 70)
    print("\n1. 确保已创建并激活虚拟环境：")
    if is_windows:
        print("   python -m venv venv")
        print("   source venv/Scripts/activate  # Git Bash")
    else:
        print("   python3 -m venv venv")
        print("   source venv/bin/activate")
    print("\n2. 安装Whisper：")
    print("   pip install openai-whisper")
    print("\n3. 或安装所有项目依赖：")
    print("   pip install -r requirements.txt")
    print("=" * 70)
    sys.exit(1)


# 环境检查
is_ok, error_msg = check_environment()
if not is_ok:
    print_environment_error(error_msg)

# 导入模块（环境检查通过后）
import whisper

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def remove_punctuation(text: str) -> str:
    """
    去除标点符号
    
    Args:
        text: 原始文本
        
    Returns:
        去除标点后的文本
    """
    # 保留中文字符、英文字母、数字和空格
    # 去除所有标点符号
    text = re.sub(r'[^\w\s\u4e00-\u9fff]', '', text)
    # 去除多余空格
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def transcribe_audio_file(
    audio_path: Path,
    model,
    language: str = "zh",
    remove_punct: bool = False
) -> str:
    """
    转录音频文件
    
    Args:
        audio_path: 音频文件路径
        model: Whisper模型
        language: 语言代码
        remove_punct: 是否去除标点符号
        
    Returns:
        转录文本
    """
    logger.info(f"正在转录: {audio_path.name}")
    
    try:
        result = model.transcribe(str(audio_path), language=language)
        text = result["text"].strip()
        
        if remove_punct:
            text = remove_punctuation(text)
            logger.debug("已去除标点符号")
        
        logger.info(f"转录完成: {text[:50]}...")
        return text
    except Exception as e:
        logger.error(f"转录失败: {str(e)}")
        raise


def process_audio_files(
    audio_dir: str = "data/audios",
    output_dir: Optional[str] = None,
    whisper_model: str = "base",
    language: str = "zh",
    remove_punct: bool = False,
    skip_existing: bool = True
) -> dict:
    """
    处理音频文件，生成文本文件
    
    Args:
        audio_dir: 音频文件目录
        output_dir: 输出目录（默认与音频同目录）
        whisper_model: Whisper模型大小
        language: 语言代码
        remove_punct: 是否去除标点符号
        skip_existing: 是否跳过已存在的文本文件
        
    Returns:
        处理结果统计
    """
    audio_path = Path(audio_dir)
    if not audio_path.exists():
        raise FileNotFoundError(f"音频目录不存在: {audio_dir}")
    
    # 输出目录
    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
    else:
        output_path = audio_path
    
    # 支持的音频格式
    audio_extensions = [".wav", ".mp3", ".flac", ".m4a", ".m4v", ".mp4"]
    
    # 扫描音频文件
    audio_files = []
    for ext in audio_extensions:
        audio_files.extend(audio_path.glob(f"*{ext}"))
        audio_files.extend(audio_path.glob(f"*{ext.upper()}"))
    
    # 去重（Windows不区分大小写）
    audio_files = sorted(list(set(audio_files)))
    
    if not audio_files:
        logger.warning(f"在 {audio_dir} 中未找到音频文件")
        return {
            "total": 0,
            "success": 0,
            "failed": 0,
            "skipped": 0
        }
    
    logger.info(f"扫描到 {len(audio_files)} 个音频文件")
    
    # 加载Whisper模型
    logger.info(f"正在加载Whisper模型: {whisper_model}")
    logger.info("首次使用会自动下载模型，请耐心等待...")
    model = whisper.load_model(whisper_model)
    logger.info("模型加载完成")
    
    # 统计信息
    stats = {
        "total": len(audio_files),
        "success": 0,
        "failed": 0,
        "skipped": 0
    }
    
    logger.info("=" * 60)
    logger.info(f"开始处理，共 {len(audio_files)} 个文件")
    logger.info(f"语言: {language}")
    logger.info(f"去除标点: {'是' if remove_punct else '否'}")
    logger.info("=" * 60)
    
    # 处理每个文件
    for i, audio_file in enumerate(audio_files, 1):
        logger.info(f"\n[{i}/{len(audio_files)}] {audio_file.name}")
        
        # 生成输出文件路径
        text_file = output_path / f"{audio_file.stem}.txt"
        
        # 检查是否已存在
        if text_file.exists() and skip_existing:
            logger.info(f"  跳过（文本文件已存在）: {text_file.name}")
            stats["skipped"] += 1
            continue
        
        try:
            # 转录音频
            text = transcribe_audio_file(
                audio_file,
                model,
                language=language,
                remove_punct=remove_punct
            )
            
            # 保存文本文件（UTF-8编码）
            with open(text_file, "w", encoding="utf-8") as f:
                f.write(text)
            
            logger.info(f"  文本已保存: {text_file.name}")
            logger.info(f"  文本长度: {len(text)} 字符")
            stats["success"] += 1
            
        except Exception as e:
            logger.error(f"  处理失败: {str(e)}")
            stats["failed"] += 1
            continue
    
    # 打印统计信息
    logger.info("\n" + "=" * 60)
    logger.info("处理完成！")
    logger.info(f"总计: {stats['total']} 个文件")
    logger.info(f"成功: {stats['success']} 个")
    logger.info(f"失败: {stats['failed']} 个")
    logger.info(f"跳过: {stats['skipped']} 个")
    logger.info("=" * 60)
    
    return stats


def main():
    parser = argparse.ArgumentParser(
        description="音频转文本：读取音频文件，使用Whisper生成对应的txt文件"
    )
    parser.add_argument(
        "--audio_dir",
        type=str,
        default="data/audios",
        help="音频文件目录（默认: data/audios）"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="输出目录（默认与音频同目录）"
    )
    parser.add_argument(
        "--whisper_model",
        type=str,
        default="base",
        choices=["tiny", "base", "small", "medium", "large"],
        help="Whisper模型大小（默认: base）"
    )
    parser.add_argument(
        "--language",
        type=str,
        default="zh",
        help="音频语言代码（默认: zh）"
    )
    parser.add_argument(
        "--remove_punct",
        action="store_true",
        help="去除标点符号"
    )
    parser.add_argument(
        "--no_skip",
        action="store_true",
        help="不跳过已存在的文本文件（会覆盖）"
    )
    
    args = parser.parse_args()
    
    # 检查输入目录
    audio_dir_path = Path(args.audio_dir)
    if not audio_dir_path.exists():
        print("=" * 70)
        print("输入目录不存在！")
        print("=" * 70)
        print(f"\n目录: {args.audio_dir}")
        print("\n请执行以下步骤：")
        print(f"  1. 创建目录: mkdir -p {args.audio_dir}")
        print(f"  2. 将音频文件放入该目录")
        print(f"  3. 重新运行脚本")
        print("=" * 70)
        sys.exit(1)
    
    # 处理音频文件
    try:
        stats = process_audio_files(
            audio_dir=args.audio_dir,
            output_dir=args.output_dir,
            whisper_model=args.whisper_model,
            language=args.language,
            remove_punct=args.remove_punct,
            skip_existing=not args.no_skip
        )
        
        if stats["failed"] > 0:
            logger.warning(f"有 {stats['failed']} 个文件处理失败，请检查日志")
            sys.exit(1)
        else:
            logger.info("所有文件处理成功！")
            
    except Exception as e:
        logger.error(f"处理过程出错: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()

