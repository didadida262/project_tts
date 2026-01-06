"""
自动化脚本：扫描音频文件，使用Whisper转录音频，生成metadata.csv
"""

import sys
import os
import csv
import argparse
import logging
from pathlib import Path
from typing import List, Dict, Tuple

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))


def check_environment() -> Tuple[bool, str]:
    """
    检查运行环境
    
    Returns:
        (是否通过, 错误信息)
    """
    errors = []
    
    # 1. 检查Python版本
    if sys.version_info < (3, 8):
        errors.append(f"Python版本过低: {sys.version_info.major}.{sys.version_info.minor}，需要Python 3.8+")
    
    # 2. 检查必要的依赖库
    missing_modules = []
    try:
        import librosa
    except ImportError:
        missing_modules.append("librosa")
    
    try:
        import soundfile
    except ImportError:
        missing_modules.append("soundfile")
    
    try:
        import numpy
    except ImportError:
        missing_modules.append("numpy")
    
    if missing_modules:
        errors.append(f"缺少依赖库: {', '.join(missing_modules)}")
    
    # 3. 检查是否能导入项目模块
    try:
        from src.audio_processor import AudioProcessor
    except ImportError as e:
        errors.append(f"无法导入项目模块: {str(e)}")
    
    if errors:
        error_msg = "\n".join([f"  [{i+1}] {err}" for i, err in enumerate(errors)])
        return False, error_msg
    
    return True, ""


def print_environment_error(error_msg: str):
    """打印环境错误信息（默认bash环境）"""
    import platform
    
    # 检测操作系统
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
        print("   或: venv\\Scripts\\activate     # CMD/PowerShell")
    else:
        print("   python3 -m venv venv")
        print("   source venv/bin/activate")
    print("\n2. 安装项目依赖：")
    print("   pip install -r requirements.txt")
    print("\n3. 或单独安装音频处理库：")
    print("   pip install librosa soundfile numpy")
    print("\n4. 如果使用国内网络，可使用镜像源：")
    print("   pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple")
    print("\n提示：如果虚拟环境已创建，请确保已激活（命令提示符前应显示 (venv)）")
    print("=" * 70)
    sys.exit(1)


# 环境检查
is_ok, error_msg = check_environment()
if not is_ok:
    print_environment_error(error_msg)

# 导入模块（环境检查通过后）
from src.audio_processor import AudioProcessor

# 检查Whisper（可选，用于转录功能）
try:
    import whisper
    WHISPER_AVAILABLE = True
except ImportError:
    WHISPER_AVAILABLE = False

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class AutoMetadataGenerator:
    """自动生成metadata.csv的类"""
    
    def __init__(
        self,
        audio_dir: str = "data/audios",
        output_file: str = "data/metadata.csv",
        whisper_model: str = "base",
        language: str = "zh",
        delimiter: str = "|"
    ):
        """
        初始化
        
        Args:
            audio_dir: 音频文件目录
            output_file: 输出的metadata.csv路径
            whisper_model: Whisper模型大小
            language: 音频语言代码
            delimiter: CSV分隔符
        """
        self.audio_dir = Path(audio_dir)
        self.output_file = Path(output_file)
        self.whisper_model = whisper_model
        self.language = language
        self.delimiter = delimiter
        
        # 支持的音频格式
        self.audio_extensions = [".wav", ".mp3", ".flac", ".m4a", ".m4v", ".mp4"]
        
        # 初始化Whisper模型
        self.whisper_model_obj = None
        if WHISPER_AVAILABLE:
            self._load_whisper_model()
        
        # 音频处理器
        self.audio_processor = AudioProcessor()
    
    def _load_whisper_model(self):
        """加载Whisper模型"""
        if not WHISPER_AVAILABLE:
            raise ImportError("openai-whisper 未安装，请运行: pip install openai-whisper")
        
        logger.info(f"正在加载Whisper模型: {self.whisper_model}")
        logger.info("首次使用会自动下载模型，请耐心等待...")
        self.whisper_model_obj = whisper.load_model(self.whisper_model)
        logger.info("Whisper模型加载完成")
    
    def scan_audio_files(self) -> List[Path]:
        """
        扫描音频文件
        
        Returns:
            音频文件路径列表
        """
        if not self.audio_dir.exists():
            raise FileNotFoundError(f"音频目录不存在: {self.audio_dir}")
        
        audio_files = []
        for ext in self.audio_extensions:
            audio_files.extend(self.audio_dir.glob(f"*{ext}"))
            audio_files.extend(self.audio_dir.glob(f"*{ext.upper()}"))
        
        audio_files = sorted(audio_files)
        logger.info(f"扫描到 {len(audio_files)} 个音频文件")
        return audio_files
    
    def get_audio_info(self, audio_path: Path) -> Dict:
        """
        获取音频文件信息
        
        Args:
            audio_path: 音频文件路径
            
        Returns:
            音频信息字典
        """
        try:
            info = self.audio_processor.get_audio_info(str(audio_path))
            return info
        except Exception as e:
            logger.warning(f"获取音频信息失败 {audio_path.name}: {str(e)}")
            return {
                "file_path": str(audio_path),
                "sample_rate": 0,
                "duration": 0,
                "channels": 0,
                "samples": 0,
                "max_amplitude": 0
            }
    
    def transcribe_audio(self, audio_path: Path) -> str:
        """
        使用Whisper转录音频
        
        Args:
            audio_path: 音频文件路径
            
        Returns:
            转录文本
        """
        if not WHISPER_AVAILABLE:
            raise ImportError("openai-whisper 未安装")
        
        if self.whisper_model_obj is None:
            self._load_whisper_model()
        
        try:
            logger.info(f"正在转录: {audio_path.name}")
            result = self.whisper_model_obj.transcribe(
                str(audio_path),
                language=self.language
            )
            text = result["text"].strip()
            logger.info(f"转录完成: {text[:50]}...")
            return text
        except Exception as e:
            logger.error(f"转录失败 {audio_path.name}: {str(e)}")
            return ""
    
    def process_audio_file(
        self,
        audio_path: Path,
        transcribe: bool = True
    ) -> Tuple[str, str, Dict]:
        """
        处理单个音频文件
        
        Args:
            audio_path: 音频文件路径
            transcribe: 是否转录音频
            
        Returns:
            (相对路径, 文本, 音频信息)
        """
        # 获取音频信息
        audio_info = self.get_audio_info(audio_path)
        
        # 转录音频
        if transcribe:
            text = self.transcribe_audio(audio_path)
        else:
            # 尝试读取已有的文本文件
            text_file = audio_path.with_suffix(".txt")
            if text_file.exists():
                with open(text_file, "r", encoding="utf-8") as f:
                    text = f.read().strip()
            else:
                text = ""
        
        # 计算相对路径（相对于项目根目录）
        try:
            # 尝试相对于项目根目录
            project_root = Path(__file__).parent.parent
            relative_path = audio_path.relative_to(project_root)
        except ValueError:
            # 如果不在项目根目录下，使用绝对路径
            relative_path = audio_path
        
        return str(relative_path), text, audio_info
    
    def generate_metadata(
        self,
        transcribe: bool = True,
        skip_existing: bool = False
    ) -> str:
        """
        生成metadata.csv文件
        
        Args:
            transcribe: 是否使用Whisper转录音频
            skip_existing: 如果文本文件已存在，是否跳过转录
            
        Returns:
            metadata.csv文件路径
        """
        # 扫描音频文件
        audio_files = self.scan_audio_files()
        
        if not audio_files:
            raise ValueError(f"在 {self.audio_dir} 中未找到音频文件")
        
        # 确保输出目录存在
        self.output_file.parent.mkdir(parents=True, exist_ok=True)
        
        # 处理每个音频文件
        metadata_rows = []
        total_duration = 0.0
        
        logger.info(f"开始处理 {len(audio_files)} 个音频文件...")
        
        for i, audio_file in enumerate(audio_files, 1):
            logger.info(f"\n[{i}/{len(audio_files)}] 处理: {audio_file.name}")
            
            try:
                # 检查是否已有文本文件
                text_file = audio_file.with_suffix(".txt")
                should_transcribe = transcribe
                
                if skip_existing and text_file.exists():
                    logger.info(f"  文本文件已存在，跳过转录")
                    should_transcribe = False
                
                # 处理音频文件
                relative_path, text, audio_info = self.process_audio_file(
                    audio_file,
                    transcribe=should_transcribe
                )
                
                if not text:
                    logger.warning(f"  警告: 未获取到文本内容，跳过此文件")
                    continue
                
                # 保存文本到文件（如果使用转录）
                if should_transcribe:
                    text_file.parent.mkdir(parents=True, exist_ok=True)
                    with open(text_file, "w", encoding="utf-8") as f:
                        f.write(text)
                    logger.info(f"  文本已保存: {text_file}")
                
                # 添加到metadata
                metadata_rows.append({
                    "audio_path": relative_path,
                    "text": text,
                    "duration": audio_info.get("duration", 0),
                    "sample_rate": audio_info.get("sample_rate", 0)
                })
                
                total_duration += audio_info.get("duration", 0)
                logger.info(f"  完成: 时长={audio_info.get('duration', 0):.2f}秒, "
                           f"文本长度={len(text)}字符")
                
            except Exception as e:
                logger.error(f"  处理失败: {str(e)}")
                continue
        
        if not metadata_rows:
            raise ValueError("未生成任何有效的metadata记录")
        
        # 写入metadata.csv
        logger.info(f"\n正在写入metadata.csv...")
        with open(self.output_file, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["audio_path", "text"],
                delimiter=self.delimiter
            )
            writer.writeheader()
            for row in metadata_rows:
                writer.writerow({
                    "audio_path": row["audio_path"],
                    "text": row["text"]
                })
        
        # 打印统计信息
        logger.info("\n" + "="*50)
        logger.info("处理完成！")
        logger.info(f"metadata.csv: {self.output_file}")
        logger.info(f"总文件数: {len(metadata_rows)}")
        logger.info(f"总时长: {total_duration/60:.2f} 分钟 ({total_duration:.2f} 秒)")
        logger.info(f"平均文本长度: {sum(len(r['text']) for r in metadata_rows) / len(metadata_rows):.1f} 字符")
        logger.info("="*50)
        
        return str(self.output_file)


def main():
    parser = argparse.ArgumentParser(
        description="自动扫描音频文件，使用Whisper转录音频，生成metadata.csv"
    )
    parser.add_argument(
        "--audio_dir",
        type=str,
        default="data/audios",
        help="音频文件目录（默认: data/audios）"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/metadata.csv",
        help="输出的metadata.csv路径（默认: data/metadata.csv）"
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
        "--no_transcribe",
        action="store_true",
        help="不转录音频，只使用已有的文本文件"
    )
    parser.add_argument(
        "--skip_existing",
        action="store_true",
        help="如果文本文件已存在，跳过转录"
    )
    
    args = parser.parse_args()
    
    # 检查Whisper是否可用（如果需要转录）
    if not args.no_transcribe and not WHISPER_AVAILABLE:
        print("=" * 70)
        print("错误: openai-whisper 未安装")
        print("=" * 70)
        print("\n此脚本需要 Whisper 来进行音频转录。")
        print("\n请安装:")
        print("  pip install openai-whisper")
        print("\n或安装所有项目依赖:")
        print("  pip install -r requirements.txt")
        print("=" * 70)
        sys.exit(1)
    
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
    
    # 创建生成器
    generator = AutoMetadataGenerator(
        audio_dir=args.audio_dir,
        output_file=args.output,
        whisper_model=args.whisper_model,
        language=args.language
    )
    
    # 生成metadata
    try:
        metadata_path = generator.generate_metadata(
            transcribe=not args.no_transcribe,
            skip_existing=args.skip_existing
        )
        logger.info(f"\n成功！metadata.csv已生成: {metadata_path}")
    except FileNotFoundError as e:
        logger.error(str(e))
        sys.exit(1)
    except Exception as e:
        logger.error(f"生成metadata失败: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()

