"""
批量将 m4a 音频文件转换为 wav 格式
脚本执行：
source venv/Scripts/activate
python scripts/convert_m4a_to_wav.py

"""

import sys
import os
import argparse
import logging
from pathlib import Path
from typing import List, Tuple

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))


def check_ffmpeg() -> Tuple[bool, str]:
    """
    检查ffmpeg是否可用
    
    Returns:
        (是否可用, 错误信息)
    """
    import subprocess
    import shutil
    
    # 方法1: 使用which/where命令查找
    try:
        # Windows
        result = subprocess.run(
            ["where", "ffmpeg"],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0 and result.stdout.strip():
            return True, ""
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    
    # 方法2: 使用shutil.which (跨平台)
    if shutil.which("ffmpeg"):
        return True, ""
    
    # 方法3: 尝试直接运行ffmpeg
    try:
        result = subprocess.run(
            ["ffmpeg", "-version"],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            return True, ""
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    
    return False, "ffmpeg未在PATH中找到，可能需要重启终端/Cursor以刷新环境变量"


def check_environment() -> Tuple[bool, str]:
    """
    检查运行环境
    
    Returns:
        (是否通过, 错误信息)
    """
    errors = []
    warnings = []
    
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
    
    # 4. 检查ffmpeg（用于m4a文件处理）
    ffmpeg_ok, ffmpeg_msg = check_ffmpeg()
    if not ffmpeg_ok:
        warnings.append(f"ffmpeg检测: {ffmpeg_msg}")
    
    if errors:
        error_msg = "\n".join([f"  [{i+1}] {err}" for i, err in enumerate(errors)])
        if warnings:
            error_msg += "\n\n警告：\n" + "\n".join([f"  - {w}" for w in warnings])
        return False, error_msg
    
    # 如果有警告但没有错误，只显示警告（logger可能还未初始化，使用print）
    if warnings:
        print("\n" + "=" * 70)
        print("环境检查警告：")
        for w in warnings:
            print(f"  - {w}")
        print("\n提示：如果m4a文件转换失败，请：")
        print("  1. 重启终端/Cursor以刷新环境变量")
        print("  2. 或确保ffmpeg已添加到系统PATH环境变量")
        print("=" * 70 + "\n")
    
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
        
        # 初始化音频处理器（格式转换不需要时长限制）
        self.processor = AudioProcessor(
            target_sample_rate=target_sample_rate,
            target_channels=target_channels,
            min_duration=0.1,  # 最小时长0.1秒
            max_duration=float('inf')  # 无最大时长限制
        )
    
    def scan_m4a_files(self) -> List[Path]:
        """
        扫描输入目录下的所有 m4a 文件
        
        Returns:
            m4a文件路径列表
        """
        if not self.input_dir.exists():
            raise FileNotFoundError(f"输入目录不存在: {self.input_dir}")
        
        # 扫描m4a文件（使用set去重，因为Windows不区分大小写）
        m4a_files = set()
        m4a_files.update(self.input_dir.glob("*.m4a"))
        m4a_files.update(self.input_dir.glob("*.M4A"))
        m4a_files = sorted(list(m4a_files))
        
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
    
    # 检查输入目录
    input_path = Path(args.input_dir)
    if not input_path.exists():
        logger.error("=" * 70)
        logger.error("输入目录不存在！")
        logger.error("=" * 70)
        logger.error(f"\n目录: {args.input_dir}")
        logger.error("\n请执行以下步骤：")
        logger.error(f"  1. 创建目录: mkdir -p {args.input_dir}")
        logger.error(f"  2. 将 m4a 音频文件放入该目录")
        logger.error(f"  3. 重新运行脚本")
        logger.error("=" * 70)
        sys.exit(1)
    
    # 创建转换器
    try:
        converter = AudioConverter(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            target_sample_rate=args.sample_rate,
            target_channels=args.channels
        )
    except Exception as e:
        logger.error(f"初始化转换器失败: {str(e)}")
        sys.exit(1)
    
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
            
    except FileNotFoundError as e:
        logger.error(str(e))
        sys.exit(1)
    except Exception as e:
        logger.error(f"转换过程出错: {str(e)}")
        logger.error("\n如果问题持续，请检查：")
        logger.error("  1. 音频文件是否损坏")
        logger.error("  2. 磁盘空间是否充足")
        logger.error("  3. 文件权限是否正确")
        sys.exit(1)


if __name__ == "__main__":
    main()

