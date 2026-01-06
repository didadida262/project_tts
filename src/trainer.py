"""
模型训练模块
用于训练Coqui TTS模型
"""

import os
import json
import subprocess
from pathlib import Path
from typing import Optional, Dict
import logging

logger = logging.getLogger(__name__)


class ModelTrainer:
    """模型训练器"""
    
    def __init__(
        self,
        config_path: Optional[str] = None,
        output_path: str = "./models/trained_model"
    ):
        """
        初始化训练器
        
        Args:
            config_path: 配置文件路径
            output_path: 模型输出路径
        """
        # 检查TTS是否可用
        self._check_tts_available()
        
        self.config_path = config_path
        self.output_path = Path(output_path)
        self.output_path.mkdir(parents=True, exist_ok=True)
    
    def _check_tts_available(self):
        """检查TTS是否可用"""
        import sys
        import platform
        is_windows = platform.system() == "Windows"
        
        try:
            import TTS
        except ImportError:
            # 检查Python版本
            if sys.version_info >= (3, 13):
                # Windows特定提示
                if is_windows:
                    raise RuntimeError(
                        f"当前Python版本 {sys.version_info.major}.{sys.version_info.minor} 不兼容TTS。\n"
                        "TTS需要Python 3.9-3.12。\n\n"
                        "解决方案（Windows）：\n"
                        "1. 下载并安装Python 3.11（推荐）或3.12：\n"
                        "   https://www.python.org/downloads/\n"
                        "   或使用Microsoft Store搜索'Python 3.11'\n\n"
                        "2. 安装后，使用Python启动器创建虚拟环境：\n"
                        "   py -3.11 -m venv venv  # 使用Python 3.11\n"
                        "   或\n"
                        "   py -3.12 -m venv venv  # 使用Python 3.12\n\n"
                        "3. 激活虚拟环境（Git Bash）：\n"
                        "   source venv/Scripts/activate\n\n"
                        "4. 安装TTS：\n"
                        "   pip install TTS\n\n"
                        "5. 重新运行训练脚本"
                    )
                else:
                    raise RuntimeError(
                        f"当前Python版本 {sys.version_info.major}.{sys.version_info.minor} 不兼容TTS。\n"
                        "TTS需要Python 3.9-3.12。\n\n"
                        "解决方案：\n"
                        "1. 安装Python 3.9-3.12（如未安装）\n"
                        "2. 使用Python 3.9-3.12创建新的虚拟环境：\n"
                        "   python3.11 -m venv venv  # 或使用其他3.9-3.12版本\n"
                        "   source venv/bin/activate\n"
                        "   pip install TTS"
                    )
            else:
                raise RuntimeError(
                    "TTS未安装。请运行：\n"
                    "  pip install TTS"
                )
    
    def create_config_from_model(
        self,
        model_name: str = "tts_models/multilingual/multi-dataset/xtts_v2",
        output_config: str = "config.json"
    ) -> str:
        """
        从预训练模型创建配置文件
        
        Args:
            model_name: 模型名称
            output_config: 输出配置文件路径
            
        Returns:
            配置文件路径
        """
        logger.info(f"正在从模型 {model_name} 创建配置...")
        
        try:
            # 方法1: 尝试使用 TTS API 获取配置
            try:
                from TTS.utils.manage import ModelManager
                manager = ModelManager()
                model_path, config_path, model_item = manager.download_model(model_name)
                
                # 如果模型已下载，直接复制配置文件
                if config_path and os.path.exists(config_path):
                    import shutil
                    # 确保使用绝对路径
                    output_config_abs = os.path.abspath(output_config)
                    shutil.copy2(config_path, output_config_abs)
                    # 验证文件是否真的创建成功
                    if os.path.exists(output_config_abs):
                        logger.info(f"配置文件已创建: {output_config_abs}")
                        return output_config_abs
                    else:
                        raise RuntimeError(f"配置文件复制失败: {output_config_abs}")
            except Exception as api_error:
                logger.debug(f"API方法失败: {api_error}，尝试命令行方法...")
            
            # 方法2: 尝试使用 tts 命令行工具
            import sys
            import shutil
            tts_cmd = shutil.which("tts")
            if tts_cmd:
                result = subprocess.run(
                    [tts_cmd, "--model_name", model_name, "--config_path", output_config],
                    capture_output=True,
                    text=True,
                    check=True
                )
                logger.info(f"配置文件已创建: {output_config}")
                return output_config
            
            # 方法3: 尝试使用 python -m TTS.bin.synthesize 或其他方式
            # 如果都失败，尝试直接下载模型配置
            from TTS.config import load_config
            from TTS.utils.manage import ModelManager
            
            manager = ModelManager()
            # 下载模型（如果未下载）
            model_path, config_path, model_item = manager.download_model(model_name)
            
            if config_path and os.path.exists(config_path):
                import shutil
                # 确保使用绝对路径
                output_config_abs = os.path.abspath(output_config)
                shutil.copy2(config_path, output_config_abs)
                # 验证文件是否真的创建成功
                if os.path.exists(output_config_abs):
                    logger.info(f"配置文件已创建: {output_config_abs}")
                    return output_config_abs
                else:
                    raise RuntimeError(f"配置文件复制失败: {output_config_abs}")
            else:
                raise RuntimeError(f"无法获取模型配置: {model_name}")
                
        except ImportError as e:
            raise RuntimeError(f"TTS未正确安装: {str(e)}\n请运行: pip install TTS")
        except Exception as e:
            logger.error(f"创建配置失败: {str(e)}")
            raise RuntimeError(f"无法创建配置文件: {str(e)}")
    
    def update_config(
        self,
        config_path: str,
        updates: Dict
    ):
        """
        更新配置文件
        
        Args:
            config_path: 配置文件路径
            updates: 要更新的配置项
        """
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)
        
        # 递归更新配置
        def update_dict(d, u):
            for k, v in u.items():
                if isinstance(v, dict) and k in d and isinstance(d[k], dict):
                    update_dict(d[k], v)
                else:
                    d[k] = v
        
        update_dict(config, updates)
        
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        
        logger.info(f"配置文件已更新: {config_path}")
    
    def train(
        self,
        metadata_path: str,
        config_path: Optional[str] = None,
        restore_path: Optional[str] = None,
        epochs: Optional[int] = None,
        batch_size: Optional[int] = None
    ):
        """
        开始训练模型
        
        Args:
            metadata_path: metadata.csv文件路径
            config_path: 配置文件路径
            restore_path: 预训练模型路径（用于fine-tuning）
            epochs: 训练轮数
            batch_size: 批次大小
        """
        if config_path is None:
            config_path = self.config_path
        
        if config_path is None:
            raise ValueError("必须提供配置文件路径")
        
        # 构建训练命令（使用python -m TTS，更可靠）
        import sys
        cmd = [sys.executable, "-m", "TTS", "train"]
        
        if config_path:
            cmd.extend(["--config_path", config_path])
        
        if restore_path:
            cmd.extend(["--restore_path", restore_path])
        
        if metadata_path:
            cmd.extend(["--train_dataset_path", metadata_path])
        
        cmd.extend(["--output_path", str(self.output_path)])
        
        if epochs:
            cmd.extend(["--epochs", str(epochs)])
        
        if batch_size:
            cmd.extend(["--batch_size", str(batch_size)])
        
        logger.info("开始训练模型...")
        logger.info(f"命令: {' '.join(cmd)}")
        
        try:
            # 运行训练命令
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True
            )
            
            # 实时输出日志
            for line in process.stdout:
                print(line, end="")
                logger.info(line.strip())
            
            process.wait()
            
            if process.returncode == 0:
                logger.info("训练完成！")
                logger.info(f"模型保存在: {self.output_path}")
            else:
                raise RuntimeError(f"训练失败，退出码: {process.returncode}")
                
        except FileNotFoundError:
            raise RuntimeError("TTS未找到，请确保已安装TTS: pip install TTS")
        except Exception as e:
            logger.error(f"训练过程出错: {str(e)}")
            raise
    
    def fine_tune_xtts(
        self,
        metadata_path: str,
        base_model: str = "tts_models/multilingual/multi-dataset/xtts_v2",
        epochs: int = 100,
        batch_size: int = 4
    ):
        """
        使用XTTS-v2进行微调（快速训练方案）
        
        Args:
            metadata_path: metadata.csv文件路径
            base_model: 基础模型名称
            epochs: 训练轮数
            batch_size: 批次大小
        """
        logger.info("使用XTTS-v2进行微调...")
        
        # 创建临时配置文件
        temp_config = self.output_path / "temp_config.json"
        self.create_config_from_model(base_model, str(temp_config))
        
        # 更新配置
        updates = {
            "datasets": [{
                "formatter": "coqui",
                "dataset_name": "my_dataset",
                "path": metadata_path,
                "meta_file_train": metadata_path
            }],
            "output_path": str(self.output_path),
            "trainer": {
                "epochs": epochs,
                "batch_size": batch_size
            }
        }
        
        self.update_config(str(temp_config), updates)
        
        # 开始训练
        self.train(
            metadata_path=metadata_path,
            config_path=str(temp_config),
            restore_path=base_model,
            epochs=epochs,
            batch_size=batch_size
        )

