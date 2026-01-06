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
        
        import json
        import os
        
        # 检查是否是 XTTS 模型
        is_xtts = False
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config_data = json.load(f)
                if config_data.get("model") == "xtts":
                    is_xtts = True
        except Exception as e:
            logger.warning(f"无法读取配置文件判断模型类型: {e}")
        
        if is_xtts:
            logger.info("检测到 XTTS 模型，使用 GPTTrainer 进行训练...")
            self._train_xtts(metadata_path, config_path, restore_path, epochs, batch_size)
        else:
            logger.info("使用标准训练流程...")
            self._train_standard(metadata_path, config_path, restore_path, epochs, batch_size)
    
    def _train_xtts(
        self,
        metadata_path: str,
        config_path: str,
        restore_path: Optional[str] = None,
        epochs: Optional[int] = None,
        batch_size: Optional[int] = None
    ):
        """使用 GPTTrainer 训练 XTTS 模型"""
        try:
            from trainer import Trainer, TrainerArgs
            from TTS.config.shared_configs import BaseDatasetConfig
            from TTS.tts.datasets import load_tts_samples
            from TTS.tts.layers.xtts.trainer.gpt_trainer import GPTTrainer, GPTTrainerConfig
            from TTS.config import load_config
            import json
            import os
            
            logger.info("开始训练 XTTS 模型...")
            logger.info(f"配置文件: {config_path}")
            logger.info(f"输出路径: {self.output_path}")
            
            # 加载配置
            config = load_config(config_path)
            
            # 更新输出路径
            config.output_path = str(self.output_path)
            
            # 更新训练参数
            if epochs:
                config.epochs = epochs
            if batch_size:
                config.batch_size = batch_size
            
            # 检查并下载 XTTS 所需的文件（tokenizer, checkpoint等）
            from TTS.utils.manage import ModelManager
            
            # 创建检查点目录
            checkpoints_dir = os.path.join(str(self.output_path), "XTTS_v2_files")
            os.makedirs(checkpoints_dir, exist_ok=True)
            
            # XTTS v2 文件链接
            TOKENIZER_FILE_LINK = "https://coqui.gateway.scarf.sh/hf-coqui/XTTS-v2/main/vocab.json"
            DVAE_CHECKPOINT_LINK = "https://coqui.gateway.scarf.sh/hf-coqui/XTTS-v2/main/dvae.pth"
            MEL_NORM_LINK = "https://coqui.gateway.scarf.sh/hf-coqui/XTTS-v2/main/mel_stats.pth"
            XTTS_CHECKPOINT_LINK = "https://coqui.gateway.scarf.sh/hf-coqui/XTTS-v2/main/model.pth"
            
            # 文件路径
            TOKENIZER_FILE = os.path.join(checkpoints_dir, "vocab.json")
            DVAE_CHECKPOINT = os.path.join(checkpoints_dir, "dvae.pth")
            MEL_NORM_FILE = os.path.join(checkpoints_dir, "mel_stats.pth")
            XTTS_CHECKPOINT = os.path.join(checkpoints_dir, "model.pth")
            
            # 下载必需的文件
            files_to_download = []
            if not os.path.isfile(TOKENIZER_FILE):
                files_to_download.append(TOKENIZER_FILE_LINK)
            if not os.path.isfile(DVAE_CHECKPOINT):
                files_to_download.append(DVAE_CHECKPOINT_LINK)
            if not os.path.isfile(MEL_NORM_FILE):
                files_to_download.append(MEL_NORM_LINK)
            
            if files_to_download:
                logger.info("下载 XTTS 必需文件...")
                ModelManager._download_model_files(files_to_download, checkpoints_dir, progress_bar=True)
            
            # 设置必需的 XTTS 文件路径
            # 使用 setattr 或直接赋值，取决于 config.model_args 的类型
            if not getattr(config.model_args, 'tokenizer_file', None) or config.model_args.tokenizer_file == "":
                config.model_args.tokenizer_file = TOKENIZER_FILE
                logger.info(f"设置 tokenizer_file: {TOKENIZER_FILE}")
            
            # 设置 DVAE 和 mel_norm 文件
            if not getattr(config.model_args, 'dvae_checkpoint', None) or config.model_args.dvae_checkpoint == "":
                config.model_args.dvae_checkpoint = DVAE_CHECKPOINT
                logger.info(f"设置 dvae_checkpoint: {DVAE_CHECKPOINT}")
            
            if not getattr(config.model_args, 'mel_norm_file', None) or config.model_args.mel_norm_file == "":
                config.model_args.mel_norm_file = MEL_NORM_FILE
                logger.info(f"设置 mel_norm_file: {MEL_NORM_FILE}")
            
            # 如果需要 checkpoint，也下载它
            if restore_path and ("tts_models" in str(restore_path) or "XTTS" in str(restore_path)):
                # 检查文件是否存在
                if not os.path.isfile(XTTS_CHECKPOINT):
                    logger.info("下载 XTTS checkpoint 文件（约 1.87GB，可能需要较长时间）...")
                    try:
                        ModelManager._download_model_files([XTTS_CHECKPOINT_LINK], checkpoints_dir, progress_bar=True)
                        if os.path.isfile(XTTS_CHECKPOINT):
                            file_size = os.path.getsize(XTTS_CHECKPOINT)
                            logger.info(f"✓ 下载完成！文件大小: {file_size / 1024 / 1024 / 1024:.2f} GB")
                        else:
                            raise RuntimeError("下载 XTTS checkpoint 失败：文件不存在")
                    except Exception as e:
                        raise RuntimeError(f"下载 XTTS checkpoint 失败: {e}")
                else:
                    file_size = os.path.getsize(XTTS_CHECKPOINT)
                    logger.info(f"XTTS checkpoint 文件已存在 ({file_size / 1024 / 1024 / 1024:.2f} GB)")
                
                # 更新 checkpoint 路径
                if not getattr(config.model_args, 'gpt_checkpoint', None):
                    config.model_args.gpt_checkpoint = XTTS_CHECKPOINT
                # 也设置 xtts_checkpoint
                if not getattr(config.model_args, 'xtts_checkpoint', None):
                    config.model_args.xtts_checkpoint = XTTS_CHECKPOINT
                    logger.info(f"设置 xtts_checkpoint: {XTTS_CHECKPOINT}")
            
            # 准备数据集配置
            # 使用项目根目录作为 path，因为 metadata.csv 中的路径是相对于项目根目录的
            project_root = os.path.dirname(os.path.dirname(os.path.abspath(metadata_path)))
            metadata_abs_path = os.path.abspath(metadata_path)
            config_dataset = BaseDatasetConfig(
                formatter="coqui",
                dataset_name="my_dataset",
                path=project_root,  # 使用项目根目录
                meta_file_train=metadata_abs_path,  # 使用绝对路径
                meta_file_val="",
                language=config.get("language", ""),
            )
            
            # 加载训练样本
            train_samples, eval_samples = load_tts_samples(
                [config_dataset],
                eval_split=True,
                eval_split_max_size=config.get("eval_split_max_size", 256),
                eval_split_size=config.get("eval_split_size", 0.1),
            )
            
            logger.info(f"训练样本数: {len(train_samples)}, 评估样本数: {len(eval_samples)}")
            
            # 初始化模型
            model = GPTTrainer.init_from_config(config)
            
            # 初始化训练器
            trainer = Trainer(
                TrainerArgs(
                    restore_path=restore_path,
                    skip_train_epoch=False,
                    start_with_eval=False,
                    grad_accum_steps=1,
                ),
                config,
                output_path=str(self.output_path),
                model=model,
                train_samples=train_samples,
                eval_samples=eval_samples,
            )
            
            # 开始训练
            trainer.fit()
            
            logger.info("训练完成！")
            logger.info(f"模型保存在: {self.output_path}")
            
        except ImportError as e:
            raise RuntimeError(
                f"XTTS 训练所需的模块导入失败: {e}\n"
                "请确保已正确安装 TTS 和相关依赖。"
            )
        except Exception as e:
            logger.error(f"XTTS 训练过程出错: {str(e)}")
            raise
    
    def _train_standard(
        self,
        metadata_path: str,
        config_path: str,
        restore_path: Optional[str] = None,
        epochs: Optional[int] = None,
        batch_size: Optional[int] = None
    ):
        """使用标准训练流程"""
        import sys
        import shutil
        import os
        
        logger.info("开始训练模型...")
        logger.info(f"配置文件: {config_path}")
        logger.info(f"输出路径: {self.output_path}")
        
        # 尝试多种方式找到训练脚本
        cmd = None
        
        # 方式1: 尝试查找 TTS 包中的 train_tts.py
        try:
            import TTS
            tts_dir = os.path.dirname(TTS.__file__)
            train_script = os.path.join(tts_dir, "bin", "train_tts.py")
            if os.path.exists(train_script):
                cmd = [sys.executable, train_script]
                logger.info(f"找到训练脚本: {train_script}")
        except Exception as e:
            logger.debug(f"无法查找 TTS 包: {e}")
        
        # 方式2: 如果找不到，尝试使用 train.py
        if not cmd:
            try:
                import TTS
                tts_dir = os.path.dirname(TTS.__file__)
                train_script = os.path.join(tts_dir, "train.py")
                if os.path.exists(train_script):
                    cmd = [sys.executable, train_script]
                    logger.info(f"找到训练脚本: {train_script}")
            except:
                pass
        
        # 方式3: 如果都找不到，提示用户手动运行
        if not cmd:
            raise RuntimeError(
                "无法找到 TTS 训练脚本。\n\n"
                "请尝试手动运行训练命令：\n"
                f"1. 确保已激活虚拟环境\n"
                f"2. 运行以下命令：\n"
                f"   python -m TTS.trainer.train --config_path {config_path} --output_path {self.output_path}\n\n"
                "或参考 Coqui TTS 官方文档：https://tts.readthedocs.io/en/latest/training/index.html"
            )
        
        # 添加参数
        if config_path:
            cmd.extend(["--config_path", config_path])
        if restore_path:
            cmd.extend(["--restore_path", restore_path])
        cmd.extend(["--output_path", str(self.output_path)])
        if epochs:
            cmd.extend(["--epochs", str(epochs)])
        if batch_size:
            cmd.extend(["--batch_size", str(batch_size)])
        
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
    