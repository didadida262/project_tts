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
        self.config_path = config_path
        self.output_path = Path(output_path)
        self.output_path.mkdir(parents=True, exist_ok=True)
    
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
            # 使用TTS CLI获取配置
            result = subprocess.run(
                ["tts", "--model_name", model_name, "--config_path", output_config],
                capture_output=True,
                text=True,
                check=True
            )
            logger.info(f"配置文件已创建: {output_config}")
            return output_config
        except subprocess.CalledProcessError as e:
            logger.error(f"创建配置失败: {e.stderr}")
            raise
        except FileNotFoundError:
            raise RuntimeError("TTS命令行工具未找到，请确保已安装TTS: pip install TTS")
    
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
        
        # 构建训练命令
        cmd = ["tts", "train"]
        
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
            raise RuntimeError("TTS命令行工具未找到，请确保已安装TTS: pip install TTS")
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

