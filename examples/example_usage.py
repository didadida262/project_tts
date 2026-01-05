"""
使用示例
演示如何使用项目中的各个模块
"""

import sys
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

# ========== 示例1: 音频处理 ==========
def example_audio_processing():
    """音频处理示例"""
    from src.audio_processor import AudioProcessor
    
    processor = AudioProcessor(target_sample_rate=22050)
    
    # 处理单个音频文件
    audio, sr = processor.process_audio(
        input_path="data/audio/example.wav",
        output_path="data/processed/example.wav"
    )
    
    # 批量处理
    processed_files = processor.batch_process(
        input_dir="data/audio",
        output_dir="data/processed"
    )
    print(f"已处理 {len(processed_files)} 个文件")


# ========== 示例2: 音频转文本 ==========
def example_transcription():
    """音频转文本示例"""
    from src.transcriber import Transcriber
    
    transcriber = Transcriber(model_size="base", language="zh")
    
    # 转录单个文件
    text = transcriber.transcribe_file(
        audio_path="data/audio/example.wav",
        output_path="data/transcripts/example.txt"
    )
    print(f"转录结果: {text}")
    
    # 批量转录
    results = transcriber.batch_transcribe("data/audio", "data/transcripts")
    print(f"共转录 {len(results)} 个文件")


# ========== 示例3: 数据准备 ==========
def example_data_preparation():
    """数据准备示例"""
    from src.data_preparer import DataPreparer
    
    preparer = DataPreparer()
    
    # 生成metadata.csv
    metadata_path = preparer.create_metadata(
        audio_dir="data/audio",
        output_file="data/metadata.csv"
    )
    
    # 验证metadata
    validation = preparer.validate_metadata(metadata_path)
    print(f"验证结果: {validation}")
    
    # 获取统计信息
    stats = preparer.get_statistics(metadata_path)
    print(f"数据统计: {stats}")


# ========== 示例4: 模型训练 ==========
def example_training():
    """模型训练示例"""
    from src.trainer import ModelTrainer
    
    trainer = ModelTrainer(output_path="./models/trained_model")
    
    # Fine-tuning方式（推荐）
    trainer.fine_tune_xtts(
        metadata_path="data/metadata.csv",
        base_model="tts_models/multilingual/multi-dataset/xtts_v2",
        epochs=100,
        batch_size=4
    )


# ========== 示例5: 使用模型 ==========
def example_model_usage():
    """模型使用示例"""
    from src.model_loader import ModelLoader
    
    # 加载模型
    loader = ModelLoader(
        model_path="./models/trained_model",
        config_path="./models/trained_model/config.json"
    )
    loader.load()
    
    # 单个文本合成
    loader.synthesize(
        text="你好，这是声音合成测试。",
        output_path="output/example.wav"
    )
    
    # 批量合成
    texts = [
        "第一段文本内容。",
        "第二段文本内容。",
        "第三段文本内容。"
    ]
    
    generated_files = loader.batch_synthesize(
        texts=texts,
        output_dir="output/",
        prefix="output"
    )
    print(f"已生成 {len(generated_files)} 个音频文件")


# ========== 完整流程示例 ==========
def example_full_pipeline():
    """完整流程示例"""
    print("=== 步骤1: 音频转文本 ===")
    from src.transcriber import Transcriber
    transcriber = Transcriber(model_size="base", language="zh")
    transcriber.batch_transcribe("data/audio", "data/transcripts")
    
    print("\n=== 步骤2: 生成metadata ===")
    from src.data_preparer import DataPreparer
    preparer = DataPreparer()
    metadata_path = preparer.create_metadata(
        audio_dir="data/audio",
        output_file="data/metadata.csv"
    )
    
    print("\n=== 步骤3: 训练模型 ===")
    from src.trainer import ModelTrainer
    trainer = ModelTrainer(output_path="./models/trained_model")
    trainer.fine_tune_xtts(
        metadata_path=metadata_path,
        base_model="tts_models/multilingual/multi-dataset/xtts_v2",
        epochs=100,
        batch_size=4
    )
    
    print("\n=== 步骤4: 使用模型 ===")
    from src.model_loader import ModelLoader
    loader = ModelLoader(model_path="./models/trained_model")
    loader.load()
    loader.synthesize(
        text="训练完成，可以使用模型了！",
        output_path="output/final.wav"
    )
    
    print("\n完成！")


if __name__ == "__main__":
    # 运行示例（根据需要取消注释）
    # example_audio_processing()
    # example_transcription()
    # example_data_preparation()
    # example_training()
    # example_model_usage()
    # example_full_pipeline()
    
    print("请根据需要取消注释相应的示例函数")

