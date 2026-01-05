# Coqui TTS 声音模型训练项目

基于Coqui TTS的声音模型训练和使用系统。

## 项目结构

```
project_tts/
├── src/                    # 源代码模块
│   ├── audio_processor.py  # 音频处理
│   ├── data_preparer.py    # 数据准备
│   ├── transcriber.py      # 音频转文本（Whisper）
│   ├── trainer.py          # 模型训练
│   └── model_loader.py     # 模型加载和使用
├── scripts/                # 脚本文件
│   ├── prepare_data.py     # 数据准备脚本
│   ├── train_model.py      # 训练脚本
│   └── use_model.py        # 模型使用脚本
├── data/                   # 数据目录（不提交到git）
│   ├── audio/              # 原始音频文件
│   ├── processed/          # 处理后的音频
│   └── metadata.csv        # 训练数据元数据
├── models/                 # 训练好的模型（不提交到git）
├── requirements.txt        # 依赖列表
└── PRD.md                  # 产品需求文档
```

## 快速开始

### 1. 安装依赖

```bash
# 创建虚拟环境
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac

# 安装PyTorch（CPU版本）
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu

# 安装项目依赖
pip install -r requirements.txt
```

### 2. 准备训练数据

#### 方式1：手动准备
- 将音频文件放入 `data/audio/` 目录
- 为每个音频文件创建对应的 `.txt` 文本文件（同名）
- 运行数据准备脚本：

```bash
python scripts/prepare_data.py --audio_dir data/audio --metadata_file data/metadata.csv
```

#### 方式2：自动转录（使用Whisper）
```bash
python scripts/prepare_data.py \
    --audio_dir data/audio \
    --metadata_file data/metadata.csv \
    --transcribe \
    --whisper_model base \
    --language zh
```

### 3. 训练模型

#### 快速训练（Fine-tuning XTTS-v2）
```bash
python scripts/train_model.py \
    --metadata data/metadata.csv \
    --output ./models/trained_model \
    --base_model tts_models/multilingual/multi-dataset/xtts_v2 \
    --epochs 100 \
    --batch_size 4 \
    --fine_tune
```

#### 标准训练
```bash
# 首先生成配置文件
tts --model_name tts_models/multilingual/multi-dataset/xtts_v2 --config_path config.json

# 修改config.json后训练
python scripts/train_model.py \
    --metadata data/metadata.csv \
    --config config.json \
    --output ./models/trained_model \
    --epochs 100 \
    --batch_size 4
```

### 4. 使用训练好的模型

#### 单个文本
```bash
python scripts/use_model.py \
    --model ./models/trained_model \
    --text "要合成的文本内容" \
    --output output.wav
```

#### 批量生成
```bash
# 创建文本文件 texts.txt（每行一段文本）
python scripts/use_model.py \
    --model ./models/trained_model \
    --text_file texts.txt \
    --output output/
```

## Python API使用

### 数据准备

```python
from src.data_preparer import DataPreparer
from src.transcriber import Transcriber

# 转录音频
transcriber = Transcriber(model_size="base", language="zh")
transcriber.batch_transcribe("data/audio", "data/transcripts")

# 生成metadata
preparer = DataPreparer()
metadata_path = preparer.create_metadata(
    audio_dir="data/audio",
    output_file="data/metadata.csv"
)
```

### 训练模型

```python
from src.trainer import ModelTrainer

trainer = ModelTrainer(output_path="./models/trained_model")
trainer.fine_tune_xtts(
    metadata_path="data/metadata.csv",
    base_model="tts_models/multilingual/multi-dataset/xtts_v2",
    epochs=100,
    batch_size=4
)
```

### 使用模型

```python
from src.model_loader import ModelLoader

# 加载模型
loader = ModelLoader(
    model_path="./models/trained_model",
    config_path="./models/trained_model/config.json"
)
loader.load()

# 生成语音
loader.synthesize(
    text="要合成的文本",
    output_path="output.wav"
)

# 批量生成
texts = ["文本1", "文本2", "文本3"]
loader.batch_synthesize(texts, "output/")
```

## 详细文档

- [PRD.md](PRD.md) - 完整的产品需求文档和使用步骤
- [音频转文本工具推荐.md](音频转文本工具推荐.md) - 音频转文本工具推荐

## 注意事项

1. **数据要求**：
   - 至少5分钟总时长
   - 建议50-200个音频文件
   - 音频清晰无噪音

2. **训练时间**：
   - GPU训练：几小时到几天
   - CPU训练：可能需要数天

3. **模型文件**：
   - 训练好的模型保存在 `models/` 目录
   - 可以复制到其他项目使用

## 参考资源

- [Coqui TTS官方文档](https://tts.readthedocs.io/)
- [训练指南](https://tts.readthedocs.io/en/latest/training/index.html)
- [XTTS Fine-tuning](https://github.com/coqui-ai/TTS/wiki/XTTS)
