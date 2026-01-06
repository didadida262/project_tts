# Coqui TTS 声音模型训练与使用步骤

## 一、环境准备

### 1. 创建虚拟环境

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

### 2. 安装依赖

```bash
# 安装PyTorch (CPU版本)
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu

# 或GPU版本 (CUDA 11.8)
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu118

# 安装Coqui TTS
pip install TTS

# 安装其他依赖
pip install librosa soundfile numpy
```

---

## 二、准备训练数据

### 1. 音频要求

- **格式**: WAV格式（推荐）
- **采样率**: 22050Hz或更高
- **时长**: 总时长至少5分钟，单文件建议10-30秒
- **质量**: 清晰无噪音，单一说话人，无背景音乐
- **数量**: 建议50-200个音频文件

### 2. 准备文本转录

为每个音频文件创建对应的文本文件，确保：
- 文本与音频内容完全一致
- 使用UTF-8编码
- 去除标点符号（可选，根据模型要求）

### 3. 整理数据格式

创建CSV文件 `metadata.csv`，格式如下：

```csv
audio_path|text
data/audio_001.wav|这是第一段音频的文本内容
data/audio_002.wav|这是第二段音频的文本内容
data/audio_003.wav|这是第三段音频的文本内容
```

**目录结构：**
```
project_tts/
├── data/
│   ├── audio_001.wav
│   ├── audio_002.wav
│   └── ...
└── metadata.csv
```

---

## 三、训练模型

### 方式一：Fine-tuning 模式（推荐，自动生成配置）

使用 `--fine_tune` 参数，脚本会自动生成配置文件，无需手动创建：

```bash
python scripts/train_model.py --metadata data/metadata.csv --fine_tune \
    --output ./models/trained_model \
    --base_model tts_models/multilingual/multi-dataset/xtts_v2 \
    --epochs 100 --batch_size 4
```

**参数说明：**
- `--metadata`: metadata.csv文件路径
- `--fine_tune`: 启用fine-tuning模式（自动生成配置）
- `--output`: 模型输出目录
- `--base_model`: 基础模型（用于fine-tuning）
- `--epochs`: 训练轮数
- `--batch_size`: 批次大小

### 方式二：标准训练模式（需手动创建配置文件）

如果需要自定义配置，可以手动创建 `config.json`：

```bash
# 1. 从基础模型生成配置文件
python -m TTS --model_name tts_models/multilingual/multi-dataset/xtts_v2 --config_path config.json

# 2. 编辑 config.json，设置：
#    - datasets: 数据路径和metadata路径
#    - output_path: 模型保存路径
#    - trainer: 训练参数（epochs, batch_size等）

# 3. 开始训练
python scripts/train_model.py --metadata data/metadata.csv \
    --config config.json --output ./models/trained_model \
    --base_model tts_models/multilingual/multi-dataset/xtts_v2 \
    --epochs 100 --batch_size 4
```

**训练参数说明：**
- 训练时间：取决于数据量和硬件（GPU推荐）
- 模型保存：训练过程中会自动保存checkpoint
- 监控：查看训练日志了解进度

### 训练完成

训练完成后，模型文件保存在配置的 `output_path` 目录中，包含：
- `model_file.pth`: 模型权重
- `config.json`: 模型配置
- 其他相关文件

---

## 四、在其他项目中使用训练好的模型

### 1. 复制模型文件

将训练好的模型文件复制到新项目：

```
new_project/
└── models/
    ├── model_file.pth
    └── config.json
```

### 2. 加载和使用模型

在新项目中安装依赖后，加载模型：

```python
from TTS.api import TTS

# 加载训练好的模型
tts = TTS(model_path="models/model_file.pth", config_path="models/config.json")

# 生成语音
tts.tts_to_file(
    text="要合成的文本内容",
    file_path="output.wav"
)
```

### 3. 完整示例

```python
from TTS.api import TTS
import os

# 初始化模型
tts = TTS(
    model_path="models/your_trained_model.pth",
    config_path="models/config.json"
)

# 批量生成
texts = [
    "第一段文本",
    "第二段文本",
    "第三段文本"
]

output_dir = "output"
os.makedirs(output_dir, exist_ok=True)

for i, text in enumerate(texts, 1):
    output_path = f"{output_dir}/output_{i}.wav"
    tts.tts_to_file(text=text, file_path=output_path)
    print(f"已生成: {output_path}")
```

---

## 五、快速训练方案（使用XTTS-v2 Fine-tuning）

如果使用XTTS-v2进行微调，可以使用更简单的方式：

### 1. 准备数据

同上，准备音频和metadata.csv

### 2. 运行训练命令

```bash
tts train \
    --config_path config.json \
    --restore_path tts_models/multilingual/multi-dataset/xtts_v2 \
    --train_dataset_path metadata.csv \
    --output_path ./models/trained_model
```

### 3. 使用训练好的模型

```python
from TTS.api import TTS

# 加载微调后的模型
tts = TTS(model_path="./models/trained_model")

# 使用
tts.tts_to_file(text="文本内容", file_path="output.wav")
```

---

## 六、常见问题

**Q: 训练需要多长时间？**
A: 取决于数据量和硬件。GPU训练：几小时到几天；CPU训练：可能需要数天。

**Q: 最少需要多少音频？**
A: 建议至少5分钟总时长，50个以上音频文件。数据越多，效果越好。

**Q: 可以在CPU上训练吗？**
A: 可以，但速度很慢。强烈建议使用GPU。

**Q: 模型文件有多大？**
A: 通常几百MB到几GB，取决于模型类型。

**Q: 训练中断了怎么办？**
A: 可以从最近的checkpoint恢复训练，使用 `--restore_path` 参数。

---

## 七、参考资源

- [Coqui TTS官方文档](https://tts.readthedocs.io/)
- [训练指南](https://tts.readthedocs.io/en/latest/training/index.html)
- [XTTS Fine-tuning](https://github.com/coqui-ai/TTS/wiki/XTTS)
