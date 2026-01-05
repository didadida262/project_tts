# 自动生成metadata.csv脚本使用说明

## 功能

自动扫描 `data/audios` 目录下的音频文件，使用 Whisper 自动转录音频生成文本，并按照 PRD 要求的格式生成 `metadata.csv` 文件。

## 使用方法

### 基本使用

```bash
# 扫描 data/audios 目录，自动转录音频，生成 data/metadata.csv
python scripts/auto_prepare_metadata.py
```

### 指定目录和输出文件

```bash
python scripts/auto_prepare_metadata.py \
    --audio_dir data/audios \
    --output data/metadata.csv
```

### 使用不同的Whisper模型

```bash
# 使用更小的模型（更快，但准确率略低）
python scripts/auto_prepare_metadata.py --whisper_model tiny

# 使用更大的模型（更准确，但更慢）
python scripts/auto_prepare_metadata.py --whisper_model small
```

### 指定语言

```bash
# 英文音频
python scripts/auto_prepare_metadata.py --language en

# 中文音频（默认）
python scripts/auto_prepare_metadata.py --language zh
```

### 跳过已有文本文件

```bash
# 如果文本文件已存在，跳过转录（节省时间）
python scripts/auto_prepare_metadata.py --skip_existing
```

### 不使用转录（仅使用已有文本文件）

```bash
# 不转录音频，只读取已有的 .txt 文件
python scripts/auto_prepare_metadata.py --no_transcribe
```

## 完整参数说明

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--audio_dir` | 音频文件目录 | `data/audios` |
| `--output` | 输出的metadata.csv路径 | `data/metadata.csv` |
| `--whisper_model` | Whisper模型大小 (tiny/base/small/medium/large) | `base` |
| `--language` | 音频语言代码 | `zh` |
| `--no_transcribe` | 不转录音频，只使用已有文本文件 | False |
| `--skip_existing` | 如果文本文件已存在，跳过转录 | False |

## 工作流程

1. **扫描音频文件**：扫描指定目录下的所有音频文件（支持 .wav, .mp3, .flac, .m4a 等格式）
2. **获取音频信息**：获取每个音频文件的时长、采样率等信息
3. **转录音频**：使用 Whisper 自动转录音频生成文本
4. **保存文本文件**：将转录的文本保存为同名的 .txt 文件
5. **生成metadata.csv**：按照 `audio_path|text` 格式生成 metadata.csv

## 输出格式

生成的 `metadata.csv` 文件格式：

```csv
audio_path|text
data/audios/audio_001.wav|这是第一段音频的文本内容
data/audios/audio_002.wav|这是第二段音频的文本内容
data/audios/audio_003.wav|这是第三段音频的文本内容
```

## 示例

### 示例1：基本使用

```bash
# 1. 将音频文件放入 data/audios 目录
# 2. 运行脚本
python scripts/auto_prepare_metadata.py

# 输出：
# - data/metadata.csv（metadata文件）
# - data/audios/*.txt（每个音频对应的文本文件）
```

### 示例2：批量处理大量音频

```bash
# 使用较小的模型加快速度
python scripts/auto_prepare_metadata.py \
    --whisper_model tiny \
    --audio_dir data/audios \
    --output data/metadata.csv
```

### 示例3：处理英文音频

```bash
python scripts/auto_prepare_metadata.py \
    --language en \
    --audio_dir data/audios_en \
    --output data/metadata_en.csv
```

## 注意事项

1. **首次使用**：首次运行会自动下载 Whisper 模型，需要一些时间
2. **依赖安装**：确保已安装 `openai-whisper`：
   ```bash
   pip install openai-whisper
   ```
3. **音频格式**：支持常见音频格式，但推荐使用 WAV 格式
4. **处理时间**：转录时间取决于音频数量和长度，以及使用的模型大小
5. **文本文件**：脚本会自动为每个音频生成对应的 .txt 文本文件

## 故障排除

### 问题1：找不到音频文件
- 检查 `--audio_dir` 参数是否正确
- 确认音频文件在指定目录中

### 问题2：Whisper未安装
```bash
pip install openai-whisper
```

### 问题3：转录失败
- 检查音频文件是否损坏
- 尝试使用不同的 Whisper 模型
- 检查语言参数是否正确

### 问题4：内存不足
- 使用较小的 Whisper 模型（如 `tiny` 或 `base`）
- 分批处理音频文件

## Python API使用

```python
from scripts.auto_prepare_metadata import AutoMetadataGenerator

# 创建生成器
generator = AutoMetadataGenerator(
    audio_dir="data/audios",
    output_file="data/metadata.csv",
    whisper_model="base",
    language="zh"
)

# 生成metadata
metadata_path = generator.generate_metadata(transcribe=True)
print(f"metadata.csv已生成: {metadata_path}")
```

