# 音频格式转换脚本使用说明

## 功能

批量将 `data/origdata` 目录下的 m4a 音频文件转换为 wav 格式，并输出到 `data/audios` 文件夹中。

## 使用方法

### 基本使用

```bash
# 将 data/origdata 下的所有 m4a 文件转换为 wav，输出到 data/audios
python scripts/convert_m4a_to_wav.py
```

### 指定输入输出目录

```bash
python scripts/convert_m4a_to_wav.py \
    --input_dir data/origdata \
    --output_dir data/audios
```

### 指定采样率和声道

```bash
# 转换为 16kHz 单声道（适合TTS训练）
python scripts/convert_m4a_to_wav.py \
    --sample_rate 16000 \
    --channels 1

# 转换为 22050Hz 单声道（默认，推荐）
python scripts/convert_m4a_to_wav.py \
    --sample_rate 22050 \
    --channels 1
```

### 覆盖已存在的文件

```bash
# 如果输出文件已存在，覆盖它
python scripts/convert_m4a_to_wav.py --overwrite
```

## 完整参数说明

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--input_dir` | 输入目录（m4a文件所在目录） | `data/origdata` |
| `--output_dir` | 输出目录（wav文件保存目录） | `data/audios` |
| `--sample_rate` | 目标采样率（Hz） | `22050` |
| `--channels` | 目标声道数（1=单声道, 2=立体声） | `1` |
| `--overwrite` | 覆盖已存在的输出文件 | False |
| `--no_skip` | 不跳过已存在的文件 | False（默认跳过） |

## 功能特性

1. **自动扫描**：自动扫描输入目录下的所有 m4a 文件
2. **格式转换**：将 m4a 转换为 wav 格式
3. **重采样**：自动重采样到指定采样率
4. **声道转换**：自动转换为单声道或立体声
5. **音频归一化**：自动进行音量归一化处理
6. **智能跳过**：自动跳过已存在的文件（节省时间）
7. **进度显示**：显示转换进度和统计信息

## 处理流程

1. 扫描 `data/origdata` 目录下的所有 `.m4a` 文件
2. 对每个文件进行：
   - 格式转换（m4a -> wav）
   - 重采样到目标采样率
   - 转换为目标声道数（单声道/立体声）
   - 音量归一化
3. 保存到 `data/audios` 目录
4. 显示转换统计信息

## 示例

### 示例1：基本转换

```bash
# 1. 将 m4a 文件放入 data/origdata 目录
# 2. 运行转换脚本
python scripts/convert_m4a_to_wav.py

# 输出：
# - data/audios/*.wav（转换后的wav文件）
```

### 示例2：转换为16kHz单声道（TTS训练推荐）

```bash
python scripts/convert_m4a_to_wav.py \
    --input_dir data/origdata \
    --output_dir data/audios \
    --sample_rate 16000 \
    --channels 1
```

### 示例3：批量转换并覆盖已存在文件

```bash
python scripts/convert_m4a_to_wav.py \
    --overwrite
```

## 输出信息

脚本会显示：
- 扫描到的文件数量
- 每个文件的转换进度
- 文件大小变化
- 转换统计（成功/失败/跳过）

示例输出：
```
扫描到 50 个 m4a 文件
开始批量转换，共 50 个文件...
输出目录: data/audios
目标采样率: 22050 Hz
目标声道: 1 (单声道)
============================================================

[1/50] audio_001.m4a
正在转换: audio_001.m4a
完成: audio_001.m4a -> audio_001.wav
  大小: 2.34 MB -> 1.89 MB

...

============================================================
批量转换完成！
总计: 50 个文件
成功: 50 个
失败: 0 个
跳过: 0 个
============================================================
```

## 注意事项

1. **依赖库**：需要安装 `librosa` 和 `soundfile`（已在 requirements.txt 中）
2. **文件格式**：支持 .m4a 和 .M4A 格式
3. **处理时间**：转换时间取决于文件数量和大小
4. **磁盘空间**：确保有足够的磁盘空间存储转换后的文件
5. **文件命名**：输出文件名与输入文件名相同，只是扩展名改为 .wav

## 故障排除

### 问题1：找不到输入目录
- 检查 `--input_dir` 参数是否正确
- 确认目录存在且包含 m4a 文件

### 问题2：转换失败
- 检查音频文件是否损坏
- 确认已安装所有依赖：`pip install -r requirements.txt`
- 检查磁盘空间是否充足

### 问题3：输出文件已存在
- 使用 `--overwrite` 参数覆盖已存在的文件
- 或删除输出目录中的文件后重新运行

## Python API使用

```python
from scripts.convert_m4a_to_wav import AudioConverter

# 创建转换器
converter = AudioConverter(
    input_dir="data/origdata",
    output_dir="data/audios",
    target_sample_rate=22050,
    target_channels=1
)

# 批量转换
stats = converter.batch_convert(overwrite=False)
print(f"转换完成: {stats['success']} 个文件成功")
```

## 完整工作流程

```bash
# 1. 将原始 m4a 文件放入 data/origdata
# 2. 转换为 wav 格式
python scripts/convert_m4a_to_wav.py

# 3. 使用转换后的 wav 文件生成 metadata.csv
python scripts/auto_prepare_metadata.py
```

