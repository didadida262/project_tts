# Coqui TTS 声音克隆项目

基于Coqui TTS的声音克隆系统，能够从少量音频样本中学习并克隆指定说话人的声音特征。

## 项目特性

- 🎯 **高质量声音克隆**: 使用XTTS-v2模型实现高质量的声音克隆
- 🌍 **多语言支持**: 支持中文、英文等17种语言
- ⚡ **快速合成**: GPU加速支持，快速生成语音
- 🎛️ **灵活配置**: 支持语速、音调等参数调整
- 📦 **易于使用**: 简单的API和命令行接口

## 快速开始

### 安装

```bash
# 1. 创建虚拟环境
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac

# 2. 安装PyTorch (CPU版本)
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu

# 3. 安装依赖
pip install -r requirements.txt
```

详细安装说明请查看 [INSTALL.md](INSTALL.md)

### 基本使用

```python
from src.voice_cloner import VoiceCloner

# 初始化
cloner = VoiceCloner()

# 克隆声音
cloner.clone_voice(
    reference_audio="reference.wav",
    text="你好，这是声音克隆测试。",
    output_path="output.wav",
    language="zh"
)
```

更多使用示例请查看 [USAGE.md](USAGE.md)

## 项目结构

```
project_tts/
├── README.md              # 项目说明
├── PRD.md                 # 产品需求文档
├── INSTALL.md             # 安装指南
├── USAGE.md               # 使用指南
├── requirements.txt       # 依赖列表
├── config.yaml            # 配置文件
├── src/                   # 源代码
│   ├── voice_cloner.py    # 声音克隆核心
│   ├── audio_processor.py # 音频处理
│   └── model_manager.py   # 模型管理
├── examples/              # 示例文件
│   ├── reference_audio/   # 参考音频
│   └── output/            # 输出音频
└── scripts/               # 脚本文件
    └── clone_voice.py     # 命令行工具
```

## 文档

- [PRD.md](PRD.md) - 完整的产品需求文档，包含详细的技术方案和实施步骤
- [INSTALL.md](INSTALL.md) - 详细的安装指南
- [USAGE.md](USAGE.md) - 使用指南和示例

## 系统要求

- Python 3.8+
- 4GB+ 内存（推荐8GB+）
- 可选：NVIDIA GPU（CUDA 11.8+ 或 12.1+）

## 支持的语言

中文(zh)、英语(en)、西班牙语(es)、法语(fr)、德语(de)、意大利语(it)、葡萄牙语(pt)、波兰语(pl)、土耳其语(tr)、俄语(ru)、荷兰语(nl)、捷克语(cs)、阿拉伯语(ar)、日语(ja)、匈牙利语(hu)、韩语(ko)

## 注意事项

- 参考音频需要3-10秒，清晰无噪音
- 首次运行会自动下载模型（约1-2GB）
- 仅用于合法用途，尊重他人声音权

## 参考资料

- [Coqui TTS官方文档](https://tts.readthedocs.io/)
- [Coqui TTS GitHub](https://github.com/coqui-ai/TTS)
- [XTTS模型说明](https://github.com/coqui-ai/TTS/wiki/XTTS)

## 许可证

本项目遵循相应的开源许可证。

## 贡献

欢迎提交Issue和Pull Request！

---

**版本**: v1.0  
**最后更新**: 2024