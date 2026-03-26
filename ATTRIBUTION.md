# 第三方与上游代码说明

本仓库为**可独立克隆**的推理与训练脚本合集，内含从上游复制的源码，**版权与许可证归各自项目**。使用请遵守原项目协议。

| 组件 | 来源 | 许可证 / 说明 |
|------|------|----------------|
| `cosyvoice/` | [FunAudioLLM/CosyVoice](https://github.com/FunAudioLLM/CosyVoice) | 见根目录 `LICENSE.CosyVoice`（Apache 2.0） |
| `third_party/Matcha-TTS/` | [shivammehta25/Matcha-TTS](https://github.com/shivammehta25/Matcha-TTS) | 见 `LICENSE.Matcha-TTS`（若存在） |
| `conf/cosyvoice_aishell_s3tok1024_25hz.yaml` | 基于 CosyVoice 示例配置改编 / 沿用 | 与 CosyVoice 一致 |
| `tools/infer_*.py`, `tools/eval_*.py` | 在 CosyVoice 工具脚本基础上修改路径与说明 | 与 CosyVoice 一致 |
| `tools/extract_speech_token_s3.py`, `tools/make_parquet_list.py` | CosyVoice `tools/` 拷贝（后者仅头注释保留上游版权） | Apache 2.0（CosyVoice） |
| `s3tokenizer_train/` | 本工作区因果 S3Tokenizer 训练与导出代码 | 若对外发布请自行标注你的许可证 |

**权重不随仓库分发**：`CosyVoice-300M`、自训 Flow、S3 tokenizer 等见 `pretrained_weights/README.md`。
