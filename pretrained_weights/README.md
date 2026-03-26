# 权重放置说明（推理必需）

本目录**默认不上传大文件**。按下列结构放入你从 Google Drive / Hugging Face 下载的文件后即可运行 `tools/` 下脚本。

## 1. 自训链路（1024 词表 / 25 Hz + S3）

| 路径 | 内容 |
|------|------|
| `s3tokenizer.pt` | 与训练一致的 S3Tokenizer 导出（`torch.load` 含 `config` + `model`） |
| `flow_torch_ddp/epoch_*_whole.pt` | 自训 Flow 检查点（至少放一个；脚本会取目录下**最大** epoch 号） |

## 2. CosyVoice-300M 附属（声码器 + 说话人，与 token 语义无关）

从 [FunAudioLLM/CosyVoice-300M](https://huggingface.co/FunAudioLLM/CosyVoice-300M) 下载后，放入：

```
pretrained_weights/CosyVoice-300M/
  campplus.onnx
  hift.pt
```

（可选）若要用 **`--preset official_cosyvoice1_50hz`** 做官方对比，同一目录还需：

- `flow.pt`
- `cosyvoice.yaml`
- `speech_tokenizer_v1.onnx`

示例（需已安装 `huggingface-cli`）：

```bash
HF_HUB_ENABLE_HF_TRANSFER=0 huggingface-cli download FunAudioLLM/CosyVoice-300M \
  --local-dir pretrained_weights/CosyVoice-300M
```

下载后把自训的 `s3tokenizer.pt` 与 `flow_torch_ddp/*.pt` 仍按上表放在 `pretrained_weights/` 下即可。
