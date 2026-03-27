# S3 Causal Tokenizer + CosyVoice Flow

## Quick start（推理）

```bash
git clone https://github.com/wangbxj1234/s3-causal-flow-cosy-pack.git
cd s3-causal-flow-cosy-pack

python3 -m venv .venv && source .venv/bin/activate
# 先安装 torch / torchaudio：https://pytorch.org/get-started/locally/
pip install -r requirements-infer.txt
export PYTHONPATH="$(pwd):$(pwd)/third_party/Matcha-TTS"
```

按下面「权重」放好文件后：

```bash
bash examples/infer_self_prompt_example.sh /path/to/in.wav /path/to/out.wav
bash examples/infer_cross_speaker_example.sh /path/to/content.wav /path/to/speaker.wav /path/to/out.wav
```

或直接：

```bash
python tools/infer_flow_reconstruct_causal_s3tok25hz.py \
  --wav /path/to/in.wav --out_wav /path/to/out.wav \
  --assets_dir pretrained_weights/CosyVoice-300M --n_timesteps 20
```

需要 `pretrained_weights/s3tokenizer.pt`、`pretrained_weights/flow_torch_ddp/` 下 Flow 权重、`pretrained_weights/CosyVoice-300M/` 下 `campplus.onnx` 与 `hift.pt`（见下）。

---

## 权重

| 来源 | 内容 |
|------|------|
| [Google Drive](https://drive.google.com/drive/folders/1KwHVm4fNiKRTt-LqZrDkDSN9k9hEgC4B?usp=drive_link) | `s3tokenizer_export_epoch15.pt`、`epoch_199_whole.pt`、`cosyvoice_aishell_s3tok1024_25hz.yaml` |
| [Hugging Face](https://huggingface.co/FunAudioLLM/CosyVoice-300M) | `campplus.onnx`、`hift.pt`；训练 Flow 时还需要同目录的 `flow.pt` |

目录示例：

```text
pretrained_weights/s3tokenizer.pt
pretrained_weights/flow_torch_ddp/epoch_199_whole.pt
pretrained_weights/CosyVoice-300M/campplus.onnx
pretrained_weights/CosyVoice-300M/hift.pt
pretrained_weights/CosyVoice-300M/flow.pt    # 仅训练 Flow 时需要
conf/cosyvoice_aishell_s3tok1024_25hz.yaml
```

将 Drive 里的 `s3tokenizer_export_epoch15.pt` 复制为 `pretrained_weights/s3tokenizer.pt`，`epoch_199_whole.pt` 放到 `pretrained_weights/flow_torch_ddp/`。

---

## 训练（可选，按顺序）

```bash
bash examples/train_tokenizer_causal_example.sh
bash examples/prepare_flow_data_from_causal_tokenizer_example.sh
bash examples/train_flow_officialinit_causal_example.sh
```

数据与路径默认值写在各脚本顶部，按需改 `DATA_ROOT`、`CUDA_VISIBLE_DEVICES` 等。

---

## 常见问题

- `ModuleNotFoundError: matcha`：确认已 `export PYTHONPATH="$(pwd):$(pwd)/third_party/Matcha-TTS"`  
- 缺 tokenizer：放置 `pretrained_weights/s3tokenizer.pt`，或传 `--tokenizer_pt` / 设置 `COSYVOICE_S3_TOKENIZER_PT`  
- `torch` / `torchaudio` 版本不一致：用同一 PyTorch 官方渠道重装  

许可与上游说明见 [`ATTRIBUTION.md`](ATTRIBUTION.md)。
