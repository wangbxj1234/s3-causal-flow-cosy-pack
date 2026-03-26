# S3 Causal Tokenizer + CosyVoice Flow (Standalone Repro Repo)

本仓库提供可独立复现的 S3-causal Flow 推理环境，无需额外克隆完整 CosyVoice 主仓库。`cosyvoice`、`Matcha-TTS`、`s3tokenizer_train` 源码均已内置。

## 权重完整性检查

权重发布目录：
- [s3_causal_flow folder](https://drive.google.com/drive/folders/1KwHVm4fNiKRTt-LqZrDkDSN9k9hEgC4B?usp=drive_link)

当前目录中文件及作用：

| 文件 | 状态 | 作用 |
|------|--------|---------|
| `epoch_199_whole.pt` | 必需 | Flow checkpoint |
| `s3tokenizer_export_epoch15.pt` | 必需 | S3 tokenizer checkpoint |
| `cosyvoice_aishell_s3tok1024_25hz.yaml` | 必需 | Flow config |
| `epoch_199_whole.yaml` | 可选 | 训练元信息记录 |

还需补充以下 2 个文件（来自 CosyVoice-300M）：
- [campplus.onnx](https://huggingface.co/FunAudioLLM/CosyVoice-300M/blob/main/campplus.onnx)
- [hift.pt](https://huggingface.co/FunAudioLLM/CosyVoice-300M/blob/main/hift.pt)

---

## 3 分钟快速上手（最推荐）

在仓库根目录执行。

### 1) 环境

```bash
python3 -m venv .venv
source .venv/bin/activate

# 先按你的 CUDA/CPU 安装 torch + torchaudio
# https://pytorch.org/get-started/locally/

pip install -r requirements-infer.txt
export PYTHONPATH="$(pwd):$(pwd)/third_party/Matcha-TTS"
```

### 2) 放权重（按这个结构）

```text
pretrained_weights/
  s3tokenizer.pt
  flow_torch_ddp/
    epoch_199_whole.pt
  CosyVoice-300M/
    campplus.onnx
    hift.pt
```

对应来源：
- Drive：`s3tokenizer_export_epoch15.pt` -> `pretrained_weights/s3tokenizer.pt`
- Drive：`epoch_199_whole.pt` -> `pretrained_weights/flow_torch_ddp/epoch_199_whole.pt`
- Drive：`cosyvoice_aishell_s3tok1024_25hz.yaml` -> `conf/cosyvoice_aishell_s3tok1024_25hz.yaml`
- HF：`campplus.onnx`、`hift.pt` -> `pretrained_weights/CosyVoice-300M/`

### 3) 跑一条推理

```bash
python tools/infer_flow_reconstruct_s3tok25hz.py \
  --wav /path/to/input.wav \
  --out_wav /tmp/out_self.wav \
  --flow_ckpt pretrained_weights/flow_torch_ddp/epoch_199_whole.pt \
  --tokenizer_pt pretrained_weights/s3tokenizer.pt \
  --train_config conf/cosyvoice_aishell_s3tok1024_25hz.yaml \
  --assets_dir pretrained_weights/CosyVoice-300M \
  --n_timesteps 20
```

---

## 完整推理命令

### 自提示重建（single wav）

```bash
python tools/infer_flow_reconstruct_s3tok25hz.py \
  --wav /path/to/input.wav \
  --out_wav /tmp/out_self.wav \
  --flow_ckpt pretrained_weights/flow_torch_ddp/epoch_199_whole.pt \
  --tokenizer_pt pretrained_weights/s3tokenizer.pt \
  --train_config conf/cosyvoice_aishell_s3tok1024_25hz.yaml \
  --assets_dir pretrained_weights/CosyVoice-300M \
  --n_timesteps 20
```

### 跨说话人（content 与 speaker 分离）

```bash
python tools/infer_flow_reconstruct_cross_speaker_s3tok25hz.py \
  --content_wav /path/to/content.wav \
  --speaker_wav /path/to/speaker_ref.wav \
  --out_wav /tmp/out_cross.wav \
  --flow_ckpt pretrained_weights/flow_torch_ddp/epoch_199_whole.pt \
  --tokenizer_pt pretrained_weights/s3tokenizer.pt \
  --train_config conf/cosyvoice_aishell_s3tok1024_25hz.yaml \
  --assets_dir pretrained_weights/CosyVoice-300M \
  --n_timesteps 20
```

### Mel 指标评估

```bash
python tools/eval_flow_reconstruct_mel.py \
  --wav /path/to/input.wav \
  --preset custom_s3tok25hz \
  --flow_ckpt pretrained_weights/flow_torch_ddp/epoch_199_whole.pt \
  --tokenizer_pt pretrained_weights/s3tokenizer.pt \
  --train_config conf/cosyvoice_aishell_s3tok1024_25hz.yaml \
  --assets_dir pretrained_weights/CosyVoice-300M \
  --n_timesteps 20
```

---

## 训练复现（脚本职责与顺序）

已包含：
- `s3tokenizer_train/`（因果 tokenizer 训练与导出）
- `cosyvoice/bin/train.py`（Flow 训练入口）
- `tools/extract_speech_token_s3.py`、`tools/make_parquet_list.py`（数据准备）

三份训练脚本的定位如下：

| 脚本 | 作用 | 输入前提 | 产出 |
|------|------|----------|------|
| `scripts/host/s3tokenizer_watchdog_causal.sh` | 训练因果 S3 tokenizer（watchdog 自动重启/续训） | `data/aishell_s3/{train,dev}` 已准备 | `exp/s3tokenizer_causal_streaming/` 下 checkpoint，后续可导出 `.pt` |
| `scripts/host/run_cosyvoice1_flow_aishell_s3.sh` | 一条龙：导出 tokenizer、抽 token、做 parquet、生成 data.list、启动 Flow | 本机路径已按脚本写死；默认走 `step2_c_commit005` 相关路径 | Flow 训练目录 + `train.data.list`/`dev.data.list` |
| `examples/train_flow_officialinit_causal_example.sh` | 仅启动 Flow 训练（官方 `flow.pt` 初始化） | 已有 `train.data.list`/`dev.data.list` + `flow.pt` + campplus 资源 | `exp/my_causal_flow/flow/torch_ddp` |

推荐顺序（更清晰、可控）：
1. 先跑 `scripts/host/s3tokenizer_watchdog_causal.sh` 训练 tokenizer，并导出 `s3tokenizer.pt`。
2. 使用 `tools/extract_speech_token_s3.py` + `tools/make_parquet_list.py` 准备 `train.data.list` / `dev.data.list`。
3. 最后跑 `examples/train_flow_officialinit_causal_example.sh` 启动 Flow 训练。

说明：`scripts/host/run_cosyvoice1_flow_aishell_s3.sh` 更像“历史实验的一键脚本模板”，包含较多硬编码路径。新环境复现时建议按上面 1→2→3 拆步执行，问题更容易定位。

---

## 常见问题（最快定位）

- `ModuleNotFoundError: matcha`：确认 `PYTHONPATH` 包含 `$(pwd)/third_party/Matcha-TTS`。
- `custom preset needs --tokenizer_pt`：检查 `pretrained_weights/s3tokenizer.pt` 是否存在或显式传 `--tokenizer_pt`。
- `FileNotFoundError: campplus.onnx / hift.pt`：确认放在 `pretrained_weights/CosyVoice-300M/`。
- `torchaudio` / CUDA 符号错误：通常是 `torch` 与 `torchaudio` 版本不匹配，重装成同一渠道版本。

---

## 目录结构（摘要）

```text
.
├── cosyvoice/
├── third_party/Matcha-TTS/
├── s3tokenizer_train/
├── conf/
├── tools/
├── pretrained_weights/
├── examples/
├── scripts/host/
├── requirements-infer.txt
└── ATTRIBUTION.md
```

上游来源与许可证见 [`ATTRIBUTION.md`](ATTRIBUTION.md)。
