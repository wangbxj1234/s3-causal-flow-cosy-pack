# S3 Causal Tokenizer + CosyVoice Flow (Standalone Repro Repo)

本 README 只保留读者需要的四个入口脚本（训练/推理最小闭环）：

1) `examples/train_tokenizer_causal_example.sh`（训练 causal tokenizer）  
2) `examples/prepare_flow_data_from_causal_tokenizer_example.sh`（导出 tokenizer + 生成 Flow 训练数据）  
3) `examples/train_flow_officialinit_causal_example.sh`（训练 Flow）  
4) `examples/infer_self_prompt_example.sh` / `examples/infer_cross_speaker_example.sh`（Flow 推理）

本仓库已内置 `cosyvoice`、`Matcha-TTS`、`s3tokenizer_train` 源码，不需要再额外克隆 CosyVoice 主仓库。

### 因果工作流（必读）

本项目的 **Flow 必须在与训练相同的数据协议上推理**：

1. **因果 S3 tokenizer**：在 `s3tokenizer_train` 的 **strict-causal / streaming** 设定下训练，导出为 `.pt`（例如 Drive 里的 `s3tokenizer_export_epoch15.pt`，放到 `pretrained_weights/s3tokenizer.pt`）。
2. **抽 token**：`tools/extract_speech_token_s3.py` 必须使用 **上述因果导出**，生成各 utterance 的 speech token，再打 parquet / `train.data.list`。
3. **训 Flow**：在 **该 token** 上训好的 checkpoint 才与推理一致。

历史上若曾用 **非因果** tokenizer 抽 token 却当作「因果线」训 Flow，会出现 token 分布与模型假设不一致。改正方式：用因果导出重做 `utt2speech_token.pt` → parquet → data.list → 重训（或使用和旧数据一致的旧 ckpt）。

**推荐推理入口**（自动使用 `pretrained_weights/s3tokenizer.pt` + `officialinit_causaldata` 实验目录下的最新 Flow）：

- `tools/infer_flow_reconstruct_causal_s3tok25hz.py`
- `tools/infer_flow_reconstruct_cross_speaker_causal_s3tok25hz.py`

底层仍调用 `infer_flow_reconstruct_*_s3tok25hz.py`；若需显式指定，继续传 `--tokenizer_pt` 或设置 `COSYVOICE_S3_TOKENIZER_PT`。  
本仓库不再提供 tokenizer 的“自动回退路径”。

---

## 0. 前置条件

- Linux（推荐 Ubuntu 20.04/22.04）
- Python 3.10
- NVIDIA GPU（推理推荐 12GB+ 显存）
- 已安装 CUDA 驱动（`nvidia-smi` 可用）

---

## 1. 获取代码

```bash
git clone https://github.com/wangbxj1234/s3-causal-flow-cosy-pack.git
cd s3-causal-flow-cosy-pack
```

---

## 2. 安装环境

### 2.1 创建虚拟环境

```bash
python3 -m venv .venv
source .venv/bin/activate
python -V
```

### 2.2 安装 PyTorch / TorchAudio

按官方安装页选择与你机器匹配的命令：  
[PyTorch Get Started](https://pytorch.org/get-started/locally/)

安装后验证：

```bash
python - <<'PY'
import torch, torchaudio
print("torch:", torch.__version__)
print("torchaudio:", torchaudio.__version__)
print("cuda_available:", torch.cuda.is_available())
PY
```

### 2.3 安装本项目依赖

```bash
pip install -r requirements-infer.txt
```

### 2.4 设置运行环境变量

每次新开 shell 需设置：

```bash
export PYTHONPATH="$(pwd):$(pwd)/third_party/Matcha-TTS"
```

---

## 3. 下载权重并放置到固定目录

### 3.1 权重来源

#### A) Google Drive（核心 3 个）

- [s3_causal_flow folder](https://drive.google.com/drive/folders/1KwHVm4fNiKRTt-LqZrDkDSN9k9hEgC4B?usp=drive_link)

使用其中：
- `epoch_199_whole.pt`（必需）
- `s3tokenizer_export_epoch15.pt`（必需）
- `cosyvoice_aishell_s3tok1024_25hz.yaml`（必需）
- `epoch_199_whole.yaml`（可选，元信息）

#### B) Hugging Face（附加 2 个，推理必需）

- [campplus.onnx](https://huggingface.co/FunAudioLLM/CosyVoice-300M/blob/main/campplus.onnx)
- [hift.pt](https://huggingface.co/FunAudioLLM/CosyVoice-300M/blob/main/hift.pt)

### 3.2 放置目录（严格按此结构）

```text
s3-causal-flow-cosy-pack/
  conf/
    cosyvoice_aishell_s3tok1024_25hz.yaml
  pretrained_weights/
    s3tokenizer.pt
    flow_torch_ddp/
      epoch_199_whole.pt
    CosyVoice-300M/
      campplus.onnx
      hift.pt
```

### 3.3 文件映射关系

- `s3tokenizer_export_epoch15.pt` -> `pretrained_weights/s3tokenizer.pt`
- `epoch_199_whole.pt` -> `pretrained_weights/flow_torch_ddp/epoch_199_whole.pt`
- `cosyvoice_aishell_s3tok1024_25hz.yaml` -> `conf/cosyvoice_aishell_s3tok1024_25hz.yaml`
- `campplus.onnx` -> `pretrained_weights/CosyVoice-300M/campplus.onnx`
- `hift.pt` -> `pretrained_weights/CosyVoice-300M/hift.pt`

### 3.4 下载后检查（必做）

```bash
ls -lh \
  pretrained_weights/s3tokenizer.pt \
  pretrained_weights/flow_torch_ddp/epoch_199_whole.pt \
  conf/cosyvoice_aishell_s3tok1024_25hz.yaml \
  pretrained_weights/CosyVoice-300M/campplus.onnx \
  pretrained_weights/CosyVoice-300M/hift.pt
```

> 注意：`custom_s3tok25hz` 模式下不再使用本地默认 tokenizer 回退。必须显式传 `--tokenizer_pt`，或设置 `COSYVOICE_S3_TOKENIZER_PT`。

---

## 4. 从 0 开始跑推理

以下命令均在仓库根目录执行，且已激活 `.venv`。

### 4.1 自提示重建（单条 wav）

**推荐（因果默认）**：权重按上文放置后，可直接：

```bash
python tools/infer_flow_reconstruct_causal_s3tok25hz.py \
  --wav /absolute/path/to/input.wav \
  --out_wav /absolute/path/to/out_self.wav \
  --assets_dir pretrained_weights/CosyVoice-300M \
  --n_timesteps 20
```

显式指定 ckpt / tokenizer 时用底层脚本：

```bash
python tools/infer_flow_reconstruct_s3tok25hz.py \
  --wav /absolute/path/to/input.wav \
  --out_wav /absolute/path/to/out_self.wav \
  --flow_ckpt pretrained_weights/flow_torch_ddp/epoch_199_whole.pt \
  --tokenizer_pt pretrained_weights/s3tokenizer.pt \
  --train_config conf/cosyvoice_aishell_s3tok1024_25hz.yaml \
  --assets_dir pretrained_weights/CosyVoice-300M \
  --n_timesteps 20
```

### 4.2 跨说话人重建（内容和声纹分离）

**推荐（因果默认）**：

```bash
python tools/infer_flow_reconstruct_cross_speaker_causal_s3tok25hz.py \
  --content_wav /absolute/path/to/content.wav \
  --speaker_wav /absolute/path/to/speaker_ref.wav \
  --out_wav /absolute/path/to/out_cross.wav \
  --assets_dir pretrained_weights/CosyVoice-300M \
  --n_timesteps 20
```

显式指定：

```bash
python tools/infer_flow_reconstruct_cross_speaker_s3tok25hz.py \
  --content_wav /absolute/path/to/content.wav \
  --speaker_wav /absolute/path/to/speaker_ref.wav \
  --out_wav /absolute/path/to/out_cross.wav \
  --flow_ckpt pretrained_weights/flow_torch_ddp/epoch_199_whole.pt \
  --tokenizer_pt pretrained_weights/s3tokenizer.pt \
  --train_config conf/cosyvoice_aishell_s3tok1024_25hz.yaml \
  --assets_dir pretrained_weights/CosyVoice-300M \
  --n_timesteps 20
```

### 4.3 Mel 指标验证

```bash
python tools/eval_flow_reconstruct_mel.py \
  --wav /absolute/path/to/input.wav \
  --preset custom_s3tok25hz \
  --flow_ckpt pretrained_weights/flow_torch_ddp/epoch_199_whole.pt \
  --tokenizer_pt pretrained_weights/s3tokenizer.pt \
  --train_config conf/cosyvoice_aishell_s3tok1024_25hz.yaml \
  --assets_dir pretrained_weights/CosyVoice-300M \
  --n_timesteps 20
```

---

## 5. 四个入口脚本（读者只看这部分）

### 1) 训练 causal tokenizer

```bash
bash examples/train_tokenizer_causal_example.sh
```

默认产物：`exp/s3tokenizer_causal_streaming/*`

### 2) 准备 Flow 训练数据（导出 tokenizer + 抽 token + parquet）

```bash
bash examples/prepare_flow_data_from_causal_tokenizer_example.sh
```

默认产物：

- `pretrained_weights/s3tokenizer.pt`
- `data/aishell_s3_causal_flow/train.data.list`
- `data/aishell_s3_causal_flow/dev.data.list`

### 3) 训练 Flow（官方 flow.pt 初始化）

```bash
bash examples/train_flow_officialinit_causal_example.sh
```

默认输入：`data/aishell_s3_causal_flow/train.data.list`、`data/aishell_s3_causal_flow/dev.data.list`  
默认产物：`exp/my_causal_flow/flow/torch_ddp/epoch_*_whole.pt`

### 4) 推理（自提示 / 跨说话人）

```bash
bash examples/infer_self_prompt_example.sh /abs/input.wav /abs/out_self.wav
bash examples/infer_cross_speaker_example.sh /abs/content.wav /abs/speaker.wav /abs/out_cross.wav
```

说明：这两个推理脚本都走 causal wrapper，默认使用 `pretrained_weights/s3tokenizer.pt`。

---

## 6. 常见错误与处理

- `ModuleNotFoundError: matcha`  
  检查是否设置：`export PYTHONPATH="$(pwd):$(pwd)/third_party/Matcha-TTS"`

- `custom preset needs --tokenizer_pt`  
  检查 `pretrained_weights/s3tokenizer.pt` 是否存在，或显式传 `--tokenizer_pt`

- `FileNotFoundError: campplus.onnx / hift.pt`  
  检查是否放在 `pretrained_weights/CosyVoice-300M/`

- `torchaudio` / CUDA symbol 报错  
  典型原因是 `torch` 与 `torchaudio` 版本不匹配，按同一官方渠道重装

- `watchdog` 是什么？  
  之前的 `scripts/host/s3tokenizer_watchdog_causal.sh` 是“保活脚本”，用于长训时被外部信号打断后自动拉起并续训。  
  它是内部运维工具，不是读者复现必需路径；当前读者文档已不依赖它。

---

## 7. 仓库结构

```text
.
├── cosyvoice/
├── third_party/Matcha-TTS/
├── s3tokenizer_train/
├── conf/
├── tools/
│   ├── infer_flow_reconstruct_causal_s3tok25hz.py   # 推荐：因果 tokenizer + causal-data Flow 默认
│   └── infer_flow_reconstruct_cross_speaker_causal_s3tok25hz.py
├── pretrained_weights/
├── examples/
│   ├── train_tokenizer_causal_example.sh
│   ├── prepare_flow_data_from_causal_tokenizer_example.sh
│   ├── train_flow_officialinit_causal_example.sh
│   ├── infer_self_prompt_example.sh
│   └── infer_cross_speaker_example.sh
├── requirements-infer.txt
└── ATTRIBUTION.md
```

上游来源与许可证见 [`ATTRIBUTION.md`](ATTRIBUTION.md)。
