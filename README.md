# S3 Causal Tokenizer + CosyVoice Flow (Standalone Repro Repo)

本 README 面向从零开始的复现流程。按顺序执行即可完成：
1) 环境安装  
2) 权重下载与放置  
3) 推理验证  
4)（可选）训练复现

本仓库已内置 `cosyvoice`、`Matcha-TTS`、`s3tokenizer_train` 源码，不需要再额外克隆 CosyVoice 主仓库。

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

---

## 4. 从 0 开始跑推理

以下命令均在仓库根目录执行，且已激活 `.venv`。

### 4.1 自提示重建（单条 wav）

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

## 5. 训练复现（完整流程）

本仓库提供训练相关源码与脚本。推荐按下列顺序执行，不要跳步。

### 步骤 1：训练因果 S3 tokenizer

脚本：`scripts/host/s3tokenizer_watchdog_causal.sh`

作用：
- 启动 `s3tokenizer_train.train`
- 断开后自动拉起
- 有 checkpoint 时自动 resume

输入前提：
- `data/aishell_s3/train`
- `data/aishell_s3/dev`

### 步骤 2：导出 tokenizer + 生成 Flow 训练数据

可使用：
- `tools/extract_speech_token_s3.py`
- `tools/make_parquet_list.py`

目标产物：
- `train.data.list`
- `dev.data.list`

### 步骤 3：训练 Flow（官方 flow.pt 初始化）

脚本：`examples/train_flow_officialinit_causal_example.sh`

需要先改这些变量：
- `TRAIN_LIST`
- `CV_LIST`
- `ONNX_PATH`
- `NPROC`
- `CUDA_VISIBLE_DEVICES`

运行后产物：
- `exp/.../flow/torch_ddp/epoch_*_whole.pt`

### 说明：另一份 host 一键脚本

`scripts/host/run_cosyvoice1_flow_aishell_s3.sh` 会串联：
- 导出 tokenizer
- 抽 token
- 生成 parquet / data.list
- 启动 Flow

该脚本内有较多本机路径，跨机器复现前必须逐项修改。

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

---

## 7. 仓库结构

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
