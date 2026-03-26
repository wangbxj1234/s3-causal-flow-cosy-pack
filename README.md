# S3 因果 Tokenizer + CosyVoice Flow：独立仓库（从 0 推理验证）

克隆本仓库后**不需要**再单独下载 CosyVoice 主仓库即可跑通 **自训 Flow + S3 tokenizer** 的推理脚本；`cosyvoice` Python 包、`Matcha-TTS`、`s3tokenizer_train` 与训练用 yaml 均已**随仓库提供**（见 [`ATTRIBUTION.md`](ATTRIBUTION.md) 中的上游版权说明）。

**权重仍需自备**：放到 `pretrained_weights/`，说明见 [`pretrained_weights/README.md`](pretrained_weights/README.md)。

---

## 目录结构（摘要）

```
.
├── cosyvoice/                 # 来自 FunAudioLLM/CosyVoice（Apache-2.0）
├── third_party/Matcha-TTS/    # 来自 shivammehta25/Matcha-TTS
├── s3tokenizer_train/         # 因果 S3Tokenizer 训练/导出代码
├── conf/
│   └── cosyvoice_aishell_s3tok1024_25hz.yaml
├── tools/
│   ├── infer_flow_reconstruct_s3tok25hz.py          # 自提示重建
│   ├── infer_flow_reconstruct_cross_speaker_s3tok25hz.py  # 跨说话人
│   ├── eval_flow_reconstruct_mel.py               # mel 指标
│   ├── extract_speech_token_s3.py                 # 用 S3 .pt 批量抽 token（训练数据准备）
│   └── make_parquet_list.py                       # wav.scp → parquet + list（CosyVoice 数据格式）
├── pretrained_weights/        # 仅 README + 占位；大文件不放 Git
├── examples/                  # 一键示例 shell
├── scripts/host/              # 训练用 host 脚本（路径需自行改）
├── requirements-infer.txt
└── ATTRIBUTION.md
```

---

## 环境（从 0）

1. **Python 3.10**（与 CosyVoice 常用版本一致；其它 3.9+ 可试）。
2. 安装 **PyTorch + TorchAudio**（按你 CUDA/CPU 从 [pytorch.org](https://pytorch.org) 选择）。
3. 安装推理依赖：

```bash
cd /path/to/this/repo
pip install -r requirements-infer.txt
```

4. 运行时 **PYTHONPATH** 需包含**仓库根目录**与 **Matcha-TTS 根**（下面示例已写）。

---

## 准备权重（必须）

1. **`campplus.onnx`**、**`hift.pt`** 请从官方 **CosyVoice-300M** 仓库自行下载，放入 `pretrained_weights/CosyVoice-300M/`（与 token 无关，不必随本仓库分发）：
   - [campplus.onnx](https://huggingface.co/FunAudioLLM/CosyVoice-300M/blob/main/campplus.onnx)
   - [hift.pt](https://huggingface.co/FunAudioLLM/CosyVoice-300M/blob/main/hift.pt)  
   仓库首页（可一次拉整包）：[FunAudioLLM/CosyVoice-300M](https://huggingface.co/FunAudioLLM/CosyVoice-300M)。命令行见 [`pretrained_weights/README.md`](pretrained_weights/README.md)。
2. 将 **自训 S3 tokenizer** 导出文件命名为 **`pretrained_weights/s3tokenizer.pt`**（或任意路径 + 环境变量 `COSYVOICE_S3_TOKENIZER_PT`）。
3. 将 **Flow** 的 `epoch_*_whole.pt` 放入 **`pretrained_weights/flow_torch_ddp/`**（可多个；脚本默认选**最大 epoch**）。

**不要**混用：不同实验的 Flow / tokenizer / yaml 必须成套。

---

## 推理命令（自训 preset，默认）

在**仓库根目录**执行：

```bash
export PYTHONPATH="$(pwd):$(pwd)/third_party/Matcha-TTS"
python tools/infer_flow_reconstruct_s3tok25hz.py \
  --wav /path/to/input.wav \
  --out_wav /tmp/out.wav \
  --n_timesteps 20
```

可选显式指定：

```bash
python tools/infer_flow_reconstruct_s3tok25hz.py \
  --wav in.wav --out_wav out.wav \
  --flow_ckpt pretrained_weights/flow_torch_ddp/epoch_199_whole.pt \
  --tokenizer_pt pretrained_weights/s3tokenizer.pt \
  --train_config conf/cosyvoice_aishell_s3tok1024_25hz.yaml \
  --assets_dir pretrained_weights/CosyVoice-300M \
  --n_timesteps 20
```

**跨说话人**（内容一条 wav，声纹另一条）：

```bash
python tools/infer_flow_reconstruct_cross_speaker_s3tok25hz.py \
  --content_wav A.wav --speaker_wav B.wav --out_wav cross.wav --n_timesteps 20
```

**Mel 指标**：

```bash
python tools/eval_flow_reconstruct_mel.py --wav in.wav --preset custom_s3tok25hz --n_timesteps 20
```

**官方 50Hz 对比**（需 `pretrained_weights/CosyVoice-300M/` 内含官方 `flow.pt`、`cosyvoice.yaml`、`speech_tokenizer_v1.onnx`）：

```bash
python tools/infer_flow_reconstruct_s3tok25hz.py \
  --preset official_cosyvoice1_50hz --wav in.wav --out_wav official.wav
```

---

## 示例脚本

```bash
chmod +x examples/*.sh
# 编辑 examples 内路径后：
./examples/infer_self_prompt_example.sh /path/to.wav /tmp/out.wav
```

---

## 训练（因果 tokenizer / Flow）

训练依赖与多卡逻辑更重，请参考 `scripts/host/` 下脚本及上游 CosyVoice 文档；训练前请自行修改脚本中的数据路径与 `torchrun` 参数。

---

## 获取代码

一般直接使用 Git 即可（便于同步更新与对照版本）：

```bash
git clone https://github.com/<你的用户名>/<仓库名>.git
cd <仓库名>
```

若需要把某次检出的源码打成压缩包自行备份或离线传递，在**该目录的上一级**执行即可，例如：

```bash
cd ..
tar -czvf my-snapshot.tar.gz <仓库目录名>
```

解压：`tar -xzvf my-snapshot.tar.gz`。
