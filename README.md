# S3 causal tokenizer → causal Flow training → Flow 推理验证（脚本包）

本目录汇总 **从训练因果 S3Tokenizer、准备 causal-token 数据、在官方 `flow.pt` 初始化下训练 Flow**，到 **自提示重建 / 跨说话人 / mel 指标** 验证所用脚本与示例命令。权重与大数据（AISHELL、checkpoint）**不包含**在内，需自备或按原项目获取。

## 推荐目录布局

```text
your_workspace/
  s3tokenizer_train/          # 本仓库或子模块：S3Tokenizer 训练与 export
  data/
    aishell_raw/              # AISHELL-1 原始或解压 wav
    aishell_s3/               # wav.scp / text 等（prepare 脚本生成）
    aishell_s3_causal_flow/   # 用因果 tokenizer 重抽 token 后的 parquet + *.data.list
  CosyVoice-main/             # CosyVoice + third_party/Matcha-TTS（见上游）
    pretrained_models/CosyVoice-300M/   # flow.pt, cosyvoice.yaml, speech_tokenizer_v1.onnx, campplus, hift
    tools/                    # 将本包 scripts/cosyvoice_tools/*.py 拷入此处（与仓库内其它 tools 并列）
```

环境变量（示例）：

- `FT_COSY`：`your_workspace` 根（含 `s3tokenizer_train` 与 `data`）
- `COSYVOICE_ROOT`：`CosyVoice-main` 绝对路径

## 环境说明（参考机）

| 项 | 说明 |
|----|------|
| OS / GPU | Linux + NVIDIA CUDA（训练曾用多卡；推理单卡即可） |
| Python | **3.10**（与 CosyVoice / torch 生态常见版本一致） |
| PyTorch | 带 **CUDA** 的 `torch` / `torchaudio`（版本需与驱动匹配） |
| 解释器 | 参考环境使用独立 venv：`marco/bin/python`；你可改用 `conda` / `venv` + `pip install` |
| 关键 pip | `onnxruntime`、`hyperpyyaml`、`omegaconf`、**openai-whisper**、`soundfile`；CosyVoice 及 Matcha 子模块的依赖见上游 `requirements.txt` / `FAQ.md` |
| `PYTHONPATH` | 训练/推理需包含：`${FT_COSY}`、`${COSYVOICE_ROOT}`、`${COSYVOICE_ROOT}/third_party/Matcha-TTS` |
| 可选 | `COSYVOICE_S3_TOKENIZER_PT`：S3 导出 tokenizer `.pt`；`COSYVOICE_S3TOKENIZER_ROOT`：含 `s3tokenizer_train` 的父目录 |

更杂的依赖名见本目录 `requirements-notes.txt`（非严格 lockfile）。

## 本包内容

| 路径 | 作用 |
|------|------|
| `scripts/cosyvoice_tools/infer_flow_reconstruct_s3tok25hz.py` | 同源 wav 自提示 Flow 重建 + HiFT 试听 |
| `scripts/cosyvoice_tools/infer_flow_reconstruct_cross_speaker_s3tok25hz.py` | 内容 wav + **独立** speaker wav → 只换 CAMPPlus 声纹 |
| `scripts/cosyvoice_tools/eval_flow_reconstruct_mel.py` | 与上同一协议，输出 pred vs GT mel 指标（JSONL） |
| `scripts/host/s3tokenizer_watchdog_causal.sh` | **因果 S3Tokenizer** 长时间训练 + 断线自恢复（`exp/s3tokenizer_causal_streaming`） |
| `scripts/host/run_cosyvoice1_flow_aishell_s3.sh` | **非因果** `aishell_s3` 数据：从 export tokenizer 到 parquet、`data.list`、启动 Flow（可作模板改路径） |
| `examples/*.sh` | 推理 / eval / official-init Flow 训练的一键示例（需改路径） |

**安装脚本到 CosyVoice：** 将 `scripts/cosyvoice_tools/` 下三个 `.py` **复制到** `CosyVoice-main/tools/`（与仓库原有 `extract_speech_token_s3.py` 等并列）。跨说话人脚本会优先加载**同目录**下的 `infer_flow_reconstruct_s3tok25hz.py`。

## 流程 1：训练因果 S3Tokenizer

1. 准备 `data/aishell_s3/{train,dev}`（如 `prepare_aishell_s3.sh`，需按你本机数据路径改脚本）。
2. 使用 **因果** 配置启动训练（参考 `scripts/host/s3tokenizer_watchdog_causal.sh` 内 `torchrun` 参数：`--batch_size`、`--grad_accum_steps`、`--persistent_workers` 等）。
3. 训练结束后用 `python -m s3tokenizer_train.export` 导出 **25 token/s、1024 码本** 的 `S3TokenizerV1` `.pt`（具体参数与 checkpoint 路径见 `s3tokenizer_train` 文档）。

## 流程 2：构建 causal-flow 用数据并训练 Flow

1. 使用 `CosyVoice-main/tools/extract_speech_token_s3.py`（**未包含在本小包**，请用完整 CosyVoice 仓库）对 `train`/`dev` 用**因果 tokenizer 导出**重抽 token，再 `make_parquet_list.py` 生成 parquet 与 `train.data.list` / `dev.data.list`，指向例如 `data/aishell_s3_causal_flow/`（与 `aishell_s3` 分离，避免混用 token 语义）。
2. Flow 配置使用 **`cosyvoice_aishell_s3tok1024_25hz.yaml`**（1024 词表、25 Hz；与 `run_cosyvoice1_flow_aishell_s3.sh` 里生成的 yaml 思路一致）。
3. **推荐**：从 **`pretrained_models/CosyVoice-300M/flow.pt`** 初始化（`train.py` 会跳过 `input_embedding` 等 shape 不匹配的键，其余加载）。示例命令见 `examples/train_flow_officialinit_causal_example.sh`（需填写 `FLOW_PT`、`--onnx_path`、`MODEL_DIR` 等）。

## 流程 3：推理验证

### 自提示重建（同源）

- 脚本：`tools/infer_flow_reconstruct_s3tok25hz.py`
- **必须成套**：自训 Flow checkpoint + 与训练一致的 yaml + **数据管线同一套** S3 `tokenizer_pt`；**不要**与官方 50Hz ONNX tokenizer 混用。
- 官方对比：`--preset official_cosyvoice1_50hz`（仅 CosyVoice-300M 目录内 yaml + flow + onnx）。

### 跨说话人（只听 Flow 对 spk 条件的响应）

- 脚本：`tools/infer_flow_reconstruct_cross_speaker_s3tok25hz.py`
- `--content_wav`：token + mel；`--speaker_wav`：**仅** CAMPPlus embedding。

### Mel 数值

- 脚本：`tools/eval_flow_reconstruct_mel.py`，`--preset` 与 ckpt/tokenizer 与 infer 对齐。

一键示例（需先 `export COSYVOICE_ROOT` `FT_COSY` 且已拷贝 py 到 `tools/`）：

```bash
chmod +x examples/*.sh
export COSYVOICE_ROOT=/path/to/CosyVoice-main FT_COSY=/path/to/workspace
./examples/infer_self_prompt_example.sh
./examples/infer_cross_speaker_example.sh
```

## 打 tarball 上传 GitHub

在包含本目录的父目录下执行（将 `PARENT` 换成你的路径）：

```bash
tar -czvf s3_causal_flow_scripts_pack.tar.gz -C PARENT github_pack_s3_causal_flow
```

将生成的压缩包或直接 **本目录** 推到你自己的仓库即可（注意许可证：CosyVoice / AISHELL / 权重各自遵循原协议）。

## 许可证与致谢

- **CosyVoice**、**Matcha-TTS**、**FunAudioLLM/CosyVoice-300M** 权重：遵循其官方许可证；本包仅摘录自定义脚本副本。
- **AISHELL-1** 数据需单独申请/遵守数据协议。

---

*Paths in `scripts/host/*.sh` are examples—**务必全局替换为你的机器路径**后再运行。*
