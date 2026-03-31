#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
同一段 wav 上自提示（prompt token + prompt mel + spk emb 均来自该 wav）的 Flow 重建试听脚本。

================================================================================
【务必对照：Flow 权重 与 Speech Token 必须同一套语义，不要混用】
================================================================================

A) 官方 CosyVoice-1（FunAudioLLM/CosyVoice-300M，仓库默认 50Hz / vocab 4096）
   - flow 权重：与官方目录中的 flow.pt 一致（本仓库默认
     pretrained_weights/CosyVoice-300M/flow.pt）
   - tokenizer：必须用同目录下的官方 speech_tokenizer_v1.onnx（Whisper-logmel → ONNX）
   - 结构描述 yaml：必须用同目录 cosyvoice.yaml（与 flow.pt 成套）
   - campplus / hift：与官方目录成套即可（本脚本用 assets_dir 下的 campplus.onnx + hift.pt）
   - 启动方式：加 --preset official_cosyvoice1_50hz（不要传 --tokenizer_pt）

B) 你的训练成果（与官方不同：1024 码本 / 25Hz + S3 tokenizer，不可与 A 混用）
   - flow 权重：torch_ddp 目录下 epoch_*_whole.pt（可用 --torch_ddp_dir 自动选最新）
   - tokenizer：与 Flow 训练时抽 token 所用为同一 S3 导出（--tokenizer_pt 或 COSYVOICE_S3_TOKENIZER_PT）
   - 便捷入口：tools/infer_flow_reconstruct_causal_s3tok25hz.py（默认 pretrained_weights/s3tokenizer.pt
     + exp/...officialinit_causaldata.../torch_ddp）
   - 结构 yaml：必须与训练 config 一致（本仓库 conf/cosyvoice_aishell_s3tok1024_25hz.yaml）
   - campplus / hift：仍用 CosyVoice-300M 目录即可（与 token 无关）
   - 默认 preset 为 custom_s3tok25hz：不传 --preset 即走这条链路；官方对比时显式加
     --preset official_cosyvoice1_50hz

【官方权重下载示例】（下载到本仓库 pretrained_weights/CosyVoice-300M/）
  HF_HUB_ENABLE_HF_TRANSFER=0 huggingface-cli download FunAudioLLM/CosyVoice-300M \\
    --local-dir pretrained_weights/CosyVoice-300M

依赖：CosyVoice-main 与 third_party/Matcha-TTS（脚本会自动把仓库根、Matcha、以及「CosyVoice-main
的上一级目录」加入 sys.path，若该目录下存在 s3tokenizer_train 包则无需再 export PYTHONPATH）。
也可显式设置 COSYVOICE_S3TOKENIZER_ROOT 指向包含 s3tokenizer_train 的目录。另需 onnxruntime、torch、
torchaudio、whisper、hyperpyyaml。

示例（官方，验证脚本本身）：
  export PYTHONPATH=third_party/Matcha-TTS:$(pwd)
  python tools/infer_flow_reconstruct_s3tok25hz.py --preset official_cosyvoice1_50hz \\
    --wav sample.wav --out_wav /tmp/official_flow.wav

示例（你的 Flow + S3 tokenizer，默认自动选 torch_ddp 下最新 epoch）：
  export PYTHONPATH=$(pwd):$(pwd)/third_party/Matcha-TTS
  python tools/infer_flow_reconstruct_s3tok25hz.py \\
    --wav sample.wav --out_wav /tmp/my_flow.wav \\
    --train_config examples/libritts/cosyvoice/conf/cosyvoice_aishell_s3tok1024_25hz.yaml \\
    --tokenizer_pt /path/to/s3tokenizer_epoch10.pt

  # 或指定 torch_ddp 目录（自动选最大 epoch_*_whole.pt）：
  python tools/infer_flow_reconstruct_s3tok25hz.py --wav sample.wav --out_wav /tmp/x.wav \\
    --torch_ddp_dir exp/cosyvoice1_flow_.../flow/torch_ddp
"""
# -----------------------------------------------------------------------------
# Standalone repo note: this file is vendored for zero-setup inference.
# Upstream CosyVoice: https://github.com/FunAudioLLM/CosyVoice
# Edits in this fork: portable defaults (conf/, pretrained_weights/), s3tokenizer in-repo.
# -----------------------------------------------------------------------------
from __future__ import annotations

import argparse
import os
import re
import sys
from io import StringIO
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
import torchaudio
import whisper


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _official_cosyvoice1_dir(repo: Path) -> Path:
    return repo / "pretrained_weights" / "CosyVoice-300M"


# 自训 S3 tokenizer 必须显式指定：
# - --tokenizer_pt /path/to/exported_s3_tokenizer.pt
# 或
# - 环境变量 COSYVOICE_S3_TOKENIZER_PT


def _latest_epoch_whole_pt(torch_ddp_dir: Path) -> Path | None:
    """Pick epoch_N_whole.pt with largest N under torch_ddp_dir."""
    if not torch_ddp_dir.is_dir():
        return None
    best: tuple[int, Path] | None = None
    for p in torch_ddp_dir.glob("epoch_*_whole.pt"):
        m = re.match(r"epoch_(\d+)_whole\.pt$", p.name)
        if not m:
            continue
        n = int(m.group(1))
        if best is None or n > best[0]:
            best = (n, p)
    return best[1] if best else None


def _ensure_import_paths():
    root = _repo_root()
    rs = str(root)
    if rs not in sys.path:
        sys.path.insert(0, rs)
    matcha = root / "third_party" / "Matcha-TTS"
    if matcha.is_dir() and (matcha / "matcha").is_dir():
        ms = str(matcha)
        if ms not in sys.path:
            sys.path.insert(0, ms)
    # s3tokenizer_train: vendored at <repo>/s3tokenizer_train/ or sibling of legacy CosyVoice layout.
    _s3_roots = []
    _env_root = os.environ.get("COSYVOICE_S3TOKENIZER_ROOT", "").strip()
    if _env_root:
        _s3_roots.append(Path(_env_root))
    _s3_roots.append(root)
    _s3_roots.append(root.parent)
    for _p in _s3_roots:
        if _p.is_dir() and (_p / "s3tokenizer_train").is_dir():
            _es = str(_p.resolve())
            if _es not in sys.path:
                sys.path.insert(0, _es)
            break


def load_wav_1ch(path: str, target_sr: int) -> torch.Tensor:
    import soundfile as sf
    data, sr = sf.read(path, dtype="float32")
    speech = torch.from_numpy(data)
    if speech.ndim == 1:
        speech = speech.unsqueeze(0)
    else:
        speech = speech.T.mean(dim=0, keepdim=True)
    if sr != target_sr:
        speech = torchaudio.transforms.Resample(sr, target_sr)(speech)
    return speech


def extract_speech_token(
    wav_16k: torch.Tensor, session: ort.InferenceSession, device: torch.device
) -> torch.Tensor:
    feat = whisper.log_mel_spectrogram(wav_16k.squeeze(0), n_mels=128)
    # ONNX speech_tokenizer expects rank-3 feats (batch, n_mels, time), same as CosyVoiceFrontEnd.
    feat_b = feat.unsqueeze(0).detach().cpu().numpy()
    t = feat_b.shape[2]
    speech_token = session.run(
        None,
        {
            session.get_inputs()[0].name: feat_b,
            session.get_inputs()[1].name: np.array([t], dtype=np.int32),
        },
    )[0].flatten()
    tok = np.asarray(speech_token, dtype=np.int32)
    return torch.tensor(tok, dtype=torch.int32, device=device).unsqueeze(0)


def extract_speech_token_s3(
    wav_16k: torch.Tensor, tokenizer_pt: str, device: torch.device
) -> torch.Tensor:
    from s3tokenizer_train.export import S3Config, S3TokenizerV1

    ckpt = torch.load(tokenizer_pt, map_location="cpu", weights_only=True)
    cfg = ckpt["config"]
    model = S3TokenizerV1(S3Config(**cfg))
    model.load_state_dict(ckpt["model"], strict=True)
    model = model.to(device).eval()

    mel = whisper.log_mel_spectrogram(wav_16k, n_mels=128).to(device)
    if mel.ndim == 2:
        mel = mel.unsqueeze(0)
    elif mel.ndim != 3:
        raise ValueError(f"unexpected mel shape for s3 tokenizer: {tuple(mel.shape)}")
    with torch.no_grad():
        tok = model.tokenize(mel)[0]
    return tok.to(torch.int32).unsqueeze(0)


def extract_spk_embedding(wav_16k: torch.Tensor, session: ort.InferenceSession, device: torch.device) -> torch.Tensor:
    import torchaudio.compliance.kaldi as kaldi

    feat = kaldi.fbank(wav_16k, num_mel_bins=80, dither=0, sample_frequency=16000)
    feat = feat - feat.mean(dim=0, keepdim=True)
    emb = session.run(
        None,
        {session.get_inputs()[0].name: feat.unsqueeze(0).cpu().numpy()},
    )[0]
    return torch.tensor(emb, device=device, dtype=torch.float32)


def align_token_mel_lengths(
    n_token: int, n_mel: int, input_frame_rate: int, hop_audio_sr: int, hop_size: int
) -> int:
    """Max token count T so that int(T * sr / (ifr * hop)) <= n_mel (training-time alignment)."""
    max_tok = int(n_mel * input_frame_rate * hop_size / hop_audio_sr)
    return min(n_token, max_tok)


def strip_meta_state_dict(obj):
    if isinstance(obj, dict) and any(k in obj for k in ("epoch", "step")):
        return {k: v for k, v in obj.items() if k not in ("epoch", "step")}
    return obj


def main():
    _ensure_import_paths()

    try:
        from hyperpyyaml import load_hyperpyyaml
        from cosyvoice.hifigan.generator import HiFTGenerator
    except ModuleNotFoundError as e:
        raise SystemExit(
            f"{e}\nInstall CosyVoice deps and Matcha-TTS (see CosyVoice-main/FAQ.md), e.g.:\n"
            "  export PYTHONPATH=third_party/Matcha-TTS:$(pwd)\n"
            "  git submodule update --init --recursive  # if third_party/Matcha-TTS is empty\n"
        ) from e

    repo = _repo_root()
    # 默认：conf 在本仓库；Flow ckpt 放在 pretrained_weights/flow_torch_ddp/epoch_*_whole.pt
    default_cfg = repo / "conf" / "cosyvoice_aishell_s3tok1024_25hz.yaml"
    default_torch_ddp = repo / "pretrained_weights" / "flow_torch_ddp"
    _auto_ckpt = _latest_epoch_whole_pt(default_torch_ddp)
    default_ckpt = (
        _auto_ckpt
        if _auto_ckpt is not None
        else (default_torch_ddp / "epoch_0_whole.pt")
    )
    official_root = _official_cosyvoice1_dir(repo)
    default_assets = str(official_root) if (official_root / "campplus.onnx").is_file() else ""

    p = argparse.ArgumentParser(description="Flow self-prompt reconstruction (same wav)")
    p.add_argument(
        "--preset",
        choices=["custom_s3tok25hz", "official_cosyvoice1_50hz"],
        default="custom_s3tok25hz",
        help="custom=自训1024/25Hz+S3 tokenizer；official=官方300M 50Hz/4096+speech_tokenizer_v1.onnx",
    )
    p.add_argument("--wav", required=True, help="Input wav (any rate; resampled internally)")
    p.add_argument(
        "--torch_ddp_dir",
        type=str,
        default="",
        help="custom preset only: directory with epoch_*_whole.pt; if set, use latest (overrides --flow_ckpt)",
    )
    p.add_argument(
        "--flow_ckpt",
        type=str,
        default=None,
        help="custom preset: explicit checkpoint; default = latest under default experiment torch_ddp_dir",
    )
    p.add_argument(
        "--train_config",
        type=str,
        default=str(default_cfg),
        help="Hyperpyyaml used for flow training (same architecture as checkpoint)",
    )
    p.add_argument(
        "--assets_dir",
        type=str,
        default=default_assets,
        help="Dir with campplus.onnx and hift.pt (and speech_tokenizer_v1.onnx if not using --tokenizer_pt). "
        "Default: pretrained_weights/CosyVoice-300M/ if campplus.onnx exists.",
    )
    p.add_argument(
        "--tokenizer_pt",
        type=str,
        default="",
        help="Path to custom S3 tokenizer .pt used in training; when set, do NOT use speech_tokenizer_v1.onnx",
    )
    p.add_argument("--out_wav", type=str, default="flow_reconstruct_out.wav", help="Output 22050 Hz wav")
    p.add_argument(
        "--prompt_text",
        type=str,
        default="这是一段用于flow重建评估的提示文本。",
        help="Non-empty prompt text placeholder for zero-shot protocol consistency",
    )
    p.add_argument(
        "--prompt_ratio",
        type=float,
        default=0.35,
        help="Fraction of aligned tokens used as prompt prefix (rest is reconstructed)",
    )
    p.add_argument(
        "--prompt_tokens",
        type=int,
        default=-1,
        help="If >0, fixed number of prompt tokens (overrides prompt_ratio)",
    )
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--n_timesteps", type=int, default=10, help="Flow sampling steps for decoder")
    args = p.parse_args()
    if not args.prompt_text.strip():
        raise ValueError("--prompt_text cannot be empty")

    if args.preset == "official_cosyvoice1_50hz":
        if not official_root.is_dir() or not (official_root / "flow.pt").is_file():
            raise SystemExit(
                f"缺少官方权重目录或 flow.pt：{official_root}\n"
                "请按脚本顶部注释下载：huggingface-cli download FunAudioLLM/CosyVoice-300M "
                f"--local-dir {official_root}"
            )
        args.tokenizer_pt = ""
        args.train_config = str(official_root / "cosyvoice.yaml")
        args.flow_ckpt = str(official_root / "flow.pt")
        args.assets_dir = str(official_root)
        print(f"[preset] official_cosyvoice1_50hz: yaml+flow+onnx tokenizer from {official_root}", flush=True)
    else:
        tdd = args.torch_ddp_dir.strip()
        if tdd:
            pick = _latest_epoch_whole_pt(Path(tdd).expanduser().resolve())
            if pick is None:
                raise SystemExit(f"--torch_ddp_dir has no epoch_*_whole.pt: {tdd}")
            args.flow_ckpt = str(pick)
            print(f"[custom] flow_ckpt from --torch_ddp_dir (latest): {args.flow_ckpt}", flush=True)
        elif args.flow_ckpt is None:
            args.flow_ckpt = str(default_ckpt)
            print(f"[custom] flow_ckpt (default latest): {args.flow_ckpt}", flush=True)
        if args.tokenizer_pt.strip():
            pass
        else:
            env_tok = os.environ.get("COSYVOICE_S3_TOKENIZER_PT", "").strip()
            if env_tok:
                args.tokenizer_pt = env_tok
            else:
                raise SystemExit(
                    "自训 preset 必须提供 S3 tokenizer：--tokenizer_pt=... 或设置环境变量 "
                    "COSYVOICE_S3_TOKENIZER_PT（见脚本顶部说明）。"
                )
        print(f"[preset] custom_s3tok25hz: tokenizer_pt={args.tokenizer_pt}", flush=True)

    device = torch.device(args.device)
    assets = args.assets_dir
    camp_path = os.path.join(assets, "campplus.onnx")
    hift_path = os.path.join(assets, "hift.pt")
    required_paths = [camp_path, hift_path, args.flow_ckpt, args.train_config, args.wav]
    if args.tokenizer_pt:
        required_paths.append(args.tokenizer_pt)
    else:
        required_paths.append(os.path.join(assets, "speech_tokenizer_v1.onnx"))
    for path in required_paths:
        if not os.path.isfile(path):
            raise FileNotFoundError(path)

    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    opts.intra_op_num_threads = 1
    tok_sess = None
    if not args.tokenizer_pt:
        tok_path = os.path.join(assets, "speech_tokenizer_v1.onnx")
        tok_sess = ort.InferenceSession(
            tok_path,
            sess_options=opts,
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
            if device.type == "cuda"
            else ["CPUExecutionProvider"],
        )
    camp_sess = ort.InferenceSession(camp_path, sess_options=opts, providers=["CPUExecutionProvider"])

    with open(args.train_config, "r", encoding="utf-8") as f:
        cfg_text = f.read()
    overrides = {"llm": None}
    if "\nhifigan:" in cfg_text or cfg_text.lstrip().startswith("hifigan:"):
        overrides["hifigan"] = None
    configs = load_hyperpyyaml(StringIO(cfg_text), overrides=overrides)

    flow = configs["flow"]
    feat_extractor = configs["feat_extractor"]
    sample_rate = int(configs["sample_rate"])
    input_frame_rate = int(flow.input_frame_rate)

    flow_state = strip_meta_state_dict(torch.load(args.flow_ckpt, map_location="cpu", weights_only=True))
    flow.load_state_dict(flow_state, strict=True)
    flow.to(device).eval()

    hift: HiFTGenerator = configs["hift"]
    hift_state = torch.load(hift_path, map_location="cpu", weights_only=True)
    if any(k.startswith("generator.") for k in hift_state):
        hift_state = {k.replace("generator.", ""): v for k, v in hift_state.items()}
    hift.load_state_dict(hift_state, strict=True)
    hift.to(device).eval()

    speech_22k = load_wav_1ch(args.wav, sample_rate).to(device)
    speech_16k = load_wav_1ch(args.wav, 16000).to(device)

    with torch.no_grad():
        mel = feat_extractor(speech_22k).squeeze(0).transpose(0, 1).contiguous()
        if args.tokenizer_pt:
            speech_token = extract_speech_token_s3(speech_16k, args.tokenizer_pt, device)
        else:
            speech_token = extract_speech_token(speech_16k, tok_sess, device)
        embedding = extract_spk_embedding(speech_16k, camp_sess, device)

    n_tok = speech_token.shape[1]
    n_mel = mel.shape[0]
    t_align = align_token_mel_lengths(n_tok, n_mel, input_frame_rate, sample_rate, hop_size=256)
    if t_align < 10:
        raise RuntimeError(f"Aligned length too short (tokens={t_align}). Try a longer wav.")
    m_align = int(t_align * sample_rate / (input_frame_rate * 256))
    m_align = min(m_align, n_mel)
    speech_token = speech_token[:, :t_align]
    mel = mel[:m_align]

    if args.prompt_tokens > 0:
        n_prompt = min(args.prompt_tokens, t_align - 1)
    else:
        n_prompt = max(1, int(t_align * args.prompt_ratio))
    n_prompt = min(n_prompt, t_align - 1)
    n_target = t_align - n_prompt

    mel_len1 = int(n_prompt * sample_rate / (input_frame_rate * 256))
    mel_len1 = min(mel_len1, mel.shape[0])
    prompt_feat = mel[:mel_len1].unsqueeze(0).to(device)
    prompt_token = speech_token[:, :n_prompt]
    target_token = speech_token[:, n_prompt : n_prompt + n_target]

    flow_cache = torch.zeros(1, 80, 0, 2, device=device, dtype=prompt_feat.dtype)

    with torch.inference_mode():
        tts_mel, _ = flow.inference(
            token=target_token.to(torch.int32),
            token_len=torch.tensor([target_token.shape[1]], dtype=torch.int32, device=device),
            prompt_token=prompt_token.to(torch.int32),
            prompt_token_len=torch.tensor([prompt_token.shape[1]], dtype=torch.int32, device=device),
            prompt_feat=prompt_feat,
            prompt_feat_len=torch.tensor([prompt_feat.shape[1]], dtype=torch.int32, device=device),
            embedding=embedding,
            flow_cache=flow_cache,
            n_timesteps=args.n_timesteps,
        )
        tts_speech, _ = hift.inference(
            speech_feat=tts_mel.to(torch.float32),
            cache_source=torch.zeros(1, 1, 0, device=device, dtype=torch.float32),
        )

    out = tts_speech.squeeze(0).detach().cpu()
    import soundfile as sf
    sf.write(args.out_wav, out.numpy(), sample_rate)
    dur = out.numel() / sample_rate
    print(
        f"Wrote {args.out_wav} ({dur:.2f}s). "
        f"prompt_tokens={n_prompt} target_tokens={n_target} "
        f"aligned_tokens={t_align} mel_frames_used={m_align} "
        f"prompt_text_len={len(args.prompt_text.strip())} "
        f"n_timesteps={args.n_timesteps}"
    )


if __name__ == "__main__":
    main()
