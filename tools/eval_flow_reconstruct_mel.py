#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Self-prompt flow reconstruction: compare predicted target mel vs GT mel (same protocol as infer_flow_reconstruct_s3tok25hz).
Standalone repo vendoring; upstream CosyVoice (FunAudioLLM/CosyVoice)."""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
from io import StringIO
from pathlib import Path

import onnxruntime as ort
import torch

_REPO = Path(__file__).resolve().parents[1]


def _load_infer_helpers():
    path = _REPO / "tools" / "infer_flow_reconstruct_s3tok25hz.py"
    spec = importlib.util.spec_from_file_location("infer_flow_reconstruct_s3tok25hz", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def _cosine_sim(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.reshape(-1).float()
    b = b.reshape(-1).float()
    return float((a @ b) / (a.norm() * b.norm() + 1e-8))


def run_one(
    *,
    wav: str,
    preset: str,
    assets_dir: str,
    flow_ckpt: str,
    train_config: str,
    tokenizer_pt: str,
    prompt_ratio: float,
    prompt_tokens: int,
    n_timesteps: int,
    device: torch.device,
) -> dict:
    inf = _load_infer_helpers()
    inf._ensure_import_paths()
    from hyperpyyaml import load_hyperpyyaml

    repo = inf._repo_root()
    official_root = inf._official_cosyvoice1_dir(repo)

    if preset == "official_cosyvoice1_50hz":
        if not official_root.is_dir() or not (official_root / "flow.pt").is_file():
            raise SystemExit(f"missing official weights: {official_root}")
        tokenizer_pt = ""
        train_config = str(official_root / "cosyvoice.yaml")
        flow_ckpt = str(official_root / "flow.pt")
        assets_dir = str(official_root)
    else:
        if not tokenizer_pt.strip():
            import os

            env_tok = os.environ.get("COSYVOICE_S3_TOKENIZER_PT", "").strip()
            if env_tok:
                tokenizer_pt = env_tok
            elif inf._default_s3_tokenizer_pt(repo).is_file():
                tokenizer_pt = str(inf._default_s3_tokenizer_pt(repo))
            else:
                raise SystemExit("custom preset needs --tokenizer_pt or COSYVOICE_S3_TOKENIZER_PT")

    camp_path = Path(assets_dir) / "campplus.onnx"
    hift_path = Path(assets_dir) / "hift.pt"
    paths = [camp_path, hift_path, Path(flow_ckpt), Path(train_config), Path(wav)]
    if tokenizer_pt:
        paths.append(Path(tokenizer_pt))
    else:
        paths.append(Path(assets_dir) / "speech_tokenizer_v1.onnx")
    for p in paths:
        if not p.is_file():
            raise FileNotFoundError(str(p))

    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    opts.intra_op_num_threads = 1
    tok_sess = None
    if not tokenizer_pt:
        tok_sess = ort.InferenceSession(
            str(Path(assets_dir) / "speech_tokenizer_v1.onnx"),
            sess_options=opts,
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
            if device.type == "cuda"
            else ["CPUExecutionProvider"],
        )
    camp_sess = ort.InferenceSession(str(camp_path), sess_options=opts, providers=["CPUExecutionProvider"])

    with open(train_config, "r", encoding="utf-8") as f:
        cfg_text = f.read()
    overrides = {"llm": None}
    if "\nhifigan:" in cfg_text or cfg_text.lstrip().startswith("hifigan:"):
        overrides["hifigan"] = None
    configs = load_hyperpyyaml(StringIO(cfg_text), overrides=overrides)

    flow = configs["flow"]
    feat_extractor = configs["feat_extractor"]
    sample_rate = int(configs["sample_rate"])
    input_frame_rate = int(flow.input_frame_rate)

    flow_state = inf.strip_meta_state_dict(torch.load(flow_ckpt, map_location="cpu", weights_only=True))
    flow.load_state_dict(flow_state, strict=True)
    flow.to(device).eval()

    speech_22k = inf.load_wav_1ch(wav, sample_rate).to(device)
    speech_16k = inf.load_wav_1ch(wav, 16000).to(device)

    with torch.no_grad():
        mel = feat_extractor(speech_22k).squeeze(0).transpose(0, 1).contiguous()
        if tokenizer_pt:
            speech_token = inf.extract_speech_token_s3(speech_16k, tokenizer_pt, device)
        else:
            speech_token = inf.extract_speech_token(speech_16k, tok_sess, device)
        embedding = inf.extract_spk_embedding(speech_16k, camp_sess, device)

    n_tok = speech_token.shape[1]
    n_mel = mel.shape[0]
    t_align = inf.align_token_mel_lengths(n_tok, n_mel, input_frame_rate, sample_rate, hop_size=256)
    if t_align < 10:
        raise RuntimeError(f"aligned length too short: tokens={t_align}")
    m_align = int(t_align * sample_rate / (input_frame_rate * 256))
    m_align = min(m_align, n_mel)
    speech_token = speech_token[:, :t_align]
    mel = mel[:m_align]

    if prompt_tokens > 0:
        n_prompt = min(prompt_tokens, t_align - 1)
    else:
        n_prompt = max(1, int(t_align * prompt_ratio))
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
            n_timesteps=n_timesteps,
        )

    # tts_mel: (1, 80, T_pred)
    t_pred = int(tts_mel.shape[2])
    t_avail = int(mel.shape[0] - mel_len1)
    t_cmp = min(t_pred, t_avail)
    if t_cmp <= 0:
        raise RuntimeError(f"no overlap for mel compare: t_pred={t_pred} t_avail={t_avail} mel_len1={mel_len1}")

    pred = tts_mel[0, :, :t_cmp].transpose(0, 1).float().cpu()
    ref = mel[mel_len1 : mel_len1 + t_cmp, :].float().cpu()
    diff = pred - ref
    mae = float(diff.abs().mean())
    mse = float((diff**2).mean())
    rmse = float(math.sqrt(mse))
    cos_global = _cosine_sim(pred, ref)

    # Mean cosine similarity per time frame (mel bands as vector)
    pn = torch.nn.functional.normalize(pred, dim=1)
    rn = torch.nn.functional.normalize(ref, dim=1)
    cos_per_frame = float((pn * rn).sum(dim=1).mean())

    return {
        "wav": str(Path(wav).resolve()),
        "preset": preset,
        "flow_ckpt": flow_ckpt,
        "train_config": train_config,
        "tokenizer_pt": tokenizer_pt or "speech_tokenizer_v1.onnx",
        "assets_dir": assets_dir,
        "sample_rate": sample_rate,
        "input_frame_rate": input_frame_rate,
        "t_align": t_align,
        "m_align": m_align,
        "n_prompt_tokens": n_prompt,
        "n_target_tokens": n_target,
        "mel_len1_prompt": mel_len1,
        "mel_target_pred_frames": t_pred,
        "mel_target_cmp_frames": t_cmp,
        "mel_truncated_frames": t_pred - t_cmp if t_pred > t_cmp else 0,
        "mae": mae,
        "mse": mse,
        "rmse": rmse,
        "cosine_global": cos_global,
        "cosine_mean_per_frame": cos_per_frame,
        "prompt_ratio_used": prompt_ratio if prompt_tokens <= 0 else None,
        "prompt_tokens_arg": prompt_tokens,
        "n_timesteps": n_timesteps,
    }


def main():
    inf = _load_infer_helpers()
    p = argparse.ArgumentParser()
    p.add_argument("--wav", required=True)
    p.add_argument("--preset", choices=["official_cosyvoice1_50hz", "custom_s3tok25hz"], required=True)
    p.add_argument("--assets_dir", default="")
    p.add_argument(
        "--torch_ddp_dir",
        default="",
        help="custom preset: pick latest epoch_*_whole.pt here (overrides --flow_ckpt)",
    )
    p.add_argument("--flow_ckpt", default="")
    p.add_argument("--train_config", default="")
    p.add_argument("--tokenizer_pt", default="")
    p.add_argument("--prompt_ratio", type=float, default=0.35)
    p.add_argument("--prompt_tokens", type=int, default=-1)
    p.add_argument("--n_timesteps", type=int, default=10)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--json_out", default="", help="append one JSON line")
    args = p.parse_args()

    repo = _REPO
    official_root = repo / "pretrained_weights" / "CosyVoice-300M"
    default_assets = str(official_root) if (official_root / "campplus.onnx").is_file() else ""
    assets_dir = args.assets_dir or default_assets
    if not assets_dir:
        raise SystemExit(
            "set --assets_dir or download campplus.onnx+hift.pt into pretrained_weights/CosyVoice-300M/ (see README)"
        )

    default_cfg = repo / "conf" / "cosyvoice_aishell_s3tok1024_25hz.yaml"
    default_torch_ddp = repo / "pretrained_weights" / "flow_torch_ddp"
    _auto = inf._latest_epoch_whole_pt(default_torch_ddp)
    default_ckpt = str(_auto) if _auto is not None else str(default_torch_ddp / "epoch_0_whole.pt")

    if args.preset == "custom_s3tok25hz" and args.torch_ddp_dir.strip():
        pick = inf._latest_epoch_whole_pt(Path(args.torch_ddp_dir.strip()).expanduser().resolve())
        if pick is None:
            raise SystemExit(f"--torch_ddp_dir has no epoch_*_whole.pt: {args.torch_ddp_dir}")
        flow_ckpt = str(pick)
    else:
        flow_ckpt = args.flow_ckpt or default_ckpt
    train_config = args.train_config or str(default_cfg)

    device = torch.device(args.device)
    out = run_one(
        wav=args.wav,
        preset=args.preset,
        assets_dir=assets_dir,
        flow_ckpt=flow_ckpt,
        train_config=train_config,
        tokenizer_pt=args.tokenizer_pt,
        prompt_ratio=args.prompt_ratio,
        prompt_tokens=args.prompt_tokens,
        n_timesteps=args.n_timesteps,
        device=device,
    )
    line = json.dumps(out, ensure_ascii=False)
    print(line)
    if args.json_out:
        Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.json_out, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
