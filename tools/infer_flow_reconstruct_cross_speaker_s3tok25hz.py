#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
跨说话人 Flow 自提示重建：内容与官方「prompt 管说话人」思路对齐的一种最小变体。

- **content_wav**：抽取 speech token + 22050 mel（与 infer_flow_reconstruct_s3tok25hz.py 相同协议：
  prompt 前缀 token / prompt 前缀 mel / 后续 target token 均来自该条音频）。
- **speaker_wav**：**仅**抽取 CAMPPlus speaker embedding（16k 整段），与 content 非同源时可听「声纹替换」
  后模型是否仍跟住内容 token。

原 CosyVoice CLI 的 zero-shot 还会用 prompt 文本等；本脚本只做 flow 段试听，不涉及 LLM。

用法与基脚本相同的 preset / ckpt / yaml / tokenizer 规则，见 infer_flow_reconstruct_s3tok25hz.py 顶部说明。

示例（自训 Flow + S3 tokenizer）：
  cd CosyVoice-main && export PYTHONPATH=/mnt/data/ft_cosy:third_party/Matcha-TTS:$(pwd)
  marco/bin/python tools/infer_flow_reconstruct_cross_speaker_s3tok25hz.py \\
    --content_wav data/aishell_raw/wav/dev/S0736/BAC009S0736W0371.wav \\
    --speaker_wav path/to/other_speaker.wav \\
    --out_wav /tmp/cross_spk.wav --n_timesteps 20
"""
# Standalone repo: vendored next to infer_flow_reconstruct_s3tok25hz.py.
# Upstream: CosyVoice project (FunAudioLLM/CosyVoice).
from __future__ import annotations

import argparse
import importlib.util
import os
import sys
from io import StringIO
from pathlib import Path

import onnxruntime as ort
import torch


def _load_base():
    here = Path(__file__).resolve().parent
    path = here / "infer_flow_reconstruct_s3tok25hz.py"
    if not path.is_file():
        root = here.parents[1] if len(here.parents) > 1 else here.parent
        path = root / "tools" / "infer_flow_reconstruct_s3tok25hz.py"
    if not path.is_file():
        raise FileNotFoundError(
            "infer_flow_reconstruct_s3tok25hz.py not found next to this script or under <repo>/tools/"
        )
    spec = importlib.util.spec_from_file_location("infer_flow_reconstruct_s3tok25hz", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def main() -> None:
    inf = _load_base()
    inf._ensure_import_paths()

    try:
        from hyperpyyaml import load_hyperpyyaml
        from cosyvoice.hifigan.generator import HiFTGenerator
    except ModuleNotFoundError as e:
        raise SystemExit(
            f"{e}\nInstall CosyVoice deps and Matcha-TTS (see CosyVoice-main/FAQ.md).\n"
            "  export PYTHONPATH=third_party/Matcha-TTS:$(pwd)\n"
        ) from e

    repo = inf._repo_root()
    default_cfg = repo / "conf" / "cosyvoice_aishell_s3tok1024_25hz.yaml"
    default_torch_ddp = repo / "pretrained_weights" / "flow_torch_ddp"
    _auto_ckpt = inf._latest_epoch_whole_pt(default_torch_ddp)
    default_ckpt = _auto_ckpt if _auto_ckpt is not None else (default_torch_ddp / "epoch_0_whole.pt")
    official_root = inf._official_cosyvoice1_dir(repo)
    default_assets = str(official_root) if (official_root / "campplus.onnx").is_file() else ""

    p = argparse.ArgumentParser(
        description="Cross-speaker flow recon: tokens+mel from content_wav; spk emb from speaker_wav"
    )
    p.add_argument(
        "--preset",
        choices=["custom_s3tok25hz", "official_cosyvoice1_50hz"],
        default="custom_s3tok25hz",
    )
    p.add_argument("--content_wav", required=True, help="Wav for speech tokens + mel (self-prompt prefix/target)")
    p.add_argument("--speaker_wav", required=True, help="Wav for CAMPPlus embedding only (can be another speaker)")
    p.add_argument("--torch_ddp_dir", type=str, default="")
    p.add_argument("--flow_ckpt", type=str, default=None)
    p.add_argument("--train_config", type=str, default=str(default_cfg))
    p.add_argument("--assets_dir", type=str, default=default_assets)
    p.add_argument("--tokenizer_pt", type=str, default="")
    p.add_argument("--out_wav", type=str, required=True)
    p.add_argument("--prompt_text", type=str, default="这是一段用于flow重建评估的提示文本。")
    p.add_argument("--prompt_ratio", type=float, default=0.35)
    p.add_argument("--prompt_tokens", type=int, default=-1)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--n_timesteps", type=int, default=10)
    args = p.parse_args()
    if not args.prompt_text.strip():
        raise ValueError("--prompt_text cannot be empty")

    if args.preset == "official_cosyvoice1_50hz":
        if not official_root.is_dir() or not (official_root / "flow.pt").is_file():
            raise SystemExit(f"missing official weights: {official_root}")
        args.tokenizer_pt = ""
        args.train_config = str(official_root / "cosyvoice.yaml")
        args.flow_ckpt = str(official_root / "flow.pt")
        args.assets_dir = str(official_root)
        print(f"[preset] official_cosyvoice1_50hz from {official_root}", flush=True)
    else:
        tdd = args.torch_ddp_dir.strip()
        if tdd:
            pick = inf._latest_epoch_whole_pt(Path(tdd).expanduser().resolve())
            if pick is None:
                raise SystemExit(f"--torch_ddp_dir has no epoch_*_whole.pt: {tdd}")
            args.flow_ckpt = str(pick)
            print(f"[custom] flow_ckpt from --torch_ddp_dir: {args.flow_ckpt}", flush=True)
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
                raise SystemExit("custom preset needs --tokenizer_pt or COSYVOICE_S3_TOKENIZER_PT")
        print(f"[custom] tokenizer_pt={args.tokenizer_pt}", flush=True)

    device = torch.device(args.device)
    assets = args.assets_dir
    camp_path = os.path.join(assets, "campplus.onnx")
    hift_path = os.path.join(assets, "hift.pt")
    paths = [camp_path, hift_path, args.flow_ckpt, args.train_config, args.content_wav, args.speaker_wav]
    if args.tokenizer_pt:
        paths.append(args.tokenizer_pt)
    else:
        paths.append(os.path.join(assets, "speech_tokenizer_v1.onnx"))
    for path in paths:
        if not os.path.isfile(path):
            raise FileNotFoundError(path)

    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    opts.intra_op_num_threads = 1
    tok_sess = None
    if not args.tokenizer_pt:
        tok_sess = ort.InferenceSession(
            os.path.join(assets, "speech_tokenizer_v1.onnx"),
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

    flow_state = inf.strip_meta_state_dict(torch.load(args.flow_ckpt, map_location="cpu", weights_only=True))
    flow.load_state_dict(flow_state, strict=True)
    flow.to(device).eval()

    hift: HiFTGenerator = configs["hift"]
    hift_state = torch.load(hift_path, map_location="cpu", weights_only=True)
    if any(k.startswith("generator.") for k in hift_state):
        hift_state = {k.replace("generator.", ""): v for k, v in hift_state.items()}
    hift.load_state_dict(hift_state, strict=True)
    hift.to(device).eval()

    # Content branch: same as single-wav script
    speech_22k = inf.load_wav_1ch(args.content_wav, sample_rate).to(device)
    speech_16k_content = inf.load_wav_1ch(args.content_wav, 16000).to(device)

    # Speaker branch: separate file
    speech_16k_spk = inf.load_wav_1ch(args.speaker_wav, 16000).to(device)

    with torch.no_grad():
        mel = feat_extractor(speech_22k).squeeze(0).transpose(0, 1).contiguous()
        if args.tokenizer_pt:
            speech_token = inf.extract_speech_token_s3(speech_16k_content, args.tokenizer_pt, device)
        else:
            speech_token = inf.extract_speech_token(speech_16k_content, tok_sess, device)
        embedding = inf.extract_spk_embedding(speech_16k_spk, camp_sess, device)

    n_tok = speech_token.shape[1]
    n_mel = mel.shape[0]
    t_align = inf.align_token_mel_lengths(n_tok, n_mel, input_frame_rate, sample_rate, hop_size=256)
    if t_align < 10:
        raise RuntimeError(f"Aligned length too short (tokens={t_align}). Try a longer content wav.")
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

    import torchaudio

    out = tts_speech.squeeze(0).detach().cpu()
    torchaudio.save(args.out_wav, out.unsqueeze(0), sample_rate)
    dur = out.numel() / sample_rate
    print(
        f"Wrote {args.out_wav} ({dur:.2f}s). cross_speaker: "
        f"content={args.content_wav} speaker_emb={args.speaker_wav} "
        f"prompt_tokens={n_prompt} target_tokens={n_target} n_timesteps={args.n_timesteps}"
    )


if __name__ == "__main__":
    main()
