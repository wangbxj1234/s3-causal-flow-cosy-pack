#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Streaming inference where tokenizer AND flow run on the same chunk boundary.

--chunk_ms controls BOTH the tokenizer input granularity and the flow
processing granularity: every time the tokenizer emits new tokens from an
audio chunk, flow.inference runs immediately for those tokens.

Pipeline per chunk:
  audio chunk → whisper mel → tokenizer.stream_step → N new tokens
  → prepend overlap tokens from previous chunk
  → flow.inference (with flow_cache for continuity)
  → mel overlap fade at boundary
  → hift vocoder (with source cache)
  → output audio chunk

The overlap tokens (--token_overlap) are re-processed at chunk boundaries
and faded in mel space to avoid audible glitches, following the CosyVoice
streaming protocol.

Supports self-prompt (single wav) and cross-speaker (--speaker_wav).

Example:
  python tools/infer_flow_streaming_s3tok25hz.py \\
    --wav sample.wav --out_wav /tmp/stream.wav --chunk_ms 640

  python tools/infer_flow_streaming_s3tok25hz.py \\
    --wav content.wav --speaker_wav spk.wav --out_wav /tmp/cross.wav \\
    --chunk_ms 200
"""
from __future__ import annotations

import argparse
import os
import sys
from io import StringIO
from pathlib import Path
from typing import Optional

import numpy as np
import onnxruntime as ort
import torch
import torchaudio
import whisper


# ---------------------------------------------------------------------------
# Repo / import helpers (same as other tools/ scripts)
# ---------------------------------------------------------------------------

def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _ensure_import_paths():
    root = _repo_root()
    for p in [root, root / "third_party" / "Matcha-TTS"]:
        s = str(p)
        if p.is_dir() and s not in sys.path:
            sys.path.insert(0, s)
    env_root = os.environ.get("COSYVOICE_S3TOKENIZER_ROOT", "").strip()
    for p in ([Path(env_root)] if env_root else []) + [root, root.parent]:
        if p.is_dir() and (p / "s3tokenizer_train").is_dir():
            s = str(p.resolve())
            if s not in sys.path:
                sys.path.insert(0, s)
            break


# ---------------------------------------------------------------------------
# Audio I/O
# ---------------------------------------------------------------------------

def load_wav_1ch(path: str, target_sr: int) -> torch.Tensor:
    speech, sr = torchaudio.load(path, backend="soundfile")
    speech = speech.mean(dim=0, keepdim=True)
    if sr != target_sr:
        speech = torchaudio.transforms.Resample(sr, target_sr)(speech)
    return speech


# ---------------------------------------------------------------------------
# Extraction helpers
# ---------------------------------------------------------------------------

def load_s3_tokenizer(tokenizer_pt: str, device: torch.device):
    from s3tokenizer_train.export import S3Config, S3TokenizerV1
    ckpt = torch.load(tokenizer_pt, map_location="cpu", weights_only=True)
    cfg = ckpt["config"]
    model = S3TokenizerV1(S3Config(**cfg))
    model.load_state_dict(ckpt["model"], strict=True)
    return model.to(device).eval()


def extract_spk_embedding(
    wav_16k: torch.Tensor, session: ort.InferenceSession, device: torch.device
) -> torch.Tensor:
    import torchaudio.compliance.kaldi as kaldi
    feat = kaldi.fbank(wav_16k, num_mel_bins=80, dither=0, sample_frequency=16000)
    feat = feat - feat.mean(dim=0, keepdim=True)
    emb = session.run(
        None, {session.get_inputs()[0].name: feat.unsqueeze(0).cpu().numpy()}
    )[0]
    return torch.tensor(emb, device=device, dtype=torch.float32)


def align_token_mel_lengths(
    n_token: int, n_mel: int, input_frame_rate: int, sr: int = 22050, hop: int = 256
) -> int:
    max_tok = int(n_mel * input_frame_rate * hop / sr)
    return min(n_token, max_tok)


def strip_meta_state_dict(obj):
    if isinstance(obj, dict) and any(k in obj for k in ("epoch", "step")):
        return {k: v for k, v in obj.items() if k not in ("epoch", "step")}
    return obj


# ---------------------------------------------------------------------------
# Streaming state
# ---------------------------------------------------------------------------

class StreamState:
    """Holds all mutable state carried across streaming chunks."""

    def __init__(self, device: torch.device, mel_overlap_len: int, mel_cache_len: int = 10):
        self.flow_cache = torch.zeros(1, 80, 0, 2, device=device)
        self.mel_overlap = torch.zeros(1, 80, 0, device=device)
        self.hift_cache: Optional[dict] = None
        self.mel_overlap_len = mel_overlap_len
        self.mel_cache_len = mel_cache_len
        self.source_cache_len = mel_cache_len * 256
        self.mel_window = np.hamming(2 * mel_overlap_len)
        self.speech_window = np.hamming(2 * self.source_cache_len)
        self.audio_chunks: list[torch.Tensor] = []
        self.overlap_tokens = torch.zeros(1, 0, dtype=torch.int32, device=device)


# ---------------------------------------------------------------------------
# Core: one chunk through flow + vocoder
# ---------------------------------------------------------------------------

def token2wav(
    flow, hift, state: StreamState,
    target_token: torch.Tensor,
    prompt_token: torch.Tensor,
    prompt_feat: torch.Tensor,
    embedding: torch.Tensor,
    device: torch.device,
    n_timesteps: int,
    finalize: bool,
):
    """Run flow + vocoder for one token chunk; update state in-place."""
    from cosyvoice.utils.common import fade_in_out

    tts_mel, state.flow_cache = flow.inference(
        token=target_token.to(torch.int32),
        token_len=torch.tensor([target_token.shape[1]], dtype=torch.int32, device=device),
        prompt_token=prompt_token.to(torch.int32),
        prompt_token_len=torch.tensor([prompt_token.shape[1]], dtype=torch.int32, device=device),
        prompt_feat=prompt_feat,
        prompt_feat_len=torch.tensor([prompt_feat.shape[1]], dtype=torch.int32, device=device),
        embedding=embedding,
        flow_cache=state.flow_cache,
        n_timesteps=n_timesteps,
    )

    if state.mel_overlap.shape[2] != 0:
        tts_mel = fade_in_out(tts_mel, state.mel_overlap, state.mel_window)

    if state.hift_cache is not None:
        hift_cache_source = state.hift_cache["source"]
        tts_mel = torch.cat([state.hift_cache["mel"], tts_mel], dim=2)
    else:
        hift_cache_source = torch.zeros(1, 1, 0, device=device)

    if not finalize:
        state.mel_overlap = tts_mel[:, :, -state.mel_overlap_len:]
        tts_mel = tts_mel[:, :, :-state.mel_overlap_len]
        tts_speech, tts_source = hift.inference(
            speech_feat=tts_mel, cache_source=hift_cache_source,
        )
        if state.hift_cache is not None:
            tts_speech = fade_in_out(tts_speech, state.hift_cache["speech"], state.speech_window)
        state.hift_cache = {
            "mel": tts_mel[:, :, -state.mel_cache_len:],
            "source": tts_source[:, :, -state.source_cache_len:],
            "speech": tts_speech[:, -state.source_cache_len:],
        }
        tts_speech = tts_speech[:, :-state.source_cache_len]
    else:
        tts_speech, tts_source = hift.inference(
            speech_feat=tts_mel, cache_source=hift_cache_source,
        )
        if state.hift_cache is not None:
            tts_speech = fade_in_out(tts_speech, state.hift_cache["speech"], state.speech_window)

    state.audio_chunks.append(tts_speech.squeeze(0).cpu())


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    _ensure_import_paths()

    from hyperpyyaml import load_hyperpyyaml
    from cosyvoice.hifigan.generator import HiFTGenerator  # noqa: F401

    repo = _repo_root()
    default_cfg = repo / "conf" / "cosyvoice_aishell_s3tok1024_25hz.yaml"
    default_tok = repo / "pretrained_weights" / "s3tokenizer.pt"
    default_assets = repo / "pretrained_weights" / "CosyVoice-300M"

    import re
    default_torch_ddp = repo / "pretrained_weights" / "flow_torch_ddp"
    default_ckpt = None
    if default_torch_ddp.is_dir():
        best: tuple[int, Path] | None = None
        for p in default_torch_ddp.glob("epoch_*_whole.pt"):
            m = re.match(r"epoch_(\d+)_whole\.pt$", p.name)
            if m:
                n = int(m.group(1))
                if best is None or n > best[0]:
                    best = (n, p)
        if best:
            default_ckpt = best[1]

    p = argparse.ArgumentParser(
        description="Streaming inference: tokenizer + flow run on the same chunk boundary"
    )
    p.add_argument("--wav", required=True, help="Input wav (content source)")
    p.add_argument("--speaker_wav", default="", help="Optional speaker wav for cross-speaker mode")
    p.add_argument("--out_wav", default="stream_out.wav", help="Output wav path")
    p.add_argument("--chunk_ms", type=int, default=640,
                   help="Chunk size in ms — controls BOTH tokenizer and flow granularity")
    p.add_argument("--prompt_ms", type=int, default=1000,
                   help="Duration of initial audio used as prompt (ms)")
    p.add_argument("--token_overlap", type=int, default=10,
                   help="Overlap tokens re-processed at chunk boundaries for mel fade (default 10)")
    p.add_argument("--n_timesteps", type=int, default=20, help="Flow ODE solver steps")
    p.add_argument("--flow_ckpt", type=str, default=str(default_ckpt or ""))
    p.add_argument("--train_config", type=str, default=str(default_cfg))
    p.add_argument("--tokenizer_pt", type=str,
                   default=str(default_tok) if default_tok.is_file() else "")
    p.add_argument("--assets_dir", type=str,
                   default=str(default_assets) if (default_assets / "campplus.onnx").is_file() else "")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    for label, path in [
        ("flow_ckpt", args.flow_ckpt),
        ("train_config", args.train_config),
        ("tokenizer_pt", args.tokenizer_pt),
        ("wav", args.wav),
    ]:
        if not path or not os.path.isfile(path):
            raise FileNotFoundError(f"--{label} not found: {path}")
    camp_path = os.path.join(args.assets_dir, "campplus.onnx")
    hift_path = os.path.join(args.assets_dir, "hift.pt")
    for path in [camp_path, hift_path]:
        if not os.path.isfile(path):
            raise FileNotFoundError(path)
    if args.speaker_wav and not os.path.isfile(args.speaker_wav):
        raise FileNotFoundError(f"--speaker_wav not found: {args.speaker_wav}")

    device = torch.device(args.device)
    print(f"[stream] chunk_ms={args.chunk_ms}  prompt_ms={args.prompt_ms}  "
          f"token_overlap={args.token_overlap}  n_timesteps={args.n_timesteps}", flush=True)

    # -----------------------------------------------------------------------
    # Load models
    # -----------------------------------------------------------------------
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

    flow_state = strip_meta_state_dict(
        torch.load(args.flow_ckpt, map_location="cpu", weights_only=True)
    )
    flow.load_state_dict(flow_state, strict=True)
    flow.to(device).eval()

    hift: HiFTGenerator = configs["hift"]
    hift_state = torch.load(hift_path, map_location="cpu", weights_only=True)
    if any(k.startswith("generator.") for k in hift_state):
        hift_state = {k.replace("generator.", ""): v for k, v in hift_state.items()}
    hift.load_state_dict(hift_state, strict=True)
    hift.to(device).eval()

    tokenizer = load_s3_tokenizer(args.tokenizer_pt, device)

    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    opts.intra_op_num_threads = 1
    camp_sess = ort.InferenceSession(camp_path, sess_options=opts, providers=["CPUExecutionProvider"])

    # -----------------------------------------------------------------------
    # Load audio
    # -----------------------------------------------------------------------
    speech_22k = load_wav_1ch(args.wav, sample_rate).to(device)
    speech_16k = load_wav_1ch(args.wav, 16000).to(device)
    spk_wav_16k = load_wav_1ch(args.speaker_wav, 16000).to(device) if args.speaker_wav else speech_16k

    # -----------------------------------------------------------------------
    # Phase 1: establish prompt
    # -----------------------------------------------------------------------
    prompt_samples_16k = int(args.prompt_ms / 1000.0 * 16000)
    prompt_samples_22k = int(args.prompt_ms / 1000.0 * sample_rate)

    prompt_16k = speech_16k[:, :prompt_samples_16k]
    prompt_22k = speech_22k[:, :prompt_samples_22k]

    with torch.no_grad():
        prompt_mel_full = feat_extractor(prompt_22k).squeeze(0).transpose(0, 1).contiguous()
        prompt_whisper_mel = whisper.log_mel_spectrogram(prompt_16k.squeeze(0), n_mels=128).to(device)
        if prompt_whisper_mel.ndim == 2:
            prompt_whisper_mel = prompt_whisper_mel.unsqueeze(0)
        prompt_tokens_all = tokenizer.tokenize(prompt_whisper_mel)[0]
        embedding = extract_spk_embedding(spk_wav_16k, camp_sess, device)

    n_prompt_tok = prompt_tokens_all.shape[0]
    n_prompt_mel = prompt_mel_full.shape[0]
    n_prompt_tok = align_token_mel_lengths(n_prompt_tok, n_prompt_mel, input_frame_rate, sample_rate)
    n_prompt_mel = int(n_prompt_tok * sample_rate / (input_frame_rate * 256))
    n_prompt_mel = min(n_prompt_mel, prompt_mel_full.shape[0])

    prompt_token = prompt_tokens_all[:n_prompt_tok].to(torch.int32).unsqueeze(0).to(device)
    prompt_feat = prompt_mel_full[:n_prompt_mel].unsqueeze(0).to(device)

    print(f"[stream] prompt: {n_prompt_tok} tokens, {n_prompt_mel} mel frames "
          f"({n_prompt_tok / input_frame_rate:.2f}s)", flush=True)

    # -----------------------------------------------------------------------
    # Phase 2: streaming — tokenizer + flow on every chunk
    # -----------------------------------------------------------------------
    remaining_16k = speech_16k[:, prompt_samples_16k:]
    total_remaining_samples = remaining_16k.shape[1]
    chunk_samples = int(args.chunk_ms / 1000.0 * 16000)

    token_overlap = args.token_overlap
    mel_overlap_len = int(token_overlap / input_frame_rate * sample_rate / 256)
    state = StreamState(device, mel_overlap_len)

    actual_device = prompt_whisper_mel.device
    tok_state = tokenizer.init_stream_state(1, actual_device, torch.float32)
    if prompt_whisper_mel.shape[2] > 0:
        _, tok_state = tokenizer.stream_step(prompt_whisper_mel, tok_state)

    # Minimum mel frames the vocoder needs (kernel_size=3 in f0 condnet)
    # plus mel_overlap we save for fade, plus mel_cache for hift.
    MIN_VOCODER_MEL = 4
    min_mel_for_call = mel_overlap_len + state.mel_cache_len + MIN_VOCODER_MEL
    mel_per_token = sample_rate / (input_frame_rate * 256)
    min_tokens_for_call = int(np.ceil(min_mel_for_call / mel_per_token))

    pending_tokens = torch.zeros(1, 0, dtype=torch.int32, device=device)
    n_chunks = 0
    n_flow_calls = 0
    offset = 0

    while offset < total_remaining_samples:
        end = min(offset + chunk_samples, total_remaining_samples)
        audio_chunk_16k = remaining_16k[:, offset:end]
        is_last_chunk = (end >= total_remaining_samples)
        offset = end

        # --- tokenize this chunk ---
        with torch.no_grad():
            mel_chunk = whisper.log_mel_spectrogram(audio_chunk_16k.squeeze(0), n_mels=128).to(device)
            if mel_chunk.ndim == 2:
                mel_chunk = mel_chunk.unsqueeze(0)
            new_tokens, tok_state = tokenizer.stream_step(mel_chunk, tok_state)
        n_chunks += 1

        if new_tokens.shape[1] > 0:
            pending_tokens = torch.cat([pending_tokens, new_tokens.to(torch.int32)], dim=1)

        # Build target: overlap from previous call + accumulated new tokens
        target = torch.cat([state.overlap_tokens, pending_tokens], dim=1)

        # Run flow when we have enough tokens for the vocoder, or on last chunk
        ready = target.shape[1] >= min_tokens_for_call
        if not ready and not is_last_chunk:
            continue
        if target.shape[1] == 0:
            continue

        with torch.inference_mode():
            token2wav(
                flow, hift, state,
                target_token=target,
                prompt_token=prompt_token,
                prompt_feat=prompt_feat,
                embedding=embedding,
                device=device,
                n_timesteps=args.n_timesteps,
                finalize=is_last_chunk,
            )
        n_flow_calls += 1

        chunk_dur = pending_tokens.shape[1] / input_frame_rate
        print(f"  chunk {n_chunks}: +{pending_tokens.shape[1]} new tokens ({chunk_dur:.3f}s) "
              f"→ flow call {n_flow_calls} ({target.shape[1]} tokens incl. overlap)"
              f"{'  [final]' if is_last_chunk else ''}", flush=True)

        # Keep tail tokens as overlap for next chunk; clear pending
        if not is_last_chunk and target.shape[1] > token_overlap:
            state.overlap_tokens = target[:, -token_overlap:]
        elif not is_last_chunk:
            state.overlap_tokens = target
        else:
            state.overlap_tokens = torch.zeros(1, 0, dtype=torch.int32, device=device)
        pending_tokens = torch.zeros(1, 0, dtype=torch.int32, device=device)

    # -----------------------------------------------------------------------
    # Write output – prepend the original prompt audio so the result is
    # complete rather than missing the initial prompt_ms.
    # -----------------------------------------------------------------------
    if not state.audio_chunks:
        raise RuntimeError("No audio generated. Input may be too short after prompt.")

    generated = torch.cat(state.audio_chunks, dim=-1)

    prompt_audio_22k = speech_22k[:, :prompt_samples_22k].squeeze(0).cpu()
    out = torch.cat([prompt_audio_22k, generated], dim=-1)

    torchaudio.save(args.out_wav, out.unsqueeze(0), sample_rate)
    dur = out.numel() / sample_rate
    print(
        f"\nWrote {args.out_wav} ({dur:.2f}s, prompt={prompt_audio_22k.numel() / sample_rate:.2f}s + "
        f"generated={generated.numel() / sample_rate:.2f}s). "
        f"audio_chunks={n_chunks}  flow_calls={n_flow_calls}  "
        f"prompt_tokens={n_prompt_tok}  chunk_ms={args.chunk_ms}  "
        f"token_overlap={token_overlap}  n_timesteps={args.n_timesteps}",
        flush=True,
    )


if __name__ == "__main__":
    main()
