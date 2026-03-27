#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
causal S3 tokenizer + causal-data Flow 专用：跨说话人重建包装脚本。

复用 tools/infer_flow_reconstruct_cross_speaker_s3tok25hz.py，默认 tokenizer / flow ckpt
规则同 infer_flow_reconstruct_causal_s3tok25hz.py。
"""
from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path


def _load_cross_module():
    here = Path(__file__).resolve().parent
    base = here / "infer_flow_reconstruct_cross_speaker_s3tok25hz.py"
    if not base.is_file():
        raise FileNotFoundError(f"base script not found: {base}")
    spec = importlib.util.spec_from_file_location(
        "infer_flow_reconstruct_cross_speaker_s3tok25hz", base
    )
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def _pick_default_torch_ddp(repo_root: Path) -> Path | None:
    exp_root = repo_root / "exp"
    if not exp_root.is_dir():
        return None

    candidates = sorted(
        exp_root.glob("cosyvoice1_flow_s3tok1024_25hz_officialinit_causaldata_*/flow/torch_ddp"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if candidates:
        return candidates[0]

    candidates = sorted(
        exp_root.glob("cosyvoice1_flow_s3tok1024_25hz_officialinit_*/flow/torch_ddp"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if candidates:
        return candidates[0]
    return None


def _pick_default_tokenizer(repo_root: Path) -> Path | None:
    p = repo_root / "pretrained_weights" / "s3tokenizer.pt"
    return p if p.is_file() else None


def _has_any(argv: list[str], flags: tuple[str, ...]) -> bool:
    for a in argv:
        if a in flags:
            return True
        for f in flags:
            if a.startswith(f + "="):
                return True
    return False


def main() -> None:
    cross = _load_cross_module()
    inf = cross._load_base()
    repo = inf._repo_root()

    argv = sys.argv[1:]
    extra: list[str] = []

    if not _has_any(argv, ("--preset",)):
        extra += ["--preset", "custom_s3tok25hz"]

    if not _has_any(argv, ("--flow_ckpt", "--torch_ddp_dir")):
        tdd = _pick_default_torch_ddp(repo)
        if tdd is not None:
            extra += ["--torch_ddp_dir", str(tdd)]

    env_tok = os.environ.get("COSYVOICE_S3_TOKENIZER_PT", "").strip()
    if not _has_any(argv, ("--tokenizer_pt",)) and not env_tok:
        tok = _pick_default_tokenizer(repo)
        if tok is not None:
            extra += ["--tokenizer_pt", str(tok)]

    sys.argv = [sys.argv[0], *extra, *argv]
    print("[causal cross wrapper] forwarding args:", " ".join(sys.argv[1:]), flush=True)
    cross.main()


if __name__ == "__main__":
    main()
