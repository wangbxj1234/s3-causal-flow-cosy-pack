import sys
import types

import torch
import torch.nn as nn


def _install_fake_whisper():
    fake_whisper = types.ModuleType("whisper")
    fake_whisper_model = types.ModuleType("whisper.model")

    class FakeResidualAttentionBlock(nn.Module):
        def __init__(self, n_state: int, n_head: int):
            super().__init__()

        def forward(self, x, mask=None):
            return x

    class FakeLayerNorm(nn.LayerNorm):
        def __init__(self, n_state: int):
            super().__init__(n_state)

    class _Unused(nn.Module):
        def forward(self, x):
            return x

    fake_whisper_model.AudioEncoder = _Unused
    fake_whisper_model.ResidualAttentionBlock = FakeResidualAttentionBlock
    fake_whisper_model.LayerNorm = FakeLayerNorm
    fake_whisper_model.sinusoids = lambda n_ctx, n_state: torch.zeros(n_ctx, n_state)
    fake_whisper.model = fake_whisper_model
    fake_whisper.Whisper = object
    fake_whisper.load_model = lambda *args, **kwargs: None
    sys.modules["whisper"] = fake_whisper
    sys.modules["whisper.model"] = fake_whisper_model


def _parse_args(extra):
    _install_fake_whisper()
    from s3tokenizer_train.train import get_args

    old = sys.argv[:]
    try:
        sys.argv = [
            "train.py",
            "--train_dir",
            "data/train",
            "--dev_dir",
            "data/dev",
            "--output_dir",
            "exp/out",
        ] + list(extra)
        return get_args()
    finally:
        sys.argv = old


def test_train_parser_has_throughput_flags():
    args = _parse_args(
        [
            "--grad_accum_steps",
            "4",
            "--use_torch_compile",
            "--prefetch_factor",
            "4",
            "--persistent_workers",
            "--allow_tf32",
        ]
    )
    assert args.grad_accum_steps == 4
    assert args.use_torch_compile is True
    assert args.prefetch_factor == 4
    assert args.persistent_workers is True
    assert args.allow_tf32 is True
