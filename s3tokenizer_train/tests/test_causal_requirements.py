import sys
import types

import pytest
import torch
import torch.nn as nn


def _install_fake_whisper_without_mask():
    fake_whisper = types.ModuleType("whisper")
    fake_whisper_model = types.ModuleType("whisper.model")

    class FakeResidualAttentionBlock(nn.Module):
        def __init__(self, n_state: int, n_head: int):
            super().__init__()

        def forward(self, x):
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


def test_whisper_blocks_must_support_mask_for_strict_causal():
    _install_fake_whisper_without_mask()
    from s3tokenizer_train.model import S3Config, WhisperWithVQ

    cfg = S3Config(
        n_mels=4,
        n_audio_ctx=16,
        n_audio_state=8,
        n_audio_head=2,
        n_encoder1_layers=2,
        n_encoder2_layers=2,
        n_codebook_size=16,
    )

    encoder = nn.Module()
    encoder.conv1 = nn.Conv1d(4, 8, kernel_size=3, padding=1)
    encoder.conv2 = nn.Conv1d(8, 8, kernel_size=3, stride=2, padding=1)
    encoder.positional_embedding = torch.zeros(16, 8)
    encoder.blocks = nn.ModuleList([sys.modules["whisper.model"].ResidualAttentionBlock(8, 2) for _ in range(4)])
    encoder.ln_post = sys.modules["whisper.model"].LayerNorm(8)
    fake_whisper = types.SimpleNamespace(encoder=encoder, decoder=nn.Identity(), dims=types.SimpleNamespace())

    with pytest.raises(RuntimeError):
        WhisperWithVQ(cfg, whisper_model=fake_whisper)
