import sys
import types

import pytest
import torch
import torch.nn as nn


def _install_fake_whisper():
    fake_whisper = types.ModuleType("whisper")
    fake_whisper_model = types.ModuleType("whisper.model")

    class FakeResidualAttentionBlock(nn.Module):
        def __init__(self, n_state: int, n_head: int):
            super().__init__()
            self.proj = nn.Linear(n_state, n_state)

        def forward(self, x, mask=None):
            return self.proj(x)

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


_install_fake_whisper()
from s3tokenizer_train.model import S3Config, WhisperWithVQ  # noqa: E402


class _FakeWhisperModel:
    def __init__(self, n_mels: int, n_state: int, n_heads: int, n_ctx: int, n_blocks: int):
        encoder = nn.Module()
        encoder.conv1 = nn.Conv1d(n_mels, n_state, kernel_size=3, padding=1)
        encoder.conv2 = nn.Conv1d(n_state, n_state, kernel_size=3, stride=2, padding=1)
        encoder.positional_embedding = torch.zeros(n_ctx, n_state)
        encoder.blocks = nn.ModuleList([sys.modules["whisper.model"].ResidualAttentionBlock(n_state, n_heads) for _ in range(n_blocks)])
        encoder.ln_post = sys.modules["whisper.model"].LayerNorm(n_state)
        self.encoder = encoder
        self.decoder = nn.Identity()
        self.dims = types.SimpleNamespace()


def _build_model(audio_ctx: int = 16) -> WhisperWithVQ:
    _install_fake_whisper()
    cfg = S3Config(
        n_mels=4,
        n_audio_ctx=audio_ctx,
        n_audio_state=8,
        n_audio_head=2,
        n_encoder1_layers=2,
        n_encoder2_layers=2,
        n_codebook_size=16,
        pre_vq_stride=2,
    )
    fake = _FakeWhisperModel(
        n_mels=cfg.n_mels,
        n_state=cfg.n_audio_state,
        n_heads=cfg.n_audio_head,
        n_ctx=cfg.n_audio_ctx,
        n_blocks=cfg.n_encoder1_layers + cfg.n_encoder2_layers,
    )
    return WhisperWithVQ(cfg, whisper_model=fake)


def test_stream_state_requires_consistent_batch_and_device():
    model = _build_model(audio_ctx=16)
    state = model.init_stream_state(batch_size=1, device=torch.device("cpu"), dtype=torch.float32)
    bad_chunk = torch.randn(2, 4, 8)
    with pytest.raises(ValueError):
        model.stream_step(bad_chunk, state)


def test_stream_step_rejects_invalid_mel_shape_or_empty_chunk():
    model = _build_model(audio_ctx=16)
    state = model.init_stream_state(batch_size=1, device=torch.device("cpu"), dtype=torch.float32)
    with pytest.raises(ValueError):
        model.stream_step(torch.randn(1, 4), state)
    with pytest.raises(ValueError):
        model.stream_step(torch.randn(1, 4, 0), state)


def test_stream_step_raises_on_positional_overflow():
    model = _build_model(audio_ctx=2)
    state = model.init_stream_state(batch_size=1, device=torch.device("cpu"), dtype=torch.float32)
    with pytest.raises(ValueError):
        model.stream_step(torch.randn(1, 4, 10), state)


def test_stream_step_requires_consistent_dtype():
    model = _build_model(audio_ctx=16)
    state = model.init_stream_state(batch_size=1, device=torch.device("cpu"), dtype=torch.float32)
    with pytest.raises(ValueError):
        model.stream_step(torch.randn(1, 4, 4, dtype=torch.float64), state)
