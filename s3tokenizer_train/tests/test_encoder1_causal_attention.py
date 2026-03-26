import sys
import types
import importlib

import torch
import torch.nn as nn


def _install_fake_whisper_with_mask_block():
    fake_whisper = types.ModuleType("whisper")
    fake_whisper_model = types.ModuleType("whisper.model")

    class FakeResidualAttentionBlock(nn.Module):
        def __init__(self, n_state: int, n_head: int):
            super().__init__()
            self.last_mask = None

        def forward(self, x, mask):
            self.last_mask = mask
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


_install_fake_whisper_with_mask_block()


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


def _build_model(audio_ctx: int = 16):
    _install_fake_whisper_with_mask_block()
    import s3tokenizer_train.model as model_mod
    model_mod = importlib.reload(model_mod)
    S3Config = model_mod.S3Config
    WhisperWithVQ = model_mod.WhisperWithVQ

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


def test_encoder1_causal_mask_shape_and_triangle_property():
    model = _build_model(audio_ctx=16)
    mel = torch.randn(1, 4, 8)
    _ = model.encode_stage1(mel)
    first_block = model.encoder1_blocks[0]
    mask = first_block.last_mask
    assert mask is not None
    t = mask.shape[0]
    assert mask.shape == (t, t)
    upper = torch.triu(mask, diagonal=1)
    assert torch.isneginf(upper[upper != 0]).all()
    lower = torch.tril(mask)
    assert torch.all(lower == 0)


def test_encoder1_incremental_with_cache_matches_full_causal():
    model = _build_model(audio_ctx=32)
    mel = torch.randn(1, 4, 14)
    offline = model.tokenize(mel)
    state = model.init_stream_state(batch_size=1, device=mel.device, dtype=mel.dtype)
    out = []
    for chunk in (mel[:, :, :4], mel[:, :, 4:9], mel[:, :, 9:]):
        tok, state = model.stream_step(chunk, state)
        if tok.numel() > 0:
            out.append(tok)
    stream = torch.cat(out, dim=1)
    assert torch.equal(offline, stream)
