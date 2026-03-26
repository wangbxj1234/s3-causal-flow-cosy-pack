import sys
import types
import importlib

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
    fake_whisper_model.LayerNorm = FakeLayerNorm
    fake_whisper_model.ResidualAttentionBlock = FakeResidualAttentionBlock
    fake_whisper_model.sinusoids = lambda n_ctx, n_state: torch.zeros(n_ctx, n_state)
    fake_whisper.model = fake_whisper_model
    fake_whisper.Whisper = object
    fake_whisper.load_model = lambda *args, **kwargs: None
    sys.modules["whisper"] = fake_whisper
    sys.modules["whisper.model"] = fake_whisper_model


_install_fake_whisper()


def test_extract_tokenizer_state_dict_includes_causal_components():
    _install_fake_whisper()
    import s3tokenizer_train.export as export_mod
    export_mod = importlib.reload(export_mod)
    extract_tokenizer_state_dict = export_mod.extract_tokenizer_state_dict

    full_state = {
        "conv1.weight": torch.randn(8, 4, 3),
        "conv2.weight": torch.randn(8, 8, 3),
        "positional_embedding": torch.randn(16, 8),
        "encoder1_blocks.0.proj.weight": torch.randn(8, 8),
        "encoder1_ln.weight": torch.randn(8),
        "pre_vq_downsample.weight": torch.randn(8, 8, 3),
        "vq.codebook.embed": torch.randn(16, 8),
    }
    out = extract_tokenizer_state_dict(full_state)
    assert "conv1.weight" in out
    assert "conv2.weight" in out
    assert "pre_vq_downsample.weight" in out
    assert "vq.codebook.embed" in out


def test_exported_tokenizer_streaming_matches_model_streaming():
    _install_fake_whisper()
    import s3tokenizer_train.export as export_mod
    export_mod = importlib.reload(export_mod)
    S3Config = export_mod.S3Config
    S3TokenizerV1 = export_mod.S3TokenizerV1

    cfg = S3Config(
        n_mels=4,
        n_audio_ctx=64,
        n_audio_state=8,
        n_audio_head=2,
        n_encoder1_layers=2,
        n_codebook_size=16,
    )
    model = S3TokenizerV1(cfg)
    mel = torch.randn(1, 4, 19)
    offline = model.tokenize(mel)
    state = model.init_stream_state(batch_size=1, device=mel.device, dtype=mel.dtype)
    out = []
    for chunk in (mel[:, :, :6], mel[:, :, 6:12], mel[:, :, 12:]):
        tok, state = model.stream_step(chunk, state)
        if tok.numel() > 0:
            out.append(tok)
    tail, _ = model.finalize_stream(state)
    if tail.numel() > 0:
        out.append(tail)
    stream = torch.cat(out, dim=1)
    assert torch.equal(offline, stream)
