import torch
import torch.nn as nn

from s3tokenizer_train.causal_ops import causal_conv1d_full, init_causal_conv1d_state, causal_conv1d_step


def test_causal_conv1d_chunked_matches_offline():
    torch.manual_seed(0)
    conv = nn.Conv1d(4, 6, kernel_size=3, stride=2, padding=0, bias=True)
    x = torch.randn(2, 4, 17)

    y_full = causal_conv1d_full(conv, x)

    state = init_causal_conv1d_state(batch_size=2, channels=4, kernel_size=3, device=x.device, dtype=x.dtype)
    y_chunks = []
    for chunk in (x[:, :, :5], x[:, :, 5:11], x[:, :, 11:]):
        y, state = causal_conv1d_step(conv, chunk, state)
        if y.numel() > 0:
            y_chunks.append(y)
    y_stream = torch.cat(y_chunks, dim=2)

    assert y_full.shape == y_stream.shape
    assert torch.allclose(y_full, y_stream, atol=1e-6, rtol=1e-6)
