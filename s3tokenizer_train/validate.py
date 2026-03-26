"""
Validate trained S3Tokenizer against the official ONNX tokenizer.

Compares:
1. Token distribution statistics
2. Token agreement rate (how often tokens match)
3. Codebook utilization

Usage:
  python -m s3tokenizer_train.validate \
    --tokenizer exp/s3tokenizer_v1/s3tokenizer.pt \
    --onnx_path /mnt/data/Marco-Voice-main/pretrained_models/marco_voice/marco_voice/speech_tokenizer_v1.onnx \
    --wav_dir data/aishell_s3/test \
    --num_samples 100
"""

import argparse
import logging
import os

import numpy as np
import onnxruntime
import torch
import whisper

from s3tokenizer_train.export import S3TokenizerV1, S3Config

logger = logging.getLogger(__name__)


def load_trained_tokenizer(path: str, device: torch.device) -> S3TokenizerV1:
    """Load exported S3Tokenizer."""
    data = torch.load(path, map_location="cpu")
    config_dict = data["config"]
    config = S3Config(**config_dict)
    model = S3TokenizerV1(config)
    model.load_state_dict(data["model"])
    model = model.to(device).eval()
    return model


def load_onnx_tokenizer(onnx_path: str):
    """Load official ONNX tokenizer."""
    option = onnxruntime.SessionOptions()
    option.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
    option.intra_op_num_threads = 1
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    return onnxruntime.InferenceSession(onnx_path, sess_options=option, providers=providers)


def tokenize_with_onnx(session, mel: torch.Tensor) -> np.ndarray:
    """Run ONNX tokenizer on a mel spectrogram."""
    out = session.run(
        None,
        {
            session.get_inputs()[0].name: mel.cpu().numpy(),
            session.get_inputs()[1].name: np.array([mel.shape[2]], dtype=np.int32),
        },
    )
    return out[0].flatten()


def main():
    parser = argparse.ArgumentParser(description="Validate S3Tokenizer")
    parser.add_argument("--tokenizer", required=True, help="Trained tokenizer path")
    parser.add_argument("--onnx_path", required=True, help="Official ONNX tokenizer path")
    parser.add_argument("--wav_dir", required=True, help="Directory with wav.scp for test audio")
    parser.add_argument("--num_samples", type=int, default=100)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger.info("Loading trained tokenizer...")
    trained = load_trained_tokenizer(args.tokenizer, device)

    logger.info("Loading ONNX tokenizer...")
    onnx_session = load_onnx_tokenizer(args.onnx_path)

    # Load wav.scp
    wav_scp = os.path.join(args.wav_dir, "wav.scp")
    wavs = []
    with open(wav_scp, "r") as f:
        for line in f:
            parts = line.strip().split(maxsplit=1)
            if len(parts) == 2:
                wavs.append((parts[0], parts[1]))

    wavs = wavs[: args.num_samples]
    logger.info(f"Evaluating on {len(wavs)} samples...")

    trained_all_tokens = []
    onnx_all_tokens = []
    n_total_frames = 0
    n_matching_frames = 0

    for utt_id, wav_path in wavs:
        audio = whisper.load_audio(wav_path)
        mel = whisper.log_mel_spectrogram(audio, n_mels=128)  # (128, T)

        # Trained tokenizer
        mel_padded = mel.unsqueeze(0).to(device)
        if mel_padded.shape[2] < 3000:
            mel_padded = torch.nn.functional.pad(mel_padded, (0, 3000 - mel_padded.shape[2]))
        trained_tokens = trained.tokenize(mel_padded)[0].cpu().numpy()

        # ONNX tokenizer
        onnx_tokens = tokenize_with_onnx(onnx_session, mel.unsqueeze(0))

        # Compare (up to min length)
        min_len = min(len(trained_tokens), len(onnx_tokens))
        if min_len > 0:
            n_match = (trained_tokens[:min_len] == onnx_tokens[:min_len]).sum()
            n_total_frames += min_len
            n_matching_frames += n_match

        trained_all_tokens.extend(trained_tokens.tolist())
        onnx_all_tokens.extend(onnx_tokens.tolist())

    # Statistics
    trained_arr = np.array(trained_all_tokens)
    onnx_arr = np.array(onnx_all_tokens)

    logger.info("=" * 60)
    logger.info("Validation Results")
    logger.info("=" * 60)

    if n_total_frames > 0:
        agreement = n_matching_frames / n_total_frames * 100
        logger.info(f"Token agreement rate: {agreement:.2f}% ({n_matching_frames}/{n_total_frames})")
    else:
        logger.info("No frames to compare")

    logger.info(f"Trained tokenizer - unique codes: {len(np.unique(trained_arr))}, "
                f"total tokens: {len(trained_arr)}")
    logger.info(f"ONNX tokenizer   - unique codes: {len(np.unique(onnx_arr))}, "
                f"total tokens: {len(onnx_arr)}")

    # Token distribution comparison
    trained_hist = np.bincount(trained_arr, minlength=4096)
    onnx_hist = np.bincount(onnx_arr.astype(int), minlength=4096)

    # Normalize
    trained_dist = trained_hist / max(trained_hist.sum(), 1)
    onnx_dist = onnx_hist / max(onnx_hist.sum(), 1)

    # KL divergence (with smoothing)
    eps = 1e-8
    kl = np.sum(onnx_dist * np.log((onnx_dist + eps) / (trained_dist + eps)))
    logger.info(f"KL divergence (ONNX || trained): {kl:.4f}")

    # Top-10 most frequent codes comparison
    trained_top10 = np.argsort(trained_hist)[-10:][::-1]
    onnx_top10 = np.argsort(onnx_hist)[-10:][::-1]
    overlap = len(set(trained_top10) & set(onnx_top10))
    logger.info(f"Top-10 code overlap: {overlap}/10")
    logger.info(f"  Trained top-10: {trained_top10.tolist()}")
    logger.info(f"  ONNX top-10:    {onnx_top10.tolist()}")

    logger.info("=" * 60)


if __name__ == "__main__":
    main()
