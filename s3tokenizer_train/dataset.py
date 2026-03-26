"""
AISHELL dataset for S3Tokenizer training.

Loads audio from wav.scp, text from text file.
Produces Whisper-compatible mel spectrograms and token sequences.
"""

import logging
import math
import os
from typing import Dict, List, Tuple

import numpy as np
import torch
import torchaudio
import whisper
import soundfile as sf
from torch.utils.data import Dataset
from scipy.signal import resample_poly

logger = logging.getLogger(__name__)

WHISPER_SAMPLE_RATE = 16000
N_MELS = 128
MAX_AUDIO_SEC = 30.0
HOP_LENGTH = 160
CHUNK_LENGTH = 30  # Whisper's expected chunk length in seconds
N_FRAMES = 3000  # CHUNK_LENGTH * WHISPER_SAMPLE_RATE / HOP_LENGTH


def load_manifest(data_dir: str) -> List[Tuple[str, str, str]]:
    """Load wav.scp and text files. Returns list of (utt_id, wav_path, text)."""
    wav_scp = os.path.join(data_dir, "wav.scp")
    text_file = os.path.join(data_dir, "text")

    wavs = {}
    with open(wav_scp, "r") as f:
        for line in f:
            parts = line.strip().split(maxsplit=1)
            if len(parts) == 2:
                wavs[parts[0]] = parts[1]

    texts = {}
    with open(text_file, "r") as f:
        for line in f:
            parts = line.strip().split(maxsplit=1)
            if len(parts) == 2:
                texts[parts[0]] = parts[1]

    items = []
    for utt_id in sorted(wavs.keys()):
        if utt_id in texts:
            items.append((utt_id, wavs[utt_id], texts[utt_id]))

    logger.info(f"Loaded {len(items)} utterances from {data_dir}")
    return items


class AishellS3Dataset(Dataset):
    """
    Dataset that returns (mel, token_ids, utt_id) tuples.

    mel: (n_mels, N_FRAMES) padded to 30 seconds
    token_ids: Whisper-format [sot, lang, task, notimestamps, ...text_tokens..., eot]
    """

    def __init__(self, data_dir: str, language: str = "zh", model_sample_rate: int = WHISPER_SAMPLE_RATE):
        self.items = load_manifest(data_dir)
        self.model_sample_rate = model_sample_rate
        self._torchaudio_fallback_logged = False
        self.tokenizer = whisper.tokenizer.get_tokenizer(
            multilingual=True, language=language, task="transcribe"
        )
        self.sot_sequence = list(self.tokenizer.sot_sequence)
        self.no_timestamps = self.tokenizer.no_timestamps
        self.eot = self.tokenizer.eot

    def __len__(self):
        return len(self.items)

    def _load_waveform(self, wav_path: str) -> Tuple[torch.Tensor, int]:
        """Load audio with torchaudio first; fallback to soundfile if needed."""
        try:
            waveform, sr = torchaudio.load(wav_path)
            if waveform.size(0) > 1:
                waveform = waveform.mean(dim=0, keepdim=True)
            return waveform.squeeze(0), int(sr)
        except Exception as exc:
            if not self._torchaudio_fallback_logged:
                logger.warning("torchaudio.load failed; using soundfile fallback (%s)", exc)
                self._torchaudio_fallback_logged = True
            audio, sr = sf.read(wav_path, dtype="float32", always_2d=False)
            if isinstance(audio, np.ndarray) and audio.ndim > 1:
                # soundfile uses (T, C)
                audio = audio.mean(axis=1)
            waveform = torch.from_numpy(np.asarray(audio, dtype=np.float32))
            return waveform, int(sr)

    def _resample_waveform(self, waveform: torch.Tensor, src_sr: int) -> torch.Tensor:
        """Resample to model sample rate with robust fallback."""
        if src_sr == self.model_sample_rate:
            return waveform
        try:
            return torchaudio.functional.resample(waveform, src_sr, self.model_sample_rate)
        except Exception:
            g = math.gcd(src_sr, self.model_sample_rate)
            up = self.model_sample_rate // g
            down = src_sr // g
            resampled = resample_poly(waveform.cpu().numpy(), up=up, down=down)
            return torch.from_numpy(np.asarray(resampled, dtype=np.float32))

    def __getitem__(self, idx):
        utt_id, wav_path, text = self.items[idx]

        waveform, sr = self._load_waveform(wav_path)

        if sr != self.model_sample_rate:
            waveform = self._resample_waveform(waveform, sr)

        max_samples = int(MAX_AUDIO_SEC * self.model_sample_rate)
        if waveform.numel() > max_samples:
            waveform = waveform[:max_samples]

        mel = whisper.log_mel_spectrogram(waveform, n_mels=N_MELS)
        # mel shape: (n_mels, T) where T <= N_FRAMES
        mel_len = min(mel.shape[1], N_FRAMES)

        # Pad mel to N_FRAMES (Whisper expects exactly 3000 frames for 30s)
        if mel_len < N_FRAMES:
            mel = F.pad(mel[:, :mel_len], (0, N_FRAMES - mel_len))
        else:
            mel = mel[:, :N_FRAMES]

        # Tokenize text
        text_tokens = self.tokenizer.encode(text)
        # Build Whisper decoder input: [sot, lang, task, no_timestamps, ...text..., eot]
        token_ids = self.sot_sequence + [self.no_timestamps] + text_tokens + [self.eot]
        token_ids = torch.tensor(token_ids, dtype=torch.long)

        return {
            "mel": mel,
            "mel_len": mel_len,
            "tokens": token_ids,
            "token_len": token_ids.shape[0],
            "utt_id": utt_id,
        }


def collate_fn(batch: List[Dict]) -> Dict:
    """Collate function that pads token sequences to equal length."""
    mels = torch.stack([item["mel"] for item in batch])
    mel_lens = torch.tensor([item["mel_len"] for item in batch], dtype=torch.long)
    token_lens = torch.tensor([item["token_len"] for item in batch], dtype=torch.long)

    # Pad tokens to max length in batch
    max_len = max(item["tokens"].shape[0] for item in batch)
    tokens_padded = torch.full((len(batch), max_len), fill_value=50257, dtype=torch.long)  # EOT as padding
    for i, item in enumerate(batch):
        t = item["tokens"]
        tokens_padded[i, : t.shape[0]] = t

    utt_ids = [item["utt_id"] for item in batch]

    return {
        "mel": mels,           # (B, n_mels, N_FRAMES)
        "mel_lens": mel_lens,  # (B,)
        "tokens": tokens_padded,  # (B, S_max)
        "token_lens": token_lens,  # (B,)
        "utt_ids": utt_ids,
    }


# Need F for padding
import torch.nn.functional as F
