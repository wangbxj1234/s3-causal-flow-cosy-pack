#!/usr/bin/env bash
# Streaming inference: tokenizer + flow run on every chunk boundary.
#
# Usage (self-prompt):
#   bash examples/infer_streaming_example.sh <input.wav> [out.wav] [chunk_ms]
#
# Usage (cross-speaker):
#   bash examples/infer_streaming_example.sh <content.wav> <speaker.wav> <out.wav> [chunk_ms]
#
# chunk_ms controls BOTH tokenizer and flow granularity (default 640ms).
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO"
export PYTHONPATH="${REPO}:${REPO}/third_party/Matcha-TTS"
PY="${PYTHON:-python3}"

WAV="${1:?usage: $0 <input.wav> [out.wav | speaker.wav out.wav] [chunk_ms]}"

if [[ $# -ge 3 && -f "${2}" && "${2}" == *.wav ]]; then
    SPEAKER="${2}"
    OUT="${3}"
    CHUNK_MS="${4:-640}"
    "${PY}" tools/infer_flow_streaming_s3tok25hz.py \
      --wav "${WAV}" \
      --speaker_wav "${SPEAKER}" \
      --out_wav "${OUT}" \
      --chunk_ms "${CHUNK_MS}" \
      --n_timesteps 20
else
    OUT="${2:-${REPO}/stream_out.wav}"
    CHUNK_MS="${3:-640}"
    "${PY}" tools/infer_flow_streaming_s3tok25hz.py \
      --wav "${WAV}" \
      --out_wav "${OUT}" \
      --chunk_ms "${CHUNK_MS}" \
      --n_timesteps 20
fi

echo "wrote ${OUT}"
