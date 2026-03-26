#!/usr/bin/env bash
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO"
export PYTHONPATH="${REPO}:${REPO}/third_party/Matcha-TTS"
PY="${PYTHON:-python3}"

CONTENT="${1:?usage: $0 <content.wav> <speaker.wav> [out.wav]}"
SPEAKER="${2:?}"
OUT="${3:-${REPO}/out_cross_speaker.wav}"

"${PY}" tools/infer_flow_reconstruct_cross_speaker_s3tok25hz.py \
  --content_wav "${CONTENT}" \
  --speaker_wav "${SPEAKER}" \
  --out_wav "${OUT}" \
  --n_timesteps 20

echo "wrote ${OUT}"
