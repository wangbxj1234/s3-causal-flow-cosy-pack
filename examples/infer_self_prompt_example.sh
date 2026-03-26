#!/usr/bin/env bash
# Run from anywhere; uses this repo root (directory containing cosyvoice/, tools/, conf/).
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO"
export PYTHONPATH="${REPO}:${REPO}/third_party/Matcha-TTS"
PY="${PYTHON:-python3}"

WAV="${1:?usage: $0 <input.wav> [out.wav]}"
OUT="${2:-${REPO}/out_self_prompt.wav}"

"${PY}" tools/infer_flow_reconstruct_s3tok25hz.py \
  --wav "${WAV}" \
  --out_wav "${OUT}" \
  --n_timesteps 20

echo "wrote ${OUT}"
