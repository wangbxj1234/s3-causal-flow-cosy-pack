#!/usr/bin/env bash
# Cross-speaker: content wav for tokens+mel; separate wav for CAMPPlus embedding only.
# Prerequisite: copy scripts/cosyvoice_tools/*.py into ${COSYVOICE_ROOT}/tools/
set -euo pipefail

: "${COSYVOICE_ROOT:?set COSYVOICE_ROOT to CosyVoice-main absolute path}"
: "${FT_COSY:?set FT_COSY to workspace root containing s3tokenizer_train + data}"

cd "${COSYVOICE_ROOT}"
export PYTHONPATH="${FT_COSY}:third_party/Matcha-TTS:$(pwd)"
PY="${PYTHON:-python3}"

CONTENT="${1:-${FT_COSY}/recon_demos/source_content_S0736W0371.wav}"
SPEAKER="${2:-${FT_COSY}/recon_demos/source_speaker_S0724W0121.wav}"
OUT="${3:-${FT_COSY}/recon_demos/out_cross_speaker.wav}"

"${PY}" tools/infer_flow_reconstruct_cross_speaker_s3tok25hz.py \
  --content_wav "${CONTENT}" \
  --speaker_wav "${SPEAKER}" \
  --out_wav "${OUT}" \
  --n_timesteps 20

echo "wrote ${OUT}"
