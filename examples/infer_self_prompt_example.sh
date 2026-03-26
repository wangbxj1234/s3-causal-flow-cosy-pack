#!/usr/bin/env bash
# Self-prompt flow reconstruction (same wav: tokens + mel + spk emb).
# Prerequisite: copy scripts/cosyvoice_tools/*.py into ${COSYVOICE_ROOT}/tools/ (or keep both in tools/).
set -euo pipefail

: "${COSYVOICE_ROOT:?set COSYVOICE_ROOT to CosyVoice-main absolute path}"
: "${FT_COSY:?set FT_COSY to workspace root containing s3tokenizer_train + data}"

cd "${COSYVOICE_ROOT}"
export PYTHONPATH="${FT_COSY}:third_party/Matcha-TTS:$(pwd)"
PY="${PYTHON:-python3}"

WAV="${1:-${FT_COSY}/recon_demos/source_content_S0736W0371.wav}"
OUT="${2:-${FT_COSY}/recon_demos/out_self_prompt.wav}"

"${PY}" tools/infer_flow_reconstruct_s3tok25hz.py \
  --wav "${WAV}" \
  --out_wav "${OUT}" \
  --n_timesteps 20

echo "wrote ${OUT}"
