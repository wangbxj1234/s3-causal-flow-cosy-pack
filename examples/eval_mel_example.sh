#!/usr/bin/env bash
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO"
export PYTHONPATH="${REPO}:${REPO}/third_party/Matcha-TTS"
export COSYVOICE_S3_TOKENIZER_PT="${REPO}/pretrained_weights/s3tokenizer.pt"
PY="${PYTHON:-python3}"

WAV="${1:?usage: $0 <wav> [metrics.jsonl]}"
JSONL="${2:-}"

args=( --wav "${WAV}" --preset custom_s3tok25hz --n_timesteps 20 )
[[ -n "${JSONL}" ]] && args+=( --json_out "${JSONL}" )

"${PY}" tools/eval_flow_reconstruct_mel.py "${args[@]}"
