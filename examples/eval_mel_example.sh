#!/usr/bin/env bash
# Pred vs GT mel metrics (same protocol as self-prompt infer). Append one JSON line with --json_out.
set -euo pipefail

: "${COSYVOICE_ROOT:?}"
: "${FT_COSY:?}"

cd "${COSYVOICE_ROOT}"
export PYTHONPATH="${FT_COSY}:third_party/Matcha-TTS:$(pwd)"
PY="${PYTHON:-python3}"

WAV="${1:?wav path}"
JSONL="${2:-}"

args=( --wav "${WAV}" --preset custom_s3tok25hz --n_timesteps 20 )
[[ -n "${JSONL}" ]] && args+=( --json_out "${JSONL}" )

"${PY}" tools/eval_flow_reconstruct_mel.py "${args[@]}"
