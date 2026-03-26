#!/usr/bin/env bash
# Example: fine-tune Flow on causal-token parquet with official CosyVoice-300M flow.pt init.
# Edit COSYVOICE_ROOT, FT_COSY, DATA_LIST paths, MODEL_DIR, and GPU count before running.
set -euo pipefail

COSYVOICE_ROOT="${COSYVOICE_ROOT:-/path/to/CosyVoice-main}"
FT_COSY="${FT_COSY:-/path/to/ft_cosy_workspace}"
export PYTHONPATH="${FT_COSY}:${COSYVOICE_ROOT}:${COSYVOICE_ROOT}/third_party/Matcha-TTS"

PY="${PYTHON:-python3}"
FLOW_PT="${COSYVOICE_ROOT}/pretrained_models/CosyVoice-300M/flow.pt"
CONFIG="${COSYVOICE_ROOT}/examples/libritts/cosyvoice/conf/cosyvoice_aishell_s3tok1024_25hz.yaml"
TRAIN_LIST="${FT_COSY}/data/aishell_s3_causal_flow/train.data.list"
CV_LIST="${FT_COSY}/data/aishell_s3_causal_flow/dev.data.list"
ONNX_BASE="${ONNX_BASE:-/path/to/marco_voice}"   # directory containing campplus.onnx (CosyVoice train uses --onnx_path)

MODEL_DIR="${COSYVOICE_ROOT}/exp/my_causal_flow_run/flow/torch_ddp"
TB_DIR="${COSYVOICE_ROOT}/tensorboard/my_causal_flow_run/flow/torch_ddp"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
NPROC="${NPROC:-1}"
RDZV_PORT="${RDZV_PORT:-12622}"

cd "${COSYVOICE_ROOT}"

"${PY}" -m torch.distributed.run --nnodes=1 --nproc_per_node="${NPROC}" \
  --rdzv_id=42 --rdzv_backend=c10d --rdzv_endpoint="localhost:${RDZV_PORT}" \
  cosyvoice/bin/train.py \
  --train_engine torch_ddp \
  --config "${CONFIG}" \
  --train_data "${TRAIN_LIST}" \
  --cv_data "${CV_LIST}" \
  --model flow \
  --model_dir "${MODEL_DIR}" \
  --tensorboard_dir "${TB_DIR}" \
  --checkpoint "${FLOW_PT}" \
  --onnx_path "${ONNX_BASE}" \
  --num_workers 8 \
  --prefetch 100 \
  --pin_memory \
  --use_amp
