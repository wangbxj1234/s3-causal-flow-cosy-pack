#!/usr/bin/env bash
# Example: Flow fine-tune from official CosyVoice-300M flow.pt (same conf as inference).
# Edit TRAIN_LIST, CV_LIST, ONNX_PATH, NPROC before running. Requires full CosyVoice training deps (deepspeed optional).
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
export PYTHONPATH="${REPO}:${REPO}/third_party/Matcha-TTS"

PY="${PYTHON:-python3}"
FLOW_PT="${REPO}/pretrained_weights/CosyVoice-300M/flow.pt"
CONFIG="${REPO}/conf/cosyvoice_aishell_s3tok1024_25hz.yaml"
TRAIN_LIST="${REPO}/data/aishell_s3_causal_flow/train.data.list"
CV_LIST="${REPO}/data/aishell_s3_causal_flow/dev.data.list"
ONNX_PATH="${REPO}/pretrained_weights/CosyVoice-300M"

MODEL_DIR="${REPO}/exp/my_causal_flow/flow/torch_ddp"
TB_DIR="${REPO}/tensorboard/my_causal_flow/flow/torch_ddp"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
NPROC="${NPROC:-1}"
RDZV_PORT="${RDZV_PORT:-12622}"

cd "${REPO}"

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
  --onnx_path "${ONNX_PATH}" \
  --num_workers 8 \
  --prefetch 100 \
  --pin_memory \
  --use_amp
