#!/usr/bin/env bash
# Train causal S3 tokenizer.
# Usage:
#   bash examples/train_tokenizer_causal_example.sh
set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO"
export PYTHONPATH="${REPO}:${REPO}/third_party/Matcha-TTS"
PY="${PYTHON:-python3}"

TRAIN_DIR="${REPO}/data/aishell_s3/train"
DEV_DIR="${REPO}/data/aishell_s3/dev"
OUT_DIR="${REPO}/exp/s3tokenizer_causal_streaming"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
NPROC="${NPROC:-1}"
MASTER_PORT="${MASTER_PORT:-29688}"

"${PY}" -m torch.distributed.run --nproc_per_node="${NPROC}" --master_port="${MASTER_PORT}" \
  -m s3tokenizer_train.train \
  --train_dir "${TRAIN_DIR}" \
  --dev_dir "${DEV_DIR}" \
  --output_dir "${OUT_DIR}" \
  --whisper_model large-v3 \
  --whisper_cache /tmp/whisper_check \
  --n_encoder1_layers 6 \
  --n_codebook_size 1024 \
  --vq_decay 0.99 \
  --lr_encoder1 1e-4 \
  --lr_vq 1e-4 \
  --lr_encoder2 1e-5 \
  --lr_decoder 1e-5 \
  --weight_decay 0.01 \
  --warmup_steps 2000 \
  --max_grad_norm 1.0 \
  --commit_loss_weight 0.1 \
  --batch_size 16 \
  --grad_accum_steps 2 \
  --num_workers 8 \
  --prefetch_factor 4 \
  --persistent_workers \
  --allow_tf32 \
  --epochs 50 \
  --log_interval 25 \
  --save_interval_epochs 5 \
  --save_interval_steps 500 \
  --eval_interval_epochs 1 \
  --model_sample_rate 16000 \
  --use_amp
