#!/bin/bash
set -euo pipefail

# Watchdog for causal S3Tokenizer training.
# - Keeps training running under external SIGTERM interruptions.
# - Always resumes from latest checkpoint when available.

WORK_DIR="/mnt/data/ft_cosy"
OUT_DIR="$WORK_DIR/exp/s3tokenizer_causal_streaming"
LOG="$OUT_DIR/watchdog.log"
LOCK="$OUT_DIR/watchdog.lock"
TRAIN_LOG="$OUT_DIR/train.log"

export PYTHONPATH="$WORK_DIR:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

mkdir -p "$OUT_DIR"

exec 9>"$LOCK"
if ! flock -n 9; then
  echo "[$(date -u +'%F %T')] causal watchdog already running" | tee -a "$LOG"
  exit 0
fi

echo "[$(date -u +'%F %T')] causal watchdog started" | tee -a "$LOG"

MASTER_PORT="${MASTER_PORT:-29688}"

start_train() {
  local resume_args=()
  if [[ -f "$OUT_DIR/latest.pt" ]]; then
    resume_args=(--resume "exp/s3tokenizer_causal_streaming/latest.pt")
    echo "[$(date -u +'%F %T')] resume from latest.pt" | tee -a "$LOG"
  else
    echo "[$(date -u +'%F %T')] start from scratch" | tee -a "$LOG"
  fi

  cd "$WORK_DIR"
  echo "[$(date -u +'%F %T')] launching torchrun (port=$MASTER_PORT)" | tee -a "$LOG"
  nohup torchrun --nproc_per_node=2 --master_port="$MASTER_PORT" \
    -m s3tokenizer_train.train \
    --train_dir data/aishell_s3/train \
    --dev_dir data/aishell_s3/dev \
    --output_dir exp/s3tokenizer_causal_streaming \
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
    --use_amp \
    "${resume_args[@]}" \
    >> "$TRAIN_LOG" 2>&1 &
}

while true; do
  if pgrep -af "s3tokenizer_train\\.train.*exp/s3tokenizer_causal_streaming|s3tokenizer_train\\.train .*--output_dir exp/s3tokenizer_causal_streaming" >/dev/null 2>&1; then
    echo "[$(date -u +'%F %T')] causal training running" >> "$LOG"
  else
    start_train
  fi
  sleep 30
done

