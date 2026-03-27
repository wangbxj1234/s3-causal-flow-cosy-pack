#!/usr/bin/env bash
# Export causal tokenizer, extract embeddings/tokens, build parquet and data.list.
# Usage:
#   bash examples/prepare_flow_data_from_causal_tokenizer_example.sh
set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO"
export PYTHONPATH="${REPO}:${REPO}/third_party/Matcha-TTS"
PY="${PYTHON:-python3}"

TOK_CKPT="${REPO}/exp/s3tokenizer_causal_streaming/latest.pt"
TOK_EXPORT="${REPO}/pretrained_weights/s3tokenizer.pt"
DATA_ROOT="${REPO}/data/aishell_s3_causal_flow"
ONNX_PATH="${REPO}/pretrained_weights/CosyVoice-300M/campplus.onnx"

if [ ! -f "${TOK_CKPT}" ]; then
  echo "Missing tokenizer checkpoint: ${TOK_CKPT}"
  exit 1
fi

for split in train dev; do
  if [ ! -f "${DATA_ROOT}/${split}/wav.scp" ]; then
    echo "Missing ${DATA_ROOT}/${split}/wav.scp"
    exit 1
  fi
done

echo "[1/5] export causal tokenizer"
"${PY}" -m s3tokenizer_train.export \
  --checkpoint "${TOK_CKPT}" \
  --output "${TOK_EXPORT}" \
  --n_encoder1_layers 6 \
  --n_codebook_size 1024

echo "[2/5] extract embeddings"
for split in train dev; do
  "${PY}" tools/extract_embedding.py \
    --dir "${DATA_ROOT}/${split}" \
    --onnx_path "${ONNX_PATH}" \
    --num_thread 16
done

echo "[3/5] extract speech tokens"
for split in train dev; do
  "${PY}" tools/extract_speech_token_s3.py \
    --dir "${DATA_ROOT}/${split}" \
    --tokenizer_pt "${TOK_EXPORT}" \
    --device cuda \
    --batch_size 32
done

echo "[4/5] make parquet"
for split in train dev; do
  mkdir -p "${DATA_ROOT}/${split}/parquet"
  "${PY}" tools/make_parquet_list.py \
    --src_dir "${DATA_ROOT}/${split}" \
    --des_dir "${DATA_ROOT}/${split}/parquet" \
    --num_utts_per_parquet 1000 \
    --num_processes 16
done

echo "[5/5] refresh data.list"
cp -f "${DATA_ROOT}/train/parquet/data.list" "${DATA_ROOT}/train.data.list"
cp -f "${DATA_ROOT}/dev/parquet/data.list" "${DATA_ROOT}/dev.data.list"
echo "done: ${DATA_ROOT}/train.data.list ${DATA_ROOT}/dev.data.list"
