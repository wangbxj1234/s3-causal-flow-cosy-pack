#!/usr/bin/env bash
set -euo pipefail

ROOT="/mnt/data/ft_cosy/CosyVoice-main"
DATA_ROOT="/mnt/data/ft_cosy/data/aishell_s3"
TRAIN_DIR="${DATA_ROOT}/train"
DEV_DIR="${DATA_ROOT}/dev"
OUT_TAG="cosyvoice1_flow_s3tok_step2c_$(date +%Y%m%d_%H%M%S)"
MODEL_DIR="${ROOT}/exp/${OUT_TAG}/flow/torch_ddp"
TB_DIR="${ROOT}/tensorboard/${OUT_TAG}/flow/torch_ddp"
EXPORT_TOK_PT="/mnt/data/ft_cosy/exp/step2_c_commit005/s3tokenizer_epoch10.pt"
SRC_TOK_CKPT="/mnt/data/ft_cosy/exp/step2_c_commit005/epoch_10.pt"

# dependencies from local workspace
export PYTHONPATH="/mnt/data/ft_cosy:${ROOT}"
PYTHON_BIN="python3"
if [ -x "/mnt/data/Marco-Voice-main/marco/bin/python" ]; then
  PYTHON_BIN="/mnt/data/Marco-Voice-main/marco/bin/python"
fi

echo "[1/8] prepare aishell spk maps"
"${PYTHON_BIN}" - <<'PY'
import os, re
for split in ["train", "dev"]:
    d = f"/mnt/data/ft_cosy/data/aishell_s3/{split}"
    wav = os.path.join(d, "wav.scp")
    out = os.path.join(d, "utt2spk")
    pat = re.compile(r"^(.*?S\d+)")
    with open(wav, "r", encoding="utf-8") as f, open(out, "w", encoding="utf-8") as g:
        for line in f:
            utt = line.strip().split()[0]
            m = pat.match(utt)
            spk = m.group(1) if m else utt
            g.write(f"{utt} {spk}\n")
    print(f"wrote {out}")
PY

echo "[2/8] export s3 tokenizer from checkpoint"
"${PYTHON_BIN}" -m s3tokenizer_train.export \
  --checkpoint "${SRC_TOK_CKPT}" \
  --output "${EXPORT_TOK_PT}" \
  --n_encoder1_layers 6 \
  --n_codebook_size 1024

echo "[3/8] extract campplus embeddings"
for split in train dev; do
  "${PYTHON_BIN}" "${ROOT}/tools/extract_embedding.py" \
    --dir "${DATA_ROOT}/${split}" \
    --onnx_path "/mnt/data/Marco-Voice-main/pretrained_models/marco_voice/marco_voice/campplus.onnx" \
    --num_thread 16
done

echo "[4/8] extract speech tokens with your s3 tokenizer"
for split in train dev; do
  "${PYTHON_BIN}" "${ROOT}/tools/extract_speech_token_s3.py" \
    --dir "${DATA_ROOT}/${split}" \
    --tokenizer_pt "${EXPORT_TOK_PT}" \
    --device cuda
done

echo "[5/8] build parquet"
for split in train dev; do
  mkdir -p "${DATA_ROOT}/${split}/parquet"
  "${PYTHON_BIN}" "${ROOT}/tools/make_parquet_list.py" \
    --num_utts_per_parquet 1000 \
    --num_processes 16 \
    --src_dir "${DATA_ROOT}/${split}" \
    --des_dir "${DATA_ROOT}/${split}/parquet"
done

echo "[6/8] build train/dev data.list"
cat "${TRAIN_DIR}/parquet/data.list" > "${DATA_ROOT}/train.data.list"
cat "${DEV_DIR}/parquet/data.list" > "${DATA_ROOT}/dev.data.list"

echo "[7/8] generate 1024/25hz flow config"
"${PYTHON_BIN}" - <<'PY'
from pathlib import Path
src = Path("/mnt/data/ft_cosy/CosyVoice-main/examples/libritts/cosyvoice/conf/cosyvoice.yaml")
dst = Path("/mnt/data/ft_cosy/CosyVoice-main/examples/libritts/cosyvoice/conf/cosyvoice_aishell_s3tok1024_25hz.yaml")
txt = src.read_text(encoding="utf-8")
txt = txt.replace("speech_token_size: 4096", "speech_token_size: 1024")
txt = txt.replace("vocab_size: 4096", "vocab_size: 1024")
txt = txt.replace("input_frame_rate: 50", "input_frame_rate: 25")
txt = txt.replace("language: 'en'", "language: 'zh'")
dst.write_text(txt, encoding="utf-8")
print(f"wrote {dst}")
PY

echo "[8/8] launch flow training on 2xL20X"
cd "${ROOT}"
export CUDA_VISIBLE_DEVICES=0,1
NPROC=2
"${PYTHON_BIN}" -m torch.distributed.run --nnodes=1 --nproc_per_node="${NPROC}" \
  --rdzv_id=2099 --rdzv_backend=c10d --rdzv_endpoint=localhost:12477 \
  cosyvoice/bin/train.py \
  --train_engine torch_ddp \
  --config examples/libritts/cosyvoice/conf/cosyvoice_aishell_s3tok1024_25hz.yaml \
  --train_data "${DATA_ROOT}/train.data.list" \
  --cv_data "${DATA_ROOT}/dev.data.list" \
  --model flow \
  --model_dir "${MODEL_DIR}" \
  --tensorboard_dir "${TB_DIR}" \
  --ddp.dist_backend nccl \
  --num_workers 8 \
  --prefetch 100 \
  --pin_memory \
  --use_amp \
  --deepspeed_config examples/libritts/cosyvoice/conf/ds_stage2.json \
  --deepspeed.save_states model+optimizer
