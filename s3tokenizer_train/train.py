"""
S3Tokenizer V1 training script.

Usage:
  torchrun --nproc_per_node=2 -m s3tokenizer_train.train \
    --train_dir data/aishell_s3/train \
    --dev_dir data/aishell_s3/dev \
    --output_dir exp/s3tokenizer_v1
"""

import argparse
import logging
import os
import sys
import signal
import time
import math
from datetime import timedelta

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from torch.utils.tensorboard import SummaryWriter

from s3tokenizer_train.model import S3Config, WhisperWithVQ, build_model

logger = logging.getLogger(__name__)
IGNORE_INDEX = -100


def get_args():
    parser = argparse.ArgumentParser(description="Train S3Tokenizer V1")
    parser.add_argument("--train_dir", required=True, help="Training data dir with wav.scp/text")
    parser.add_argument("--dev_dir", required=True, help="Dev data dir with wav.scp/text")
    parser.add_argument("--output_dir", required=True, help="Output directory for checkpoints")
    parser.add_argument("--whisper_model", default="large-v3", help="Whisper model name")
    parser.add_argument("--whisper_cache", default=None, help="Whisper model cache directory")

    parser.add_argument("--n_encoder1_layers", type=int, default=6)
    parser.add_argument("--n_codebook_size", type=int, default=4096)
    parser.add_argument("--vq_decay", type=float, default=0.99)

    parser.add_argument("--lr_encoder1", type=float, default=1e-4)
    parser.add_argument("--lr_vq", type=float, default=1e-4)
    parser.add_argument("--lr_encoder2", type=float, default=1e-5)
    parser.add_argument("--lr_decoder", type=float, default=1e-5)
    parser.add_argument(
        "--lr_schedule",
        type=str,
        default="constant",
        choices=["constant", "cosine", "step"],
        help="LR schedule after warmup.",
    )
    parser.add_argument(
        "--cosine_min_lr_scale",
        type=float,
        default=0.1,
        help="Min LR scale for cosine schedule.",
    )
    parser.add_argument(
        "--step_decay_epochs",
        type=str,
        default="",
        help="Comma-separated epoch indices for step LR decay (e.g. '5,8').",
    )
    parser.add_argument(
        "--step_decay_gamma",
        type=float,
        default=0.5,
        help="Multiplicative factor for each step LR decay milestone.",
    )
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_steps", type=int, default=2000)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument(
        "--freeze_encoder2_decoder_epochs",
        type=int,
        default=0,
        help="Freeze encoder2+decoder for first N epochs, then unfreeze.",
    )

    parser.add_argument("--commit_loss_weight", type=float, default=0.1,
                        help="Weight for VQ commitment loss (monitoring, small)")

    parser.add_argument("--batch_size", type=int, default=4, help="Per-GPU batch size")
    parser.add_argument("--grad_accum_steps", type=int, default=1, help="Gradient accumulation steps")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--prefetch_factor", type=int, default=2)
    parser.add_argument("--persistent_workers", action="store_true", default=False)
    parser.add_argument("--use_torch_compile", action="store_true", default=False)
    parser.add_argument("--allow_tf32", action="store_true", default=False)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--log_interval", type=int, default=50)
    parser.add_argument("--save_interval_epochs", type=int, default=5)
    parser.add_argument(
        "--save_interval_steps",
        type=int,
        default=500,
        help="Save 'latest.pt' every N training steps (rank0). "
             "Prevents losing a whole epoch on preemption/kill.",
    )
    parser.add_argument("--eval_interval_epochs", type=int, default=1)
    parser.add_argument("--model_sample_rate", type=int, default=16000,
                        help="Input audio is resampled to this rate before Whisper mel extraction.")

    parser.add_argument("--resume", default=None, help="Checkpoint path to resume from")
    parser.add_argument("--use_amp", action="store_true", default=True)
    return parser.parse_args()


def setup_distributed():
    """Initialize DDP."""
    if "RANK" in os.environ:
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        dist.init_process_group("nccl", timeout=timedelta(minutes=30))
        torch.cuda.set_device(local_rank)
        return rank, local_rank, world_size
    else:
        return 0, 0, 1


def get_lr_with_warmup(step: int, warmup_steps: int, base_lr: float) -> float:
    """Linear warmup then constant."""
    if step < warmup_steps:
        return base_lr * step / max(1, warmup_steps)
    return base_lr


def _parse_step_milestones(raw: str):
    if not raw.strip():
        return []
    out = []
    for x in raw.split(","):
        x = x.strip()
        if x:
            out.append(int(x))
    return sorted(set(out))


def get_lr_with_schedule(
    step: int,
    warmup_steps: int,
    base_lr: float,
    lr_schedule: str,
    cosine_min_lr_scale: float,
    step_decay_epochs: list,
    step_decay_gamma: float,
    epoch: int,
    total_steps: int,
) -> float:
    # Warmup always applies first.
    if step < warmup_steps:
        return base_lr * step / max(1, warmup_steps)

    if lr_schedule == "constant":
        return base_lr

    if lr_schedule == "cosine":
        after_warmup = max(1, total_steps - warmup_steps)
        t = min(max(step - warmup_steps, 0), after_warmup)
        cos = 0.5 * (1.0 + math.cos(math.pi * t / after_warmup))
        scale = cosine_min_lr_scale + (1.0 - cosine_min_lr_scale) * cos
        return base_lr * scale

    # step schedule
    n_decay = sum(1 for m in step_decay_epochs if epoch >= m)
    return base_lr * (step_decay_gamma ** n_decay)


def evaluate(
    model: WhisperWithVQ,
    dataloader: DataLoader,
    device: torch.device,
    use_amp: bool,
) -> dict:
    """Run evaluation and return metrics."""
    model.eval()
    total_asr_loss = 0.0
    total_asr_tokens = 0
    total_commit = 0.0
    n_batches = 0
    all_vq_indices = []

    with torch.no_grad():
        for batch in dataloader:
            mel = batch["mel"].to(device)
            tokens = batch["tokens"].to(device)
            token_lens = batch["token_lens"].to(device)

            with torch.amp.autocast("cuda", enabled=use_amp, dtype=torch.bfloat16):
                # Decoder input: all tokens except last; target: all tokens except first
                dec_input = tokens[:, :-1]
                target = tokens[:, 1:].clone()
                # Keep true EOS supervised; mask only padded positions.
                max_tgt_len = target.shape[1]
                tgt_valid_len = (token_lens - 1).clamp(min=0, max=max_tgt_len)
                pos = torch.arange(max_tgt_len, device=device).unsqueeze(0)
                valid_mask = pos < tgt_valid_len.unsqueeze(1)
                target.masked_fill_(~valid_mask, IGNORE_INDEX)

                logits, commit_loss, vq_indices = model(mel, dec_input)

                # Whisper-style CE: ignore padded targets only.
                asr_loss = F.cross_entropy(
                    logits.reshape(-1, logits.size(-1)),
                    target.reshape(-1),
                    ignore_index=IGNORE_INDEX,
                )

            n_valid_asr = (target != IGNORE_INDEX).sum().item()
            total_asr_loss += asr_loss.item() * n_valid_asr
            total_asr_tokens += n_valid_asr
            total_commit += commit_loss.item()
            n_batches += 1
            all_vq_indices.append(vq_indices.cpu())

    avg_asr_loss = total_asr_loss / max(total_asr_tokens, 1)
    avg_commit = total_commit / max(n_batches, 1)

    # Codebook utilization
    all_indices = torch.cat(all_vq_indices, dim=0).flatten()
    unique_codes = all_indices.unique().numel()

    model.train()
    return {
        "loss": avg_asr_loss,
        "commit_loss": avg_commit,
        "ppl": 2 ** (avg_asr_loss / 0.6931),
        "codebook_usage": unique_codes,
    }


def save_checkpoint(model, optimizer, scaler, epoch, step, output_dir, name="checkpoint"):
    """Save model checkpoint."""
    rank = int(os.environ.get("RANK", 0))
    if rank != 0:
        return
    os.makedirs(output_dir, exist_ok=True)
    state = {
        "model": model.module.state_dict() if hasattr(model, "module") else model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict() if scaler is not None else None,
        "epoch": epoch,
        "step": step,
    }
    path = os.path.join(output_dir, f"{name}.pt")
    torch.save(state, path)
    logger.info(f"Saved checkpoint to {path}")


def install_signal_handlers():
    """
    Returns a dict-like state that becomes {'stop': True, 'sig': <signal>} once SIGTERM/SIGINT received.
    We don't exit inside the handler; we let the training loop checkpoint and exit cleanly.
    """
    state = {"stop": False, "sig": None}

    def _handler(sig, frame):  # noqa: ARG001
        state["stop"] = True
        state["sig"] = sig

    try:
        signal.signal(signal.SIGTERM, _handler)
        signal.signal(signal.SIGINT, _handler)
    except Exception:
        # Some environments may restrict signal handling.
        pass

    return state


def main():
    args = get_args()
    rank, local_rank, world_size = setup_distributed()
    device = torch.device(f"cuda:{local_rank}")
    is_main = rank == 0

    logging.basicConfig(
        level=logging.INFO if is_main else logging.WARNING,
        format=f"[Rank {rank}] %(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    if is_main:
        os.makedirs(args.output_dir, exist_ok=True)

    sig_state = install_signal_handlers()
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    # Build model
    config = S3Config(
        n_encoder1_layers=args.n_encoder1_layers,
        n_codebook_size=args.n_codebook_size,
        vq_decay=args.vq_decay,
        whisper_model=args.whisper_model,
    )
    model = build_model(config, whisper_cache_dir=args.whisper_cache)
    model = model.to(device)
    if args.use_torch_compile:
        model = torch.compile(model)

    # DDP
    if world_size > 1:
        use_find_unused = args.freeze_encoder2_decoder_epochs > 0
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=use_find_unused)
    raw_model = model.module if hasattr(model, "module") else model

    # Optimizer with differential LR
    param_groups = raw_model.get_param_groups(
        lr_encoder1=args.lr_encoder1,
        lr_vq=args.lr_vq,
        lr_encoder2=args.lr_encoder2,
        lr_decoder=args.lr_decoder,
    )
    optimizer = torch.optim.AdamW(param_groups, weight_decay=args.weight_decay)

    scaler = torch.amp.GradScaler("cuda") if args.use_amp else None

    # Resume
    start_epoch = 0
    global_step = 0
    if args.resume and os.path.exists(args.resume):
        logger.info(f"Resuming from {args.resume}")
        ckpt = torch.load(args.resume, map_location="cpu")
        raw_model.load_state_dict(ckpt["model"], strict=False)
        if "optimizer" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer"])
        if scaler is not None and ckpt.get("scaler"):
            scaler.load_state_dict(ckpt["scaler"])
        start_epoch = ckpt.get("epoch", 0)
        global_step = ckpt.get("step", 0)
        logger.info(f"Resumed at epoch {start_epoch}, step {global_step}")

    # Datasets
    from s3tokenizer_train.dataset import AishellS3Dataset, collate_fn

    train_dataset = AishellS3Dataset(args.train_dir, language="zh", model_sample_rate=args.model_sample_rate)
    dev_dataset = AishellS3Dataset(args.dev_dir, language="zh", model_sample_rate=args.model_sample_rate)

    train_sampler = DistributedSampler(train_dataset, shuffle=True) if world_size > 1 else None
    dev_sampler = DistributedSampler(dev_dataset, shuffle=False) if world_size > 1 else None

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        shuffle=(train_sampler is None),
        num_workers=args.num_workers,
        prefetch_factor=(args.prefetch_factor if args.num_workers > 0 else None),
        persistent_workers=(args.persistent_workers and args.num_workers > 0),
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=True,
    )
    dev_loader = DataLoader(
        dev_dataset,
        batch_size=args.batch_size,
        sampler=dev_sampler,
        shuffle=False,
        num_workers=args.num_workers,
        prefetch_factor=(args.prefetch_factor if args.num_workers > 0 else None),
        persistent_workers=(args.persistent_workers and args.num_workers > 0),
        collate_fn=collate_fn,
        pin_memory=True,
    )

    # Tensorboard
    writer = SummaryWriter(os.path.join(args.output_dir, "tensorboard")) if is_main else None

    logger.info(f"Training: {len(train_dataset)} utts, Dev: {len(dev_dataset)} utts")
    logger.info(f"Batch size: {args.batch_size} x {world_size} GPUs = {args.batch_size * world_size}")
    logger.info(f"Grad accum steps: {args.grad_accum_steps}")
    logger.info(
        "Effective global batch per optimizer step: %d",
        args.batch_size * world_size * max(1, args.grad_accum_steps),
    )
    logger.info(f"Epochs: {args.epochs}, Warmup: {args.warmup_steps} steps")
    logger.info(f"Loss: asr_ce + {args.commit_loss_weight:.3f} * commit_loss")
    logger.info(f"Model sample rate: {args.model_sample_rate} Hz")
    logger.info(f"LR schedule: {args.lr_schedule}")
    if args.lr_schedule == "cosine":
        logger.info(f"Cosine min LR scale: {args.cosine_min_lr_scale}")
    if args.lr_schedule == "step":
        logger.info(f"Step decay epochs: {args.step_decay_epochs}, gamma={args.step_decay_gamma}")
    logger.info(f"Freeze encoder2+decoder first epochs: {args.freeze_encoder2_decoder_epochs}")

    step_milestones = _parse_step_milestones(args.step_decay_epochs)
    total_steps = args.epochs * len(train_loader)
    freeze_prev = None

    # Training loop
    model.train()
    for epoch in range(start_epoch, args.epochs):
        freeze_now = epoch < args.freeze_encoder2_decoder_epochs
        if freeze_now != freeze_prev:
            for p in raw_model.encoder2_blocks.parameters():
                p.requires_grad = not freeze_now
            for p in raw_model.encoder2_ln.parameters():
                p.requires_grad = not freeze_now
            for p in raw_model.decoder.parameters():
                p.requires_grad = not freeze_now
            if is_main:
                logger.info(
                    "Stage switch: encoder2+decoder %s at epoch %d",
                    "FROZEN" if freeze_now else "UNFROZEN",
                    epoch,
                )
            freeze_prev = freeze_now

        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        epoch_asr_loss = 0.0
        epoch_asr_tokens = 0
        epoch_start = time.time()
        optimizer.zero_grad(set_to_none=True)

        for batch_idx, batch in enumerate(train_loader):
            if sig_state["stop"]:
                if is_main:
                    logger.warning(f"Received signal {sig_state['sig']}; saving latest and exiting...")
                    save_checkpoint(model, optimizer, scaler, epoch, global_step, args.output_dir, "latest")
                if world_size > 1:
                    try:
                        dist.barrier()
                    except Exception:
                        pass
                return
            mel = batch["mel"].to(device, non_blocking=True)
            tokens = batch["tokens"].to(device, non_blocking=True)
            token_lens = batch["token_lens"].to(device, non_blocking=True)

            # Adjust LR with warmup
            for pg in optimizer.param_groups:
                base_lr = {
                    "encoder1": args.lr_encoder1,
                    "vq": args.lr_vq,
                    "encoder2": args.lr_encoder2,
                    "decoder": args.lr_decoder,
                }.get(pg["name"], args.lr_encoder1)
                pg["lr"] = get_lr_with_schedule(
                    step=global_step,
                    warmup_steps=args.warmup_steps,
                    base_lr=base_lr,
                    lr_schedule=args.lr_schedule,
                    cosine_min_lr_scale=args.cosine_min_lr_scale,
                    step_decay_epochs=step_milestones,
                    step_decay_gamma=args.step_decay_gamma,
                    epoch=epoch,
                    total_steps=total_steps,
                )

            with torch.amp.autocast("cuda", enabled=args.use_amp, dtype=torch.bfloat16):
                dec_input = tokens[:, :-1]
                target = tokens[:, 1:].clone()
                max_tgt_len = target.shape[1]
                tgt_valid_len = (token_lens - 1).clamp(min=0, max=max_tgt_len)
                pos = torch.arange(max_tgt_len, device=device).unsqueeze(0)
                valid_mask = pos < tgt_valid_len.unsqueeze(1)
                target.masked_fill_(~valid_mask, IGNORE_INDEX)

                logits, commit_loss, vq_indices = model(mel, dec_input)

                asr_loss = F.cross_entropy(
                    logits.reshape(-1, logits.size(-1)),
                    target.reshape(-1),
                    ignore_index=IGNORE_INDEX,
                )
                loss = asr_loss + args.commit_loss_weight * commit_loss

            loss = loss / max(1, args.grad_accum_steps)
            if scaler is not None:
                scaler.scale(loss).backward()
            else:
                loss.backward()
            should_step = ((batch_idx + 1) % max(1, args.grad_accum_steps) == 0) or ((batch_idx + 1) == len(train_loader))
            if should_step:
                if scaler is not None:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

            n_valid_asr = (target != IGNORE_INDEX).sum().item()
            epoch_asr_loss += asr_loss.item() * n_valid_asr
            epoch_asr_tokens += n_valid_asr

            # Step-based checkpointing for preemption resilience
            if args.save_interval_steps > 0 and (global_step % args.save_interval_steps == 0):
                save_checkpoint(model, optimizer, scaler, epoch, global_step, args.output_dir, "latest")

            if is_main and global_step % args.log_interval == 0:
                avg_asr = epoch_asr_loss / max(epoch_asr_tokens, 1)
                lr_now = optimizer.param_groups[0]["lr"]
                unique_codes = vq_indices.unique().numel()
                logger.info(
                    f"Epoch {epoch} Step {global_step} | "
                    f"loss={asr_loss.item():.4f} avg={avg_asr:.4f} "
                    f"commit={commit_loss.item():.6f} "
                    f"cb_usage={unique_codes}/{config.n_codebook_size} "
                    f"lr={lr_now:.2e}"
                )
                if writer:
                    writer.add_scalar("train/loss", asr_loss.item(), global_step)
                    writer.add_scalar("train/commit_loss", commit_loss.item(), global_step)
                    writer.add_scalar("train/codebook_usage", unique_codes, global_step)
                    writer.add_scalar("train/lr", lr_now, global_step)

        epoch_time = time.time() - epoch_start
        epoch_avg_asr = epoch_asr_loss / max(epoch_asr_tokens, 1)
        if is_main:
            logger.info(
                f"Epoch {epoch} done in {epoch_time:.0f}s | avg_loss={epoch_avg_asr:.4f}"
            )

        # Evaluation
        if (epoch + 1) % args.eval_interval_epochs == 0:
            if is_main:
                dev_metrics = evaluate(
                    raw_model,
                    dev_loader,
                    device,
                    args.use_amp,
                )
                logger.info(
                    f"[Eval] Epoch {epoch} | loss={dev_metrics['loss']:.4f} "
                    f"ppl={dev_metrics['ppl']:.2f} "
                    f"commit={dev_metrics['commit_loss']:.6f} "
                    f"cb_usage={dev_metrics['codebook_usage']}/{config.n_codebook_size}"
                )
                if writer:
                    for k, v in dev_metrics.items():
                        writer.add_scalar(f"eval/{k}", v, global_step)

        # Save checkpoint
        if (epoch + 1) % args.save_interval_epochs == 0:
            save_checkpoint(model, optimizer, scaler, epoch + 1, global_step,
                            args.output_dir, f"epoch_{epoch+1}")

        # Always save latest
        save_checkpoint(model, optimizer, scaler, epoch + 1, global_step,
                        args.output_dir, "latest")

    # Save final
    save_checkpoint(model, optimizer, scaler, args.epochs, global_step,
                    args.output_dir, "final")

    if is_main:
        logger.info("Training complete!")
        if writer:
            writer.close()

    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
