# Copyright (c) 2021 Mobvoi Inc. (authors: Binbin Zhang)
#               2023 Horizon Inc. (authors: Xingchen Song)
#               2024 Alibaba Inc (authors: Xiang Lyu)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import os
import torch
import json
import re
import datetime
import yaml

try:
    import deepspeed
    from deepspeed.runtime.zero.stage_1_and_2 import estimate_zero2_model_states_mem_needs_all_live
except Exception:  # pragma: no cover - runtime env dependent
    deepspeed = None

    def estimate_zero2_model_states_mem_needs_all_live(*args, **kwargs):
        raise RuntimeError('deepspeed is unavailable in this environment')
import torch.optim as optim
import torch.distributed as dist

from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_

from cosyvoice.dataset.dataset import Dataset
from cosyvoice.utils.scheduler import WarmupLR, NoamHoldAnnealing, ConstantLR

_INF_GRAD_WARN_COUNT = 0


def _grad_norm_is_finite(grad_norm) -> bool:
    t = grad_norm if isinstance(grad_norm, torch.Tensor) else torch.as_tensor(
        grad_norm, dtype=torch.float32
    )
    return bool(torch.isfinite(t).all().item())


def _ddp_all_ranks_ok_to_optimizer_step(local_ok: bool, model, train_engine: str) -> bool:
    """Keep optimizer.step() in lockstep across ranks under DDP.

    If one rank skips scaler.step/optimizer.step due to inf/nan grad while another
    still steps, weights diverge and the next backward's allreduce can hang forever.
    """
    if train_engine != 'torch_ddp':
        return local_ok
    if not (dist.is_available() and dist.is_initialized()):
        return local_ok
    if dist.get_world_size() <= 1:
        return local_ok
    try:
        device = next(model.parameters()).device
    except StopIteration:
        return local_ok
    if device.type != 'cuda':
        return local_ok
    flag = torch.tensor(1 if local_ok else 0, device=device, dtype=torch.int32)
    dist.all_reduce(flag, op=dist.ReduceOp.MIN)
    return bool(flag.item() == 1)


def _warn_infinite_grad_norm() -> None:
    global _INF_GRAD_WARN_COUNT
    _INF_GRAD_WARN_COUNT += 1
    if _INF_GRAD_WARN_COUNT <= 3 or _INF_GRAD_WARN_COUNT % 100 == 0:
        logging.warning(
            'get infinite grad_norm, check your code/data if it appears frequently (count=%d)',
            _INF_GRAD_WARN_COUNT,
        )


def _should_log_current_rank(info_dict=None) -> bool:
    rank = int(os.environ.get('RANK', 0))
    if info_dict is None:
        return rank == 0
    return bool(info_dict.get('log_all_ranks', False) or rank == 0)


def init_distributed(args):
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    rank = int(os.environ.get('RANK', 0))
    logging.info('training on multiple gpus, this gpu {}'.format(local_rank) +
                 ', rank {}, world_size {}'.format(rank, world_size))
    if args.train_engine == 'torch_ddp':
        torch.cuda.set_device(local_rank)
        pg_timeout = datetime.timedelta(seconds=getattr(args, 'ddp_timeout', 3600))
        dist.init_process_group(args.dist_backend, timeout=pg_timeout)
    else:
        if deepspeed is None:
            raise RuntimeError('deepspeed is unavailable in this environment')
        deepspeed.init_distributed(dist_backend=args.dist_backend)
    return world_size, local_rank, rank


def init_dataset_and_dataloader(args, configs, gan, dpo):
    data_pipeline = configs['data_pipeline_gan'] if gan is True else configs['data_pipeline']
    train_dataset = Dataset(args.train_data, data_pipeline=data_pipeline, mode='train', gan=gan, dpo=dpo, shuffle=True, partition=True)
    cv_dataset = Dataset(args.cv_data, data_pipeline=data_pipeline, mode='dev', gan=gan, dpo=dpo, shuffle=False, partition=False)

    # do not use persistent_workers=True, as whisper tokenizer opens tiktoken file each time when the for loop starts
    train_data_loader = DataLoader(train_dataset,
                                   batch_size=None,
                                   pin_memory=args.pin_memory,
                                   num_workers=args.num_workers,
                                   prefetch_factor=args.prefetch)
    cv_data_loader = DataLoader(cv_dataset,
                                batch_size=None,
                                pin_memory=args.pin_memory,
                                num_workers=args.num_workers,
                                prefetch_factor=args.prefetch)
    return train_dataset, cv_dataset, train_data_loader, cv_data_loader


def check_modify_and_save_config(args, configs):
    if args.train_engine == "torch_ddp":
        configs['train_conf']["dtype"] = 'bf16' if args.use_amp is True else 'fp32'
    else:
        if deepspeed is None:
            raise RuntimeError('deepspeed is unavailable in this environment')
        with open(args.deepspeed_config, 'r') as fin:
            ds_configs = json.load(fin)
        if "fp16" in ds_configs and ds_configs["fp16"]["enabled"]:
            configs['train_conf']["dtype"] = "fp16"
        elif "bf16" in ds_configs and ds_configs["bf16"]["enabled"]:
            configs['train_conf']["dtype"] = "bf16"
        else:
            configs['train_conf']["dtype"] = "fp32"
        assert ds_configs["train_micro_batch_size_per_gpu"] == 1
        # if use deepspeed, override ddp config
        configs['train_conf']['save_per_step'] = int(configs['train_conf']['save_per_step'] *
                                                     configs['train_conf']['accum_grad'] / ds_configs["gradient_accumulation_steps"])
        configs['train_conf']['accum_grad'] = ds_configs["gradient_accumulation_steps"]
        configs['train_conf']['grad_clip'] = ds_configs["gradient_clipping"]
        configs['train_conf']['log_interval'] = ds_configs["steps_per_print"]
    # Keep new runtime diagnostics backward compatible with older yaml.
    configs['train_conf'].setdefault('join_check_interval', getattr(args, 'join_check_interval', 50))
    configs['train_conf'].setdefault('slow_batch_sec', getattr(args, 'slow_batch_sec', 120.0))
    configs['train_conf'].setdefault('log_all_ranks', getattr(args, 'log_all_ranks', False))
    configs['train_conf'].setdefault('max_feat_len', getattr(args, 'max_feat_len', 12000))
    configs['train_conf'].setdefault('max_batches_per_epoch', getattr(args, 'max_batches_per_epoch', -1))
    return configs


def wrap_cuda_model(args, model):
    local_world_size = int(os.environ.get('LOCAL_WORLD_SIZE', 1))
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    if args.train_engine == "torch_ddp":  # native pytorch ddp
        assert (torch.cuda.is_available())
        model.cuda()
        # Flow training does not use conditional parameter branches, so disabling
        # unused-parameter traversal reduces per-step overhead and straggler risk.
        model = torch.nn.parallel.DistributedDataParallel(model, find_unused_parameters=False)
    else:
        if deepspeed is None:
            raise RuntimeError('deepspeed is unavailable in this environment')
        if int(os.environ.get('RANK', 0)) == 0:
            logging.info("Estimating model states memory needs (zero2)...")
            estimate_zero2_model_states_mem_needs_all_live(
                model,
                num_gpus_per_node=local_world_size,
                num_nodes=world_size // local_world_size)
    return model


def init_optimizer_and_scheduler(args, configs, model, gan):
    if gan is False:
        if configs['train_conf']['optim'] == 'adam':
            optimizer = optim.Adam(model.parameters(), **configs['train_conf']['optim_conf'])
        elif configs['train_conf']['optim'] == 'adamw':
            optimizer = optim.AdamW(model.parameters(), **configs['train_conf']['optim_conf'])
        else:
            raise ValueError("unknown optimizer: " + configs['train_conf'])

        if configs['train_conf']['scheduler'] == 'warmuplr':
            scheduler_type = WarmupLR
            scheduler = WarmupLR(optimizer, **configs['train_conf']['scheduler_conf'])
        elif configs['train_conf']['scheduler'] == 'NoamHoldAnnealing':
            scheduler_type = NoamHoldAnnealing
            scheduler = NoamHoldAnnealing(optimizer, **configs['train_conf']['scheduler_conf'])
        elif configs['train_conf']['scheduler'] == 'constantlr':
            scheduler_type = ConstantLR
            scheduler = ConstantLR(optimizer)
        else:
            raise ValueError("unknown scheduler: " + configs['train_conf'])

        # use deepspeed optimizer for speedup
        if args.train_engine == "deepspeed":
            if deepspeed is None:
                raise RuntimeError('deepspeed is unavailable in this environment')
            def scheduler(opt):
                return scheduler_type(opt, **configs['train_conf']['scheduler_conf'])
            model, optimizer, _, scheduler = deepspeed.initialize(
                args=args,
                model=model,
                optimizer=None,
                lr_scheduler=scheduler,
                model_parameters=model.parameters())

        optimizer_d, scheduler_d = None, None

    else:
        # currently we wrap generator and discriminator in one model, so we cannot use deepspeed
        if configs['train_conf']['optim'] == 'adam':
            optimizer = optim.Adam(model.module.generator.parameters(), **configs['train_conf']['optim_conf'])
        elif configs['train_conf']['optim'] == 'adamw':
            optimizer = optim.AdamW(model.module.generator.parameters(), **configs['train_conf']['optim_conf'])
        else:
            raise ValueError("unknown optimizer: " + configs['train_conf'])

        if configs['train_conf']['scheduler'] == 'warmuplr':
            scheduler_type = WarmupLR
            scheduler = WarmupLR(optimizer, **configs['train_conf']['scheduler_conf'])
        elif configs['train_conf']['scheduler'] == 'NoamHoldAnnealing':
            scheduler_type = NoamHoldAnnealing
            scheduler = NoamHoldAnnealing(optimizer, **configs['train_conf']['scheduler_conf'])
        elif configs['train_conf']['scheduler'] == 'constantlr':
            scheduler_type = ConstantLR
            scheduler = ConstantLR(optimizer)
        else:
            raise ValueError("unknown scheduler: " + configs['train_conf'])

        if configs['train_conf']['optim_d'] == 'adam':
            optimizer_d = optim.Adam(model.module.discriminator.parameters(), **configs['train_conf']['optim_conf_d'])
        elif configs['train_conf']['optim_d'] == 'adamw':
            optimizer_d = optim.AdamW(model.module.discriminator.parameters(), **configs['train_conf']['optim_conf_d'])
        else:
            raise ValueError("unknown optimizer: " + configs['train_conf'])

        if configs['train_conf']['scheduler_d'] == 'warmuplr':
            scheduler_type = WarmupLR
            scheduler_d = WarmupLR(optimizer_d, **configs['train_conf']['scheduler_d'])
        elif configs['train_conf']['scheduler_d'] == 'NoamHoldAnnealing':
            scheduler_type = NoamHoldAnnealing
            scheduler_d = NoamHoldAnnealing(optimizer_d, **configs['train_conf']['scheduler_d'])
        elif configs['train_conf']['scheduler'] == 'constantlr':
            scheduler_type = ConstantLR
            scheduler_d = ConstantLR(optimizer_d)
        else:
            raise ValueError("unknown scheduler: " + configs['train_conf'])
    return model, optimizer, scheduler, optimizer_d, scheduler_d


def init_summarywriter(args):
    writer = None
    if int(os.environ.get('RANK', 0)) == 0:
        os.makedirs(args.model_dir, exist_ok=True)
        writer = SummaryWriter(args.tensorboard_dir)
    return writer


def save_model(model, model_name, info_dict):
    rank = int(os.environ.get('RANK', 0))
    model_dir = info_dict["model_dir"]
    save_model_path = os.path.join(model_dir, '{}.pt'.format(model_name))

    if info_dict["train_engine"] == "torch_ddp":
        if rank == 0:
            torch.save({**model.module.state_dict(), 'epoch': info_dict['epoch'], 'step': info_dict['step']}, save_model_path)
    else:
        with torch.no_grad():
            model.save_checkpoint(save_dir=model_dir,
                                  tag=model_name,
                                  client_state=info_dict)
    if rank == 0:
        info_path = re.sub('.pt$', '.yaml', save_model_path)
        info_dict['save_time'] = datetime.datetime.now().strftime('%d/%m/%Y %H:%M:%S')
        with open(info_path, 'w') as fout:
            data = yaml.dump(info_dict)
            fout.write(data)
        logging.info('[Rank {}] Checkpoint: save to checkpoint {}'.format(rank, save_model_path))


def cosyvoice_join(group_join, info_dict):
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    rank = int(os.environ.get('RANK', 0))

    join_check_interval = max(1, int(info_dict.get('join_check_interval', 50)))
    batch_idx = int(info_dict["batch_idx"])
    if batch_idx != 0 and batch_idx % join_check_interval == 0:
        # we try to join all rank in both ddp and deepspeed mode, in case different rank has different lr
        try:
            # Pass an explicit timeout so monitored_barrier doesn't inherit the
            # (potentially very long) group creation timeout.  This keeps the
            # recovery fast when one rank has fewer batches than the other.
            barrier_timeout = datetime.timedelta(seconds=int(info_dict.get('timeout', 60)))
            dist.monitored_barrier(group=group_join, timeout=barrier_timeout)
            return False
        except RuntimeError as e:
            logging.info(
                "Detected uneven workload distribution at batch_idx=%s: %s\n"
                "Break current worker to manually join all workers, world_size %s, current rank %s, current local_rank %s\n"
                "Last batch diagnostics: train_batch_cost_sec=%s, num_utts=%s, sample_utts=%s",
                batch_idx, e, world_size, rank, local_rank,
                info_dict.get('last_batch_cost_sec', 'n/a'),
                info_dict.get('last_batch_num_utts', 'n/a'),
                info_dict.get('last_batch_sample_utts', 'n/a'),
            )
            return True
    else:
        return False


def batch_forward(model, batch, scaler, info_dict, ref_model=None, dpo_loss=None):
    device = int(os.environ.get('LOCAL_RANK', 0))

    dtype = info_dict["dtype"]
    if dtype == "fp16":
        dtype = torch.float16
    elif dtype == "bf16":
        dtype = torch.bfloat16
    else:  # fp32
        dtype = torch.float32

    if hasattr(torch, 'amp') and hasattr(torch.amp, 'autocast'):
        if info_dict['train_engine'] == 'torch_ddp':
            autocast = torch.amp.autocast('cuda', enabled=scaler is not None, dtype=dtype)
        else:
            autocast = torch.amp.autocast('cuda', enabled=True, dtype=dtype, cache_enabled=False)
    else:
        if info_dict['train_engine'] == 'torch_ddp':
            autocast = torch.cuda.amp.autocast(enabled=scaler is not None, dtype=dtype)
        else:
            autocast = torch.cuda.amp.autocast(enabled=True, dtype=dtype, cache_enabled=False)

    with autocast:
        info_dict['loss_dict'] = model(batch, device)
        if ref_model is not None and dpo_loss is not None:
            chosen_logps = info_dict['loss_dict']["chosen_logps"]
            rejected_logps = info_dict['loss_dict']["rejected_logps"]
            sft_loss = info_dict['loss_dict']['loss']
            with torch.no_grad():
                ref_loss_dict = ref_model(batch, device)
            reference_chosen_logps = ref_loss_dict["chosen_logps"]
            reference_rejected_logps = ref_loss_dict["rejected_logps"]
            preference_loss, chosen_reward, reject_reward = dpo_loss(
                chosen_logps, rejected_logps, reference_chosen_logps, reference_rejected_logps
            )
            dpo_acc = (chosen_reward > reject_reward).float().mean()
            info_dict['loss_dict']["loss"] = preference_loss + sft_loss
            info_dict['loss_dict']["sft_loss"] = sft_loss
            info_dict['loss_dict']["dpo_loss"] = preference_loss
            info_dict['loss_dict']["dpo_acc"] = dpo_acc
            info_dict['loss_dict']["chosen_reward"] = chosen_reward.mean()
            info_dict['loss_dict']["reject_reward"] = reject_reward.mean()
    return info_dict


def batch_backward(model, scaler, info_dict):
    if info_dict["train_engine"] == "deepspeed":
        scaled_loss = model.backward(info_dict['loss_dict']['loss'])
    else:
        scaled_loss = info_dict['loss_dict']['loss'] / info_dict['accum_grad']
        if scaler is not None:
            scaler.scale(scaled_loss).backward()
        else:
            scaled_loss.backward()

    info_dict['loss_dict']['loss'] = scaled_loss
    return info_dict


def update_parameter_and_lr(model, optimizer, scheduler, scaler, info_dict):
    grad_norm = 0.0
    if info_dict['train_engine'] == "deepspeed":
        info_dict["is_gradient_accumulation_boundary"] = model.is_gradient_accumulation_boundary()
        model.step()
        grad_norm = model.get_global_grad_norm()
    elif (info_dict['batch_idx'] + 1) % info_dict["accum_grad"] == 0:
        # Use mixed precision training
        if scaler is not None:
            scaler.unscale_(optimizer)
            grad_norm = clip_grad_norm_(model.parameters(), info_dict['grad_clip'])
            local_ok = _grad_norm_is_finite(grad_norm)
            step_ok = _ddp_all_ranks_ok_to_optimizer_step(
                local_ok, model, info_dict['train_engine'])
            if step_ok:
                scaler.step(optimizer)
            else:
                if local_ok:
                    logging.warning(
                        'Skipping optimizer step on this rank because another rank had '
                        'non-finite grad_norm (DDP sync).'
                    )
                else:
                    _warn_infinite_grad_norm()
            scaler.update()
        else:
            grad_norm = clip_grad_norm_(model.parameters(), info_dict['grad_clip'])
            local_ok = _grad_norm_is_finite(grad_norm)
            step_ok = _ddp_all_ranks_ok_to_optimizer_step(
                local_ok, model, info_dict['train_engine'])
            if step_ok:
                optimizer.step()
            else:
                if local_ok:
                    logging.warning(
                        'Skipping optimizer step on this rank because another rank had '
                        'non-finite grad_norm (DDP sync).'
                    )
                else:
                    _warn_infinite_grad_norm()
        optimizer.zero_grad()
        scheduler.step()
    info_dict["lr"] = optimizer.param_groups[0]['lr']
    info_dict["grad_norm"] = grad_norm
    return info_dict


def log_per_step(writer, info_dict):
    tag = info_dict["tag"]
    epoch = info_dict.get('epoch', 0)
    step = info_dict["step"]
    batch_idx = info_dict["batch_idx"]
    loss_dict = info_dict['loss_dict']
    rank = int(os.environ.get('RANK', 0))

    # only rank 0 write to tensorboard to avoid multi-process write
    if writer is not None:
        if (info_dict['train_engine'] == 'deepspeed' and info_dict['is_gradient_accumulation_boundary'] is True) or \
           (info_dict['train_engine'] == 'torch_ddp' and (info_dict['batch_idx'] + 1) % info_dict['accum_grad'] == 0):
            for k in ['epoch', 'lr', 'grad_norm']:
                writer.add_scalar('{}/{}'.format(tag, k), info_dict[k], step + 1)
            for k, v in loss_dict.items():
                writer.add_scalar('{}/{}'.format(tag, k), v, step + 1)

    # TRAIN & CV, Shell log (stdout)
    if (info_dict['batch_idx'] + 1) % info_dict['log_interval'] == 0 and _should_log_current_rank(info_dict):
        log_str = '{} Batch {}/{} '.format(tag, epoch, batch_idx + 1)
        for name, value in loss_dict.items():
            log_str += '{} {:.6f} '.format(name, value)
        if tag == "TRAIN":
            log_str += 'lr {:.8f} grad_norm {:.6f}'.format(
                info_dict["lr"], info_dict['grad_norm'])
        log_str += ' rank {}'.format(rank)
        logging.debug(log_str)


def log_per_save(writer, info_dict):
    tag = info_dict["tag"]
    epoch = info_dict["epoch"]
    step = info_dict["step"]
    loss_dict = info_dict["loss_dict"]
    lr = info_dict['lr']
    rank = int(os.environ.get('RANK', 0))
    if _should_log_current_rank(info_dict):
        logging.info(
            'Epoch {} Step {} CV info lr {} {} rank {}'.format(
                epoch, step + 1, lr, rank, ' '.join(['{} {}'.format(k, v) for k, v in loss_dict.items()])))

    if writer is not None:
        for k in ['epoch', 'lr']:
            writer.add_scalar('{}/{}'.format(tag, k), info_dict[k], step + 1)
        for k, v in loss_dict.items():
            writer.add_scalar('{}/{}'.format(tag, k), v, step + 1)
