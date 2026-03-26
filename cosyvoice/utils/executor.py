# Copyright (c) 2020 Mobvoi Inc (Binbin Zhang)
#               2024 Alibaba Inc (authors: Xiang Lyu)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
from contextlib import nullcontext
import os
import time
import datetime

import torch
import torch.distributed as dist

from cosyvoice.utils.train_utils import update_parameter_and_lr, log_per_step, log_per_save, batch_forward, batch_backward, save_model, cosyvoice_join


class Executor:

    def __init__(self, gan: bool = False, ref_model: torch.nn.Module = None, dpo_loss: torch.nn.Module = None):
        self.gan = gan
        self.ref_model = ref_model
        self.dpo_loss = dpo_loss
        self.step = 0
        self.epoch = 0
        self.rank = int(os.environ.get('RANK', 0))
        self.local_rank = int(os.environ.get('LOCAL_RANK', 0))
        self.device = torch.device('cuda:{}'.format(self.local_rank))

    @staticmethod
    def _batch_max_feat_len(batch_dict):
        feat_len = batch_dict.get('speech_feat_len', None)
        if feat_len is None:
            return None
        if torch.is_tensor(feat_len):
            if feat_len.numel() == 0:
                return None
            return int(feat_len.max().item())
        if isinstance(feat_len, (list, tuple)) and len(feat_len) > 0:
            return int(max(feat_len))
        return None

    def _should_skip_overlong_batch(self, batch_dict, info_dict, tag):
        max_feat_len = int(info_dict.get('max_feat_len', 12000))
        observed = self._batch_max_feat_len(batch_dict)
        if observed is None:
            return False
        if observed > max_feat_len:
            logging.warning(
                'Skip %s batch: observed speech_feat_len=%s > max_feat_len=%s (epoch=%s step=%s rank=%s)',
                tag, observed, max_feat_len, self.epoch, self.step + 1, self.rank
            )
            return True
        return False

    @staticmethod
    def _sanitize_batch_lengths(batch_dict):
        """Clamp length metadata to actual tensor widths for safety."""
        feat = batch_dict.get('speech_feat', None)
        feat_len = batch_dict.get('speech_feat_len', None)
        if torch.is_tensor(feat) and feat.ndim >= 2 and torch.is_tensor(feat_len):
            max_feat = int(feat.shape[1])
            batch_dict['speech_feat_len'] = torch.clamp(feat_len.to(torch.long), min=0, max=max_feat)

        token = batch_dict.get('speech_token', None)
        token_len = batch_dict.get('speech_token_len', None)
        if torch.is_tensor(token) and token.ndim >= 2 and torch.is_tensor(token_len):
            max_tok = int(token.shape[1])
            batch_dict['speech_token_len'] = torch.clamp(token_len.to(torch.long), min=0, max=max_tok)

    @staticmethod
    def _is_recoverable_batch_runtime_error(exc: RuntimeError) -> bool:
        msg = str(exc)
        recoverable_patterns = (
            'must match the size of tensor b',
            'must match the size of tensor',
            'The size of tensor a',
            'The expanded size of the tensor',
        )
        return any(p in msg for p in recoverable_patterns)

    def _safe_epoch_end_sync(self, group_join, info_dict):
        """Synchronize ranks before CV with a bounded timeout."""
        timeout_sec = int(info_dict.get('timeout', 60))
        try:
            logging.info(
                'Epoch %s post-train monitored_barrier() enter rank=%s timeout=%ss',
                self.epoch, self.rank, timeout_sec,
            )
            dist.monitored_barrier(
                group=group_join,
                timeout=datetime.timedelta(seconds=timeout_sec),
            )
            logging.info('Epoch %s post-train monitored_barrier() leave rank=%s', self.epoch, self.rank)
            return True
        except RuntimeError as e:
            logging.info(
                "Epoch %s post-train sync failed on rank=%s local_rank=%s: %s\n"
                "Skip CV/save for this epoch to avoid long hang.",
                self.epoch, self.rank, self.local_rank, e,
            )
            return False

    def train_one_epoc(self, model, optimizer, scheduler, train_data_loader, cv_data_loader, writer, info_dict, scaler, group_join, ref_model=None):
        ''' Train one epoch
        '''

        lr = optimizer.param_groups[0]['lr']
        if info_dict.get('log_all_ranks', False) or self.rank == 0:
            logging.info('Epoch {} TRAIN info lr {} rank {}'.format(self.epoch, lr, self.rank))
            logging.info('using accumulate grad, new batch size is {} times'
                         ' larger than before'.format(info_dict['accum_grad']))
        # A context manager to be used in conjunction with an instance of
        # torch.nn.parallel.DistributedDataParallel to be able to train
        # with uneven inputs across participating processes.
        model.train()
        if self.ref_model is not None:
            self.ref_model.eval()
        model_context = model.join if info_dict['train_engine'] == 'torch_ddp' else nullcontext
        with model_context():
            for batch_idx, batch_dict in enumerate(train_data_loader):
                max_batches_per_epoch = int(info_dict.get('max_batches_per_epoch', -1))
                if max_batches_per_epoch > 0 and batch_idx >= max_batches_per_epoch:
                    logging.info(
                        'Hit max_batches_per_epoch=%s at batch_idx=%s, stop epoch early (rank=%s)',
                        max_batches_per_epoch, batch_idx, self.rank
                    )
                    break
                batch_start = time.monotonic()
                info_dict["tag"] = "TRAIN"
                info_dict["step"] = self.step
                info_dict["epoch"] = self.epoch
                info_dict["batch_idx"] = batch_idx
                self._sanitize_batch_lengths(batch_dict)
                if self._should_skip_overlong_batch(batch_dict, info_dict, 'TRAIN'):
                    continue
                utts = batch_dict.get('utts', [])
                info_dict['last_batch_num_utts'] = len(utts)
                info_dict['last_batch_sample_utts'] = list(utts[:3]) if isinstance(utts, list) else 'n/a'
                if cosyvoice_join(group_join, info_dict):
                    break

                # Disable gradient synchronizations across DDP processes.
                # Within this context, gradients will be accumulated on module
                # variables, which will later be synchronized.
                if info_dict['train_engine'] == 'torch_ddp' and (batch_idx + 1) % info_dict["accum_grad"] != 0:
                    context = model.no_sync
                # Used for single gpu training and DDP gradient synchronization
                # processes.
                else:
                    context = nullcontext

                try:
                    with context():
                        info_dict = batch_forward(model, batch_dict, scaler, info_dict, ref_model=self.ref_model, dpo_loss=self.dpo_loss)
                        info_dict = batch_backward(model, scaler, info_dict)
                except torch.OutOfMemoryError:
                    logging.exception(
                        'OOM in TRAIN batch, skip and continue: epoch=%s step=%s batch_idx=%s rank=%s',
                        self.epoch, self.step + 1, batch_idx, self.rank
                    )
                    optimizer.zero_grad(set_to_none=True)
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    continue
                except RuntimeError as e:
                    if self._is_recoverable_batch_runtime_error(e):
                        logging.exception(
                            'Recoverable TRAIN runtime error, skip batch: epoch=%s step=%s batch_idx=%s rank=%s sample_utts=%s',
                            self.epoch, self.step + 1, batch_idx, self.rank,
                            info_dict.get('last_batch_sample_utts', 'n/a'),
                        )
                        optimizer.zero_grad(set_to_none=True)
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        continue
                    raise

                info_dict = update_parameter_and_lr(model, optimizer, scheduler, scaler, info_dict)
                info_dict['last_batch_cost_sec'] = round(time.monotonic() - batch_start, 3)
                slow_batch_sec = float(info_dict.get('slow_batch_sec', 120.0))
                if info_dict['last_batch_cost_sec'] > slow_batch_sec:
                    logging.warning(
                        'Slow TRAIN batch detected: epoch=%s step=%s batch_idx=%s cost_sec=%.3f rank=%s sample_utts=%s',
                        self.epoch, self.step + 1, batch_idx, info_dict['last_batch_cost_sec'], self.rank,
                        info_dict.get('last_batch_sample_utts', 'n/a'),
                    )
                log_per_step(writer, info_dict)
                # NOTE specify save_per_step in cosyvoice.yaml if you want to enable step save
                if info_dict['save_per_step'] > 0 and (self.step + 1) % info_dict['save_per_step'] == 0 and \
                   (batch_idx + 1) % info_dict["accum_grad"] == 0:
                    dist.barrier()
                    self.cv(model, cv_data_loader, writer, info_dict, on_batch_end=False)
                    model.train()
                if (batch_idx + 1) % info_dict["accum_grad"] == 0:
                    self.step += 1
        # Help debug apparent "hangs": TRAIN DEBUG stops here until barrier+CV finish.
        logging.info(
            'Epoch %s TRAIN loop done rank=%s local_rank=%s step=%s',
            self.epoch, self.rank, self.local_rank, self.step + 1,
        )
        if not self._safe_epoch_end_sync(group_join, info_dict):
            return
        self.cv(model, cv_data_loader, writer, info_dict, on_batch_end=True)

    def train_one_epoc_gan(self, model, optimizer, scheduler, optimizer_d, scheduler_d, train_data_loader, cv_data_loader,
                           writer, info_dict, scaler, group_join):
        ''' Train one epoch
        '''

        lr = optimizer.param_groups[0]['lr']
        if info_dict.get('log_all_ranks', False) or self.rank == 0:
            logging.info('Epoch {} TRAIN info lr {} rank {}'.format(self.epoch, lr, self.rank))
            logging.info('using accumulate grad, new batch size is {} times'
                         ' larger than before'.format(info_dict['accum_grad']))
        # A context manager to be used in conjunction with an instance of
        # torch.nn.parallel.DistributedDataParallel to be able to train
        # with uneven inputs across participating processes.
        model.train()
        model_context = model.join if info_dict['train_engine'] == 'torch_ddp' else nullcontext
        with model_context():
            for batch_idx, batch_dict in enumerate(train_data_loader):
                max_batches_per_epoch = int(info_dict.get('max_batches_per_epoch', -1))
                if max_batches_per_epoch > 0 and batch_idx >= max_batches_per_epoch:
                    logging.info(
                        'Hit max_batches_per_epoch=%s at batch_idx=%s, stop epoch early (GAN, rank=%s)',
                        max_batches_per_epoch, batch_idx, self.rank
                    )
                    break
                batch_start = time.monotonic()
                info_dict["tag"] = "TRAIN"
                info_dict["step"] = self.step
                info_dict["epoch"] = self.epoch
                info_dict["batch_idx"] = batch_idx
                self._sanitize_batch_lengths(batch_dict)
                if self._should_skip_overlong_batch(batch_dict, info_dict, 'TRAIN'):
                    continue
                utts = batch_dict.get('utts', [])
                info_dict['last_batch_num_utts'] = len(utts)
                info_dict['last_batch_sample_utts'] = list(utts[:3]) if isinstance(utts, list) else 'n/a'
                if cosyvoice_join(group_join, info_dict):
                    break

                # Disable gradient synchronizations across DDP processes.
                # Within this context, gradients will be accumulated on module
                # variables, which will later be synchronized.
                if info_dict['train_engine'] == 'torch_ddp' and (batch_idx + 1) % info_dict["accum_grad"] != 0:
                    context = model.no_sync
                # Used for single gpu training and DDP gradient synchronization
                # processes.
                else:
                    context = nullcontext

                try:
                    with context():
                        batch_dict['turn'] = 'discriminator'
                        info_dict = batch_forward(model, batch_dict, scaler, info_dict)
                        info_dict = batch_backward(model, scaler, info_dict)
                except torch.OutOfMemoryError:
                    logging.exception(
                        'OOM in TRAIN batch(discriminator), skip and continue: epoch=%s step=%s batch_idx=%s rank=%s',
                        self.epoch, self.step + 1, batch_idx, self.rank
                    )
                    optimizer.zero_grad(set_to_none=True)
                    optimizer_d.zero_grad(set_to_none=True)
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    continue
                except RuntimeError as e:
                    if self._is_recoverable_batch_runtime_error(e):
                        logging.exception(
                            'Recoverable TRAIN runtime error(discriminator), skip batch: epoch=%s step=%s batch_idx=%s rank=%s sample_utts=%s',
                            self.epoch, self.step + 1, batch_idx, self.rank,
                            info_dict.get('last_batch_sample_utts', 'n/a'),
                        )
                        optimizer.zero_grad(set_to_none=True)
                        optimizer_d.zero_grad(set_to_none=True)
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        continue
                    raise
                info_dict = update_parameter_and_lr(model, optimizer_d, scheduler_d, scaler, info_dict)
                optimizer.zero_grad()
                log_per_step(writer, info_dict)
                try:
                    with context():
                        batch_dict['turn'] = 'generator'
                        info_dict = batch_forward(model, batch_dict, scaler, info_dict)
                        info_dict = batch_backward(model, scaler, info_dict)
                except torch.OutOfMemoryError:
                    logging.exception(
                        'OOM in TRAIN batch(generator), skip and continue: epoch=%s step=%s batch_idx=%s rank=%s',
                        self.epoch, self.step + 1, batch_idx, self.rank
                    )
                    optimizer.zero_grad(set_to_none=True)
                    optimizer_d.zero_grad(set_to_none=True)
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    continue
                except RuntimeError as e:
                    if self._is_recoverable_batch_runtime_error(e):
                        logging.exception(
                            'Recoverable TRAIN runtime error(generator), skip batch: epoch=%s step=%s batch_idx=%s rank=%s sample_utts=%s',
                            self.epoch, self.step + 1, batch_idx, self.rank,
                            info_dict.get('last_batch_sample_utts', 'n/a'),
                        )
                        optimizer.zero_grad(set_to_none=True)
                        optimizer_d.zero_grad(set_to_none=True)
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        continue
                    raise
                info_dict = update_parameter_and_lr(model, optimizer, scheduler, scaler, info_dict)
                info_dict['last_batch_cost_sec'] = round(time.monotonic() - batch_start, 3)
                optimizer_d.zero_grad()
                slow_batch_sec = float(info_dict.get('slow_batch_sec', 120.0))
                if info_dict['last_batch_cost_sec'] > slow_batch_sec:
                    logging.warning(
                        'Slow TRAIN batch detected: epoch=%s step=%s batch_idx=%s cost_sec=%.3f rank=%s sample_utts=%s',
                        self.epoch, self.step + 1, batch_idx, info_dict['last_batch_cost_sec'], self.rank,
                        info_dict.get('last_batch_sample_utts', 'n/a'),
                    )
                log_per_step(writer, info_dict)
                # NOTE specify save_per_step in cosyvoice.yaml if you want to enable step save
                if info_dict['save_per_step'] > 0 and (self.step + 1) % info_dict['save_per_step'] == 0 and \
                   (batch_idx + 1) % info_dict["accum_grad"] == 0:
                    dist.barrier()
                    self.cv(model, cv_data_loader, writer, info_dict, on_batch_end=False)
                    model.train()
                if (batch_idx + 1) % info_dict["accum_grad"] == 0:
                    self.step += 1
        logging.info(
            'Epoch %s TRAIN loop done (GAN) rank=%s local_rank=%s step=%s',
            self.epoch, self.rank, self.local_rank, self.step + 1,
        )
        if not self._safe_epoch_end_sync(group_join, info_dict):
            return
        self.cv(model, cv_data_loader, writer, info_dict, on_batch_end=True)

    @torch.inference_mode()
    def cv(self, model, cv_data_loader, writer, info_dict, on_batch_end=True):
        ''' Cross validation on
        '''
        if info_dict.get('log_all_ranks', False) or self.rank == 0:
            logging.info('Epoch {} Step {} on_batch_end {} CV rank {}'.format(self.epoch, self.step + 1, on_batch_end, self.rank))
        model.eval()
        total_num_utts, total_loss_dict = 0, {}  # avoid division by 0
        for batch_idx, batch_dict in enumerate(cv_data_loader):
            cv_batch_start = time.monotonic()
            info_dict["tag"] = "CV"
            info_dict["step"] = self.step
            info_dict["epoch"] = self.epoch
            info_dict["batch_idx"] = batch_idx
            self._sanitize_batch_lengths(batch_dict)
            if self._should_skip_overlong_batch(batch_dict, info_dict, 'CV'):
                continue

            num_utts = len(batch_dict["utts"])
            total_num_utts += num_utts

            if self.gan is True:
                batch_dict['turn'] = 'generator'
            try:
                info_dict = batch_forward(model, batch_dict, None, info_dict)
            except torch.OutOfMemoryError:
                logging.exception(
                    'OOM in CV batch, skip and continue: epoch=%s step=%s batch_idx=%s rank=%s',
                    self.epoch, self.step + 1, batch_idx, self.rank
                )
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                continue

            for k, v in info_dict['loss_dict'].items():
                if k not in total_loss_dict:
                    total_loss_dict[k] = []
                total_loss_dict[k].append(v.mean().item() * num_utts)
            log_per_step(None, info_dict)
            batch_cost = time.monotonic() - cv_batch_start
            slow_batch_sec = float(info_dict.get('slow_batch_sec', 120.0))
            if batch_cost > slow_batch_sec and (info_dict.get('log_all_ranks', False) or self.rank == 0):
                logging.warning(
                    'Slow CV batch detected: epoch=%s step=%s batch_idx=%s cost_sec=%.3f rank=%s',
                    self.epoch, self.step + 1, batch_idx, batch_cost, self.rank,
                )
        for k, v in total_loss_dict.items():
            total_loss_dict[k] = sum(v) / total_num_utts
        info_dict['loss_dict'] = total_loss_dict
        log_per_save(writer, info_dict)
        model_name = 'epoch_{}_whole'.format(self.epoch) if on_batch_end else 'epoch_{}_step_{}'.format(self.epoch, self.step + 1)
        save_model(model, model_name, info_dict)
