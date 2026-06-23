# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
#
# Modified by Felix Benjamin Mueller, 2025

import time
import torch
from privi.jepa.src.utils.tensors import repeat_interleave_batch
from privi.jepa.src.masks.utils import apply_masks
from torch.nn import functional as F
from privi.jepa.src.utils.logging import (
    gpu_timer,
    get_logger,
    grad_logger,
    adamw_logger,
    AverageMeter)
from privi.jepa.src.utils.distributed import AllReduce
import numpy as np

logger = get_logger(__name__)

def train_epoch(
        loader,
        unsupervised_loader, 
        unsupervised_sampler, 
        ipe, 
        cfgs_mask, 
        device, 
        batch_size, 
        num_clips, 
        scheduler, 
        wd_scheduler, 
        target_encoder, 
        encoder, 
        predictor, 
        loss_exp, 
        dtype, 
        mixed_precision, 
        scaler, 
        optimizer, 
        reg_coeff, 
        start_epoch,
        num_epochs, 
        warmup, 
        clip_grad, 
        momentum_scheduler, 
        world_size, 
        csv_logger, 
        rank, 
        run, 
        log_freq,
        save_fn):
    
    samples_seen = 0
    
    # -- TRAINING LOOP
    for epoch in range(start_epoch, num_epochs):
        logger.info('Epoch %d' % (epoch + 1))

        # -- update distributed-data-loader epoch
        unsupervised_sampler.set_epoch(epoch)

        loss_meter = AverageMeter()
        input_var_meter = AverageMeter()
        input_var_min_meter = AverageMeter()
        jepa_loss_meter = AverageMeter()
        reg_loss_meter = AverageMeter()
        mask_meters = [AverageMeter() for _ in range(len(cfgs_mask))]
        gpu_time_meter = AverageMeter()
        wall_time_meter = AverageMeter()

        for itr in range(ipe):
            itr_start_time = time.time()

            try:
                udata, masks_enc, masks_pred = next(loader)
            except Exception as e:
                if not isinstance(e, StopIteration):
                    logger.warning(f'Exception refreshing data loader: {e.__class__.__name__}: {e}')
                logger.info('Exhausted data loaders. Refreshing...')
                loader = iter(unsupervised_loader)
                udata, masks_enc, masks_pred = next(loader)
            assert len(masks_enc) == len(masks_pred), \
                'Currently require num encoder masks = num predictor masks'
        
            def load_clips():
                # -- unsupervised video clips
                # Put each clip on the GPU and concatenate along batch
                # dimension
                clips = torch.cat([u.to(device, non_blocking=True) for u in udata[0]], dim=0)

                # Put each mask-enc/mask-pred pair on the GPU and reuse the
                # same mask pair for each clip
                _masks_enc, _masks_pred = [], []
                for _me, _mp in zip(masks_enc, masks_pred):
                    _me = _me.to(device, non_blocking=True)
                    _mp = _mp.to(device, non_blocking=True)
                    _me = repeat_interleave_batch(_me, batch_size, repeat=num_clips)
                    _mp = repeat_interleave_batch(_mp, batch_size, repeat=num_clips)
                    _masks_enc.append(_me)
                    _masks_pred.append(_mp)

                return (clips, _masks_enc, _masks_pred)
            clips, masks_enc, masks_pred = load_clips()

            for _i, m in enumerate(mask_meters):
                m.update(masks_enc[_i][0].size(-1))

            def train_step():
                _new_lr = scheduler.step()
                _new_wd = wd_scheduler.step()
                # --

                #print(f"masks: {[(m.shape, m.float().mean()) for m in masks_enc]}, {[(m.shape, m.float().mean()) for m in masks_pred]}")

                def forward_target(c):
                    """
                    Returns list of tensors of shape [B, N, D], one for each
                    mask-pred.
                    """
                    with torch.no_grad():
                        h = target_encoder(c)
                        h = F.layer_norm(h, (h.size(-1),))  # normalize over feature-dim  [B, N, D]
                        # -- create targets (masked regions of h)
                        h = apply_masks(h, masks_pred, concat=False)
                        #print("h0 shape", h[0].shape)
                        return h

                def forward_context(c, h):
                    """
                    Returns list of tensors of shape [B, N, D], one for each
                    mask-pred.
                    """
                    z = encoder(c, masks_enc)
                    #print("s0 shape", z[0].shape)
                    z = predictor(z, h, masks_enc, masks_pred)
                    return z

                def loss_fn(z, h):
                    loss = 0.
                    # Compute loss and accumulate for each mask-enc/mask-pred pair
                    for zi, hi in zip(z, h):
                        loss += torch.mean(torch.abs(zi - hi)**loss_exp) / loss_exp
                    loss /= len(masks_pred)
                    return loss

                def reg_fn(z):
                    return sum([torch.sqrt(zi.var(dim=1) + 0.0001) for zi in z]) / len(z)

                # Step 1. Forward
                loss_jepa, loss_reg = 0., 0.
                with torch.cuda.amp.autocast(dtype=dtype, enabled=mixed_precision):
                    h = forward_target(clips)
                    z = forward_context(clips, h)
                    loss_jepa = loss_fn(z, h)  # jepa prediction loss
                    pstd_z = reg_fn(z)  # predictor variance across patches
                    loss_reg += torch.mean(F.relu(1.-pstd_z))
                loss = loss_jepa + reg_coeff * loss_reg

                # Step 2. Backward & step
                _enc_norm, _pred_norm = 0., 0.
                if mixed_precision:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                else:
                    loss.backward()
                if (epoch > warmup) and (clip_grad is not None):
                    _enc_norm = torch.nn.utils.clip_grad_norm_(encoder.parameters(), clip_grad)
                    _pred_norm = torch.nn.utils.clip_grad_norm_(predictor.parameters(), clip_grad)
                if mixed_precision:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                grad_stats = grad_logger(encoder.named_parameters())
                grad_stats.global_norm = float(_enc_norm)
                grad_stats_pred = grad_logger(predictor.named_parameters())
                grad_stats_pred.global_norm = float(_pred_norm)
                optimizer.zero_grad()
                optim_stats = adamw_logger(optimizer)

                # Step 3. momentum update of target encoder
                m = next(momentum_scheduler)
                with torch.no_grad():
                    for param_q, param_k in zip(encoder.parameters(), target_encoder.parameters()):
                        param_k.data.mul_(m).add_((1.-m) * param_q.detach().data)

                return (
                    float(loss),
                    float(loss_jepa),
                    float(loss_reg),
                    _new_lr,
                    _new_wd,
                    grad_stats,
                    grad_stats_pred,
                    optim_stats,
                )
            (loss, loss_jepa, loss_reg, _new_lr, _new_wd, grad_stats, grad_stats_pred, optim_stats,), gpu_etime_ms = gpu_timer(train_step)
            iter_elapsed_time_ms = (time.time() - itr_start_time) * 1000.
            loss_meter.update(loss)
            input_var = float(AllReduce.apply(clips.view(clips.shape[0], -1).var(dim=1).mean(dim=0)))
            input_var_min = float(AllReduce.apply(torch.min(clips.view(clips.shape[0], -1).var(dim=1))))
            input_var_meter.update(input_var)
            input_var_min_meter.update(input_var_min)
            jepa_loss_meter.update(loss_jepa)
            reg_loss_meter.update(loss_reg)
            gpu_time_meter.update(gpu_etime_ms)
            wall_time_meter.update(iter_elapsed_time_ms)

            samples_seen += len(clips) * world_size

            # -- Logging
            def log_stats():
                csv_logger.log(
                    epoch + 1,
                    itr,
                    loss,
                    loss_jepa,
                    loss_reg,
                    grad_stats.global_norm,
                    grad_stats_pred.global_norm,
                    gpu_etime_ms,
                    iter_elapsed_time_ms)
                
                if rank == 0:
                    log_dict ={
                        "epoch": epoch + 1,
                        #"itr": itr,
                        "loss": loss,
                        #"loss-jepa": loss_jepa,
                        #"reg-loss": loss_reg,
                        #"enc-grad-norm": grad_stats.global_norm,
                        #"pred-grad-norm": grad_stats_pred.global_norm,
                        "gpu-time(ms)": gpu_etime_ms,
                        "wall-time(ms)": iter_elapsed_time_ms,
                        "samples-seen": samples_seen,
                        "learning-rate": _new_lr,
                        "weight-decay": _new_wd,
                    }

                    if reg_coeff > 0:
                        # if reg_coeff is 0, then the loss only consists of the jepa loss
                        log_dict["reg-loss"] = loss_reg
                        log_dict["loss-jepa"] = loss_jepa

                    log_dict.update({f'avg/mask-{i}': m.avg for i, m in enumerate(mask_meters)})
                    log_dict.update({
                        "avg/loss": loss_meter.avg,
                        "avg/input-var": input_var_meter.avg,
                        "avg/input-var-min": input_var_min_meter.avg,
                        "avg/gpu-time": gpu_time_meter.avg,
                        "avg/wall-time": wall_time_meter.avg,
                    })
                    log_dict["cuda-memory"] = torch.cuda.max_memory_allocated() / 1024.0**2

                    log_dict.update({
                        "optim-stats/first_moment": optim_stats.get('exp_avg').avg,
                        "optim-stats/first_moment_min": optim_stats.get('exp_avg').min,
                        "optim-stats/first_moment_max": optim_stats.get('exp_avg').max,
                        "optim-stats/second_moment": optim_stats.get('exp_avg_sq').avg,
                        "optim-stats/second_moment_min": optim_stats.get('exp_avg_sq').min,
                        "optim-stats/second_moment_max": optim_stats.get('exp_avg_sq').max,
                    })

                    log_dict.update({
                        "grad-stats/enc-first-layer": grad_stats.first_layer,
                        "grad-stats/enc-last-layer": grad_stats.last_layer,
                        "grad-stats/enc-min": grad_stats.min,
                        "grad-stats/enc-max": grad_stats.max,
                        "grad-stats/enc-global-norm": grad_stats.global_norm,
                    })

                    log_dict.update({
                        "grad-stats/pred-first-layer": grad_stats_pred.first_layer,
                        "grad-stats/pred-last-layer": grad_stats_pred.last_layer,
                        "grad-stats/pred-min": grad_stats_pred.min,
                        "grad-stats/pred-max": grad_stats_pred.max,
                        "grad-stats/pred-global-norm": grad_stats_pred.global_norm,
                    })

                    run.log(log_dict)
                if (itr % log_freq == 0) or np.isnan(loss) or np.isinf(loss):
                    logger.info(
                        '[%d, %5d] loss: %.3f | p%.3f r%.3f | '
                        'input_var: %.3f %.3f | '
                        'masks: %s '
                        '[wd: %.2e] [lr: %.2e] '
                        '[mem: %.2e] '
                        '[gpu: %.1f ms]'
                        '[wall: %.1f ms]'
                        % (epoch + 1, itr,
                            loss_meter.avg,
                            jepa_loss_meter.avg,
                            reg_loss_meter.avg,
                            input_var_meter.avg,
                            input_var_min_meter.avg,
                            '[' + ', '.join(['%.1f' % m.avg for m in mask_meters]) + ']',
                            _new_wd,
                            _new_lr,
                            torch.cuda.max_memory_allocated() / 1024.0**2,
                            gpu_time_meter.avg,
                            wall_time_meter.avg))

                    if optim_stats is not None:
                        logger.info(
                            '[%d, %5d] first moment: %.2e [%.2e %.2e] second moment: %.2e [%.2e %.2e]'
                            % (epoch + 1, itr,
                                optim_stats.get('exp_avg').avg,
                                optim_stats.get('exp_avg').min,
                                optim_stats.get('exp_avg').max,
                                optim_stats.get('exp_avg_sq').avg,
                                optim_stats.get('exp_avg_sq').min,
                                optim_stats.get('exp_avg_sq').max))

                    if grad_stats is not None:
                        logger.info(
                            '[%d, %5d] enc_grad_stats: f/l[%.2e %.2e] mn/mx(%.2e, %.2e) %.2e'
                            % (epoch + 1, itr,
                                grad_stats.first_layer,
                                grad_stats.last_layer,
                                grad_stats.min,
                                grad_stats.max,
                                grad_stats.global_norm))

                    if grad_stats_pred is not None:
                        logger.info(
                            '[%d, %5d] pred_grad_stats: f/l[%.2e %.2e] mn/mx(%.2e, %.2e) %.2e'
                            % (epoch + 1, itr,
                                grad_stats_pred.first_layer,
                                grad_stats_pred.last_layer,
                                grad_stats_pred.min,
                                grad_stats_pred.max,
                                grad_stats_pred.global_norm))
            log_stats()
            assert not np.isnan(loss), 'loss is nan'

        # -- Save Checkpoint
        logger.info('avg. loss %.3f' % loss_meter.avg)
        save_fn(epoch, loss_meter)