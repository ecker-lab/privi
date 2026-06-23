# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
#
# Modified by Felix Benjamin Mueller, 2025

import gc
import time
from privi.vjepa2.src.utils.logging import AverageMeter, get_logger, gpu_timer
import torch
from torch.nn import functional as F
import numpy as np

from privi.vjepa2.src.masks.utils import apply_masks

logger = get_logger(__name__, force=True)

def train_epoch(
        loader,
        unsupervised_sampler, 
        unsupervised_loader, 
        start_epoch,
        num_epochs, 
        batch_size, 
        dataset_fpcs, 
        ipe, 
        device, 
        sync_gc, 
        GARBAGE_COLLECT_ITR_FREQ, 
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
        momentum_scheduler, 
        csv_logger, 
        log_freq, 
        run,
        save_fn):
    
    for epoch in range(start_epoch, num_epochs):
        logger.info('Epoch %d' % (epoch + 1))

        loss_meter = AverageMeter()
        mask_meters = {fpc: AverageMeter() for fpc in dataset_fpcs}
        iter_time_meter = AverageMeter()
        gpu_time_meter = AverageMeter()
        data_elapsed_time_meter = AverageMeter()

        for itr in range(ipe):
            itr_start_time = time.time()

            iter_retries = 0
            iter_successful = False
            while not iter_successful:
                try:
                    sample = next(loader)
                    iter_successful = True
                except StopIteration:
                    logger.info("Exhausted data loaders. Refreshing...")
                    unsupervised_sampler.set_epoch(epoch)
                    loader = iter(unsupervised_loader)
                except Exception as e:
                    NUM_RETRIES = 5
                    if iter_retries < NUM_RETRIES:
                        logger.warning(f"Encountered exception when loading data (num retries {iter_retries}):\n{e}")
                        iter_retries += 1
                        time.sleep(5)
                    else:
                        logger.warning(f"Exceeded max retries ({NUM_RETRIES}) when loading data. Skipping batch.")
                        raise e

            for _fpc_sample in sample:
                bs, fpc = _fpc_sample[0][-1][0].size()
                mask_meters[fpc].update(bs / batch_size)

            def load_clips():
                all_clips, all_masks_enc, all_masks_pred = [], [], []
                for fpc_sample in sample:
                    udata, masks_enc, masks_pred = fpc_sample
                    all_clips += [udata[0][0].to(device, non_blocking=True)]
                    all_masks_enc += [[m.to(device, non_blocking=True) for m in masks_enc]]
                    all_masks_pred += [[m.to(device, non_blocking=True) for m in masks_pred]]
                return all_clips, all_masks_enc, all_masks_pred

            clips, masks_enc, masks_pred = load_clips()
            data_elapsed_time_ms = (time.time() - itr_start_time) * 1000.0

            if sync_gc and (itr + 1) % GARBAGE_COLLECT_ITR_FREQ == 0:
                logger.info("Running garbage collection...")
                gc.collect()

            def train_step():
                _new_lr = scheduler.step()
                _new_wd = wd_scheduler.step()
                # --

                def forward_target(c):
                    with torch.no_grad():
                        h = target_encoder(c)
                        h = [F.layer_norm(hi, (hi.size(-1),)) for hi in h]
                        return h

                def forward_context(c):
                    z = encoder(c, masks_enc)
                    z = predictor(z, masks_enc, masks_pred)
                    return z

                def loss_fn(z, h):
                    # Assumption: predictor will have returned only masked tokens for z
                    h = [apply_masks(hi, mi, concat=False) for hi, mi in zip(h, masks_pred)]

                    loss, n = 0, 0
                    for zi, hi in zip(z, h):
                        for zij, hij in zip(zi, hi):
                            loss += torch.mean(torch.abs(zij - hij) ** loss_exp) / loss_exp
                            n += 1
                    loss /= n
                    return loss

                # Step 1. Forward
                with torch.cuda.amp.autocast(dtype=dtype, enabled=mixed_precision):
                    h = forward_target(clips)
                    z = forward_context(clips)
                    loss = loss_fn(z, h)  # jepa prediction loss

                # Step 2. Backward & step
                if mixed_precision:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                else:
                    loss.backward()
                if mixed_precision:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad()

                # Step 3. momentum update of target encoder
                m = next(momentum_scheduler)
                with torch.no_grad():
                    params_k = []
                    params_q = []
                    for param_q, param_k in zip(encoder.parameters(), target_encoder.parameters()):
                        params_k.append(param_k)
                        params_q.append(param_q)
                    torch._foreach_mul_(params_k, m)
                    torch._foreach_add_(params_k, params_q, alpha=1 - m)

                return (
                    float(loss),
                    _new_lr,
                    _new_wd,
                )

            (
                loss,
                _new_lr,
                _new_wd,
            ), gpu_etime_ms = gpu_timer(train_step)
            iter_elapsed_time_ms = (time.time() - itr_start_time) * 1000.0
            loss_meter.update(loss)
            iter_time_meter.update(iter_elapsed_time_ms)
            gpu_time_meter.update(gpu_etime_ms)
            data_elapsed_time_meter.update(data_elapsed_time_ms)

            # -- Logging
            def log_stats():
                csv_logger.log(epoch + 1, itr, loss, iter_elapsed_time_ms, gpu_etime_ms, data_elapsed_time_ms)
                if (itr % log_freq == 0) or (itr == ipe - 1) or np.isnan(loss) or np.isinf(loss):
                    logger.info(
                        "[%d, %5d] loss: %.3f "
                        "masks: %s "
                        "[wd: %.2e] [lr: %.2e] "
                        "[mem: %.2e] "
                        "[iter: %.1f ms] "
                        "[gpu: %.1f ms] "
                        "[data: %.1f ms]"
                        % (
                            epoch + 1,
                            itr,
                            loss_meter.avg,
                            "[" + ", ".join([f"{k}: " + "%.1f" % mask_meters[k].avg for k in mask_meters]) + "]",
                            _new_wd,
                            _new_lr,
                            torch.cuda.max_memory_allocated() / 1024.0**2,
                            iter_time_meter.avg,
                            gpu_time_meter.avg,
                            data_elapsed_time_meter.avg,
                        )
                    )
                if run is not None:
                    run.log(
                        {
                            "epoch": epoch +1, 
                            "itr": itr,
                            "loss": loss_meter.avg,
                            **{f"mask/{k}" : mask_meters[k].avg for k in mask_meters},
                            "learning-rate": _new_lr,
                            "weigth-decay": _new_wd,
                            "cuda-memory": torch.cuda.max_memory_allocated() / 1024.0**2,
                            "iter-time": iter_time_meter.avg,
                            "gpu-time-meter": gpu_time_meter.avg,
                            "data-elapsed-time": data_elapsed_time_meter.avg,
                        }
                    )

            log_stats()
            assert not np.isnan(loss), "loss is nan"

        save_fn(epoch, loss_meter)