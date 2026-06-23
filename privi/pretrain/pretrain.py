# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
# Modified by Felix Benjamin Mueller, 2025

from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path
import pprint
import shutil
import gc

from tqdm import tqdm

from privi.utils.misc import setup_train

# -- FOR DISTRIBUTED TRAINING ENSURE ONLY 1 DEVICE VISIBLE PER PROCESS
try:
    # -- WARNING: IF DOING DISTRIBUTED TRAINING ON A NON-SLURM CLUSTER, MAKE
    # --          SURE TO UPDATE THIS TO GET LOCAL-RANK ON NODE, OR ENSURE
    # --          THAT YOUR JOBS ARE LAUNCHED WITH ONLY 1 DEVICE VISIBLE
    # --          TO EACH PROCESS
    os.environ['CUDA_VISIBLE_DEVICES'] = os.environ['SLURM_LOCALID']
    # Our slurm setup only exposes one cuda for each task, so nothing to do here
    pass
except Exception:
    pass

import copy
import time
import numpy as np

import torch
import torch.multiprocessing as mp
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel


import wandb

from privi.datasets.generic import make_videodataset


from privi.jepa.src.utils.distributed import init_distributed, broadcast_string
from privi.jepa.src.utils.logging import CSVLogger, get_logger


from privi.jepa.app.vjepa.utils import (
    load_checkpoint,
    init_opt,
)




# --
log_timings = True
log_freq = 10
checkpoint_freq = 1
CHECKPOINT_FREQ = 1
GARBAGE_COLLECT_ITR_FREQ = 50
# --

_GLOBAL_SEED = 0
np.random.seed(_GLOBAL_SEED)
torch.manual_seed(_GLOBAL_SEED)
torch.backends.cudnn.benchmark = True

os.environ["NCCL_DEBUG"] = "INFO"
os.environ["NCCL_TIMEOUT"] = "3600"


logger = get_logger(__name__)




def main(args, resume_preempt=False):
    # ----------------------------------------------------------------------- #
    #  PASSED IN PARAMS FROM CONFIG FILE
    # ----------------------------------------------------------------------- #

    # -- META
    fname = args.get('fname')
    cfgs_meta = args.get('meta')
    name_suffix = cfgs_meta.get("name_suffix", "")
    load_model = cfgs_meta.get('load_checkpoint') or resume_preempt
    r_file = os.path.expandvars(cfgs_meta.get('read_checkpoint', None))
    finetune = cfgs_meta.get('finetune', False)
    seed = cfgs_meta.get('seed', _GLOBAL_SEED)
    save_every_freq = cfgs_meta.get('save_every_freq', -1)
    skip_batches = cfgs_meta.get('skip_batches', -1)
    use_sdpa = cfgs_meta.get('use_sdpa', False)
    which_dtype = cfgs_meta.get('dtype')
    logger.info(f'{which_dtype=}')
    if which_dtype.lower() == 'bfloat16':
        dtype = torch.bfloat16
        mixed_precision = True
    elif which_dtype.lower() == 'float16':
        dtype = torch.float16
        mixed_precision = True
    else:
        dtype = torch.float32
        mixed_precision = False
    eval_epochs = cfgs_meta.get("eval_epochs", [])
    eval_last = cfgs_meta.get("eval_last", False)
    eval_cmds = cfgs_meta.get("eval_cmds", [])
    

    # NEW VJEPA2
    sync_gc = cfgs_meta.get("sync_gc", False)

    # -- MASK
    cfgs_mask = args.get('mask')

    # -- MODEL
    cfgs_model = args.get('model')
    model_architecture = cfgs_model.get("architecture", "vjepa")
    model_name = cfgs_model.get('model_name')
    pred_depth = cfgs_model.get('pred_depth')
    pred_embed_dim = cfgs_model.get('pred_embed_dim')
    uniform_power = cfgs_model.get('uniform_power', True)
    use_mask_tokens = cfgs_model.get('use_mask_tokens', True)
    zero_init_mask_tokens = cfgs_model.get('zero_init_mask_tokens', True)

    # NEW VJEPA2
    pred_num_heads = cfgs_model.get("pred_num_heads", None)
    use_silu = cfgs_model.get("use_silu", False)
    use_pred_silu = cfgs_model.get("use_pred_silu", False)
    wide_silu = cfgs_model.get("wide_silu", True)
    use_rope = cfgs_model.get("use_rope", False)
    use_activation_checkpointing = cfgs_model.get("use_activation_checkpointing", False)
    compile_model = cfgs_model.get("compile_model", False)

    # -- DATA
    cfgs_data = args.get('data')
    mask_type = cfgs_data.get('mask_type', 'multiblock3d')
    dataset_paths = cfgs_data.get('datasets', [])
    datasets_weights = cfgs_data.get('datasets_weights', None)
    if datasets_weights is not None:
        assert len(datasets_weights) == len(dataset_paths), 'Must have one sampling weight specified for each dataset'
    batch_size = cfgs_data.get('batch_size')
    num_clips = cfgs_data.get('num_clips')
    num_frames = cfgs_data.get('num_frames')
    tubelet_size = cfgs_data.get('tubelet_size')
    sampling_rate = cfgs_data.get('sampling_rate')
    duration = cfgs_data.get('clip_duration', None)
    crop_size = cfgs_data.get('crop_size', 224)
    patch_size = cfgs_data.get('patch_size')
    pin_mem = cfgs_data.get('pin_mem', False)
    num_workers = cfgs_data.get('num_workers', 1)
    filter_short_videos = cfgs_data.get('filter_short_videos', False)
    log_resource_util_data = cfgs_data.get('log_resource_utilization', False)
    use_caching = cfgs_data.get('use_caching', False)

    # NEW VJEPA2
    dataset_fpcs = cfgs_data.get("dataset_fpcs", [num_frames])
    max_num_frames = max(dataset_fpcs)

    # -- DATA AUGS
    cfgs_data_aug = args.get('data_aug')
    ar_range = cfgs_data_aug.get('random_resize_aspect_ratio', [3/4, 4/3])
    rr_scale = cfgs_data_aug.get('random_resize_scale', [0.3, 1.0])
    motion_shift = cfgs_data_aug.get('motion_shift', False)
    reprob = cfgs_data_aug.get('reprob', 0.)
    use_aa = cfgs_data_aug.get('auto_augment', False)
    crop_to_bboxes = cfgs_data_aug.get('crop_to_bboxes', True)
    aug_mode = cfgs_data_aug.get("aug_mode", None)
    prob_crop_to_bboxes = cfgs_data_aug.get("prob_crop_to_bboxes", 1.0 if crop_to_bboxes else 0.0)

    # -- LOSS
    cfgs_loss = args.get('loss')
    loss_exp = cfgs_loss.get('loss_exp')
    reg_coeff = cfgs_loss.get('reg_coeff')

    # -- OPTIMIZATION
    cfgs_opt = args.get('optimization')
    ipe = cfgs_opt.get('ipe', None)
    ipe_scale = cfgs_opt.get('ipe_scale', 1.0)
    clip_grad = cfgs_opt.get('clip_grad', None)
    wd = float(cfgs_opt.get('weight_decay'))
    final_wd = float(cfgs_opt.get('final_weight_decay'))
    num_epochs = cfgs_opt.get('epochs')
    warmup = cfgs_opt.get('warmup')
    start_lr = cfgs_opt.get('start_lr')
    lr = cfgs_opt.get('lr')
    final_lr = cfgs_opt.get('final_lr')
    ema = cfgs_opt.get('ema')
    betas = cfgs_opt.get('betas', (0.9, 0.999))
    eps = cfgs_opt.get('eps', 1.e-8)

    # -- LOGGING
    cfgs_logging = args.get('logging')
    folder = os.path.expandvars(cfgs_logging.get('folder'))
    tag = cfgs_logging.get('write_tag')

    # ----------------------------------------------------------------------- #
    # ----------------------------------------------------------------------- #

    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = True
    try:
        mp.set_start_method('spawn')
    except Exception:
        pass

    logger.info(f"Script: {__file__}")
    logger.info(f"CUDA_VISIBLE_DEVICES: {os.environ['CUDA_VISIBLE_DEVICES']}")

    logger.info(f"Cuda devices visible: {torch.cuda.device_count()}")

    # -- init torch distributed backend
    world_size, rank = init_distributed()
    logger.info(f'Initialized (rank/world-size) {rank}/{world_size}')

    # Copy datasets to node-local storage and create csv files for dataloaders
    # Note: This currently only works if all data is used on all nodes

    # ----------------------------------------------------------------------- #

    video_base_dir = os.path.expandvars(dataset_paths["data_path"])
    dataset_paths = [os.path.expandvars(dataset_paths["label_path"])]

    cache_dir = None if not use_caching else os.environ['LOCAL_TMPDIR']

    if isinstance(save_every_freq, int):
        if save_every_freq > 0:
            save_epochs = list(range(0, num_epochs, save_every_freq))
        else:
            save_epochs = []
    elif isinstance(save_every_freq, list):
        save_epochs = save_every_freq

    save_epochs = set(save_epochs).union(eval_epochs)


    # ----------------------------------------------------------------------- #

    if rank == 0:
        run_id, ckpt_dir = setup_train(tag, folder, f"{Path(fname).stem}{name_suffix}", resume_id=None)

        os.makedirs(folder, exist_ok=True)
        os.makedirs(ckpt_dir, exist_ok=True)

        run = wandb.init(
            project=tag,
            name=run_id,
            config=args,
            dir=folder,
        )

        with open(os.path.join(ckpt_dir, "config.yaml"), "w") as f:
            f.write(pprint.pformat(args))
    else:
        ckpt_dir = None
        run=None

    
    
    logger.info(f'MASTER_ADDR: {os.environ["MASTER_ADDR"]}')
    logger.info(f'MASTER_PORT: {os.environ["MASTER_PORT"]}')
    logger.info(f'RANK: {rank}')
    logger.info(f'WORLD_SIZE: {world_size}')

    # -- set device
    if not torch.cuda.is_available():
        logger.error('CUDA not available')
        device = torch.device('cpu')
    else:
        device = torch.device(f'cuda:0')
        torch.cuda.set_device(device)

    print("Using device:", device)

    # broadcast rank 0 ckpt_dir to all other ranks
    if world_size > 1:
        ckpt_dir = broadcast_string(rank, ckpt_dir, src=0, device=device)

    print("After broadcast ckpt_dir:", ckpt_dir)

    # -- log/checkpointing paths
    log_file = os.path.join(ckpt_dir, f'r{rank}.csv')
    latest_file = f'{tag}-latest.pth.tar'
    latest_path = os.path.join(ckpt_dir, latest_file)
    load_path = None
    if load_model:
        load_path = os.path.join(folder, r_file) if r_file is not None else latest_path
        if not os.path.exists(load_path):
            load_path = None
            load_model = False
    if finetune:
        load_path = os.path.join(folder, r_file)
        if not os.path.exists(load_path):
            logger.error(f'Finetuning: {load_path} does not exist')
            return

    config_file = os.path.join(ckpt_dir, f"{tag}_r{rank}.yaml")
    with open(config_file, "w") as f:
            f.write(pprint.pformat(args))

    # -- make csv_logger
    csv_logger = CSVLogger(
        log_file,
        ('%d', 'epoch'),
        ('%d', 'itr'),
        ('%.5f', 'loss'),
        ('%.5f', 'loss-jepa'),
        ('%.5f', 'reg-loss'),
        ('%.5f', 'enc-grad-norm'),
        ('%.5f', 'pred-grad-norm'),
        ('%d', 'gpu-time(ms)'),
        ('%d', 'wall-time(ms)'),
    )

    # -- init model
    if model_architecture == "vjepa":
        from privi.pretrain.models.vjepa import train_epoch as vjepa_train_epoch
        from privi.jepa.app.vjepa.utils import init_video_model
        encoder, predictor = init_video_model(
            uniform_power=uniform_power,
            use_mask_tokens=use_mask_tokens,
            num_mask_tokens=len(cfgs_mask),
            zero_init_mask_tokens=zero_init_mask_tokens,
            device=device,
            patch_size=patch_size,
            num_frames=num_frames,
            tubelet_size=tubelet_size,
            model_name=model_name,
            crop_size=crop_size,
            pred_depth=pred_depth,
            pred_embed_dim=pred_embed_dim,
            use_sdpa=use_sdpa,
        )
    elif model_architecture == "vjepa2":
        from privi.pretrain.models.vjepa2 import train_epoch as vjepa2_train_epoch
        from privi.vjepa2.app.vjepa.utils import init_video_model
        encoder, predictor = init_video_model(
            uniform_power=uniform_power,
            use_mask_tokens=use_mask_tokens,
            num_mask_tokens=int(len(cfgs_mask) * len(dataset_fpcs)),
            zero_init_mask_tokens=zero_init_mask_tokens,
            device=device,
            patch_size=patch_size,
            max_num_frames=max_num_frames,
            tubelet_size=tubelet_size,
            model_name=model_name,
            crop_size=crop_size,
            pred_depth=pred_depth,
            pred_num_heads=pred_num_heads,
            pred_embed_dim=pred_embed_dim,
            use_sdpa=use_sdpa,
            use_silu=use_silu,
            use_pred_silu=use_pred_silu,
            wide_silu=wide_silu,
            use_rope=use_rope,
            use_activation_checkpointing=use_activation_checkpointing,
        )
    else:
        raise ValueError("Unknown architecture")
    target_encoder = copy.deepcopy(encoder)

    if compile_model:
        logger.info("Compiling encoder, target_encoder, and predictor.")
        torch._dynamo.config.optimize_ddp = False
        encoder.compile()
        target_encoder.compile()
        predictor.compile()

    mask_unit_size = patch_size

    if model_architecture == "vjepa":
        from privi.jepa.src.masks.random_tube import MaskCollator as TubeMaskCollator
        from privi.jepa.src.masks.multiblock3d import MaskCollator as MB3DMaskCollator
        from privi.jepa.app.vjepa.transforms import make_transforms
        # -- make data transforms
        if mask_type == 'multiblock3d':
            logger.info('Initializing basic multi-block mask')
            mask_collator = MB3DMaskCollator(
                crop_size=crop_size,
                num_frames=num_frames,
                patch_size=mask_unit_size,
                tubelet_size=tubelet_size,
                cfgs_mask=cfgs_mask)
        else:
            logger.info('Initializing random tube mask')
            mask_collator = TubeMaskCollator(
                crop_size=crop_size,
                num_frames=num_frames,
                patch_size=mask_unit_size,
                tubelet_size=tubelet_size,
                cfgs_mask=cfgs_mask)
        transform = make_transforms(
            random_horizontal_flip=True,
            random_resize_aspect_ratio=ar_range,
            random_resize_scale=rr_scale,
            reprob=reprob,
            auto_augment=use_aa,
            motion_shift=motion_shift,
            crop_size=crop_size,
            aug_mode=aug_mode,)
    elif model_architecture == "vjepa2":
        from privi.vjepa2.src.masks.multiseq_multiblock3d import MaskCollator
        from privi.jepa.app.vjepa.transforms import make_transforms
        mask_collator = MaskCollator(
            cfgs_mask=cfgs_mask,
            dataset_fpcs=dataset_fpcs,
            crop_size=crop_size,
            patch_size=patch_size,
            tubelet_size=tubelet_size,
        )
        transform = make_transforms(
            random_horizontal_flip=True,
            random_resize_aspect_ratio=ar_range,
            random_resize_scale=rr_scale,
            reprob=reprob,
            auto_augment=use_aa,
            motion_shift=motion_shift,
            crop_size=crop_size,
            aug_mode=aug_mode,
        )
    else:
        raise ValueError("Unknown architecture")

    # -- init data-loaders/samplers
    (_dataset, unsupervised_loader, unsupervised_sampler) = make_videodataset(
         dataset_type="pretrain",
         label_path=dataset_paths[0],
         batch_size=batch_size,
         frames_per_clip=num_frames,
         frame_step=sampling_rate,
         filter_short_videos=filter_short_videos,
         duration=duration,
         num_clips=num_clips,
         transform=transform,
         datasets_weights=datasets_weights,
         collator=mask_collator,
         num_workers=num_workers,
         world_size=world_size,
         pin_mem=pin_mem,
         rank=rank,
         log_dir=folder if log_resource_util_data else None,
         video_base_path=video_base_dir,
         crop_to_bboxes=crop_to_bboxes,
         cache_dir=cache_dir,
         prob_crop_to_bboxes=prob_crop_to_bboxes,)
    try:
        _dlen = len(unsupervised_loader)
    except Exception:  # Different interface for webdataset
        _dlen = unsupervised_loader.num_batches
    if ipe is None:
        ipe = _dlen
    logger.info(f'iterations per epoch/dataest length: {ipe}/{_dlen}')

    if use_caching:
        if world_size > 4:
            local_rank = int(os.environ['SLURM_LOCALID'])
            local_world_size = 4
        else:
            local_rank = rank
            local_world_size = world_size

        start_time = time.time()

        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            indices = range(local_rank, len(unsupervised_loader.dataset), local_world_size)
            transfered_bytes = 0
            for ret in (bar := tqdm(executor.map(lambda i: unsupervised_loader.dataset.cache_sample(i), indices), total=len(unsupervised_loader.dataset)/world_size, desc="Caching samples (0MB)", disable=True)):
                transfered_bytes += ret
                bar.set_description(f"Caching samples ({transfered_bytes / 1024**2:.1f}MB)")

        end_time = time.time()

        logger.info(f"Rank {rank} finished caching samples in {end_time - start_time:.1f}s, {transfered_bytes / 1024**2 / (end_time-start_time):.1f}MB/s")

        torch.distributed.barrier()
        logger.info(f"Finished caching samples for rank {rank}")

    # -- init optimizer and scheduler
    # init_opt is identical in VJEPA amd VJEPA 2
    optimizer, scaler, scheduler, wd_scheduler = init_opt(
        encoder=encoder,
        predictor=predictor,
        wd=wd,
        final_wd=final_wd,
        start_lr=start_lr,
        ref_lr=lr,
        final_lr=final_lr,
        iterations_per_epoch=ipe,
        warmup=warmup,
        num_epochs=num_epochs,
        ipe_scale=ipe_scale,
        mixed_precision=mixed_precision,
        betas=betas,
        eps=eps)
    encoder = DistributedDataParallel(encoder, static_graph=True)
    predictor = DistributedDataParallel(predictor, static_graph=True)
    target_encoder = DistributedDataParallel(target_encoder)
    for p in target_encoder.parameters():
        p.requires_grad = False

    # -- momentum schedule
    momentum_scheduler = (ema[0] + i*(ema[1]-ema[0])/(ipe*num_epochs*ipe_scale)
                          for i in range(int(ipe*num_epochs*ipe_scale)+1))

    start_epoch = 0
    # -- load training checkpoint
    # Checkpoint loading VJEPA and VJEPA2 is basically the same, VJEPA is just less robust
    if load_model or os.path.exists(latest_path):
        (
            encoder,
            predictor,
            target_encoder,
            optimizer,
            scaler,
            start_epoch,
        ) = load_checkpoint(
            r_path=load_path,
            encoder=encoder,
            predictor=predictor,
            target_encoder=target_encoder,
            opt=optimizer,
            scaler=scaler)
        for _ in range(start_epoch * ipe):
            scheduler.step()
            wd_scheduler.step()
            next(momentum_scheduler)
            mask_collator.step()

    if finetune:
        # Difference to laod_model: do not load optimizer state and start from epoch 0
        (
            encoder,
            predictor,
            target_encoder,
            _,
            _,
            _,
        ) = load_checkpoint(
            r_path=load_path,
            encoder=encoder,
            predictor=predictor,
            target_encoder=target_encoder,
            opt=None,
            scaler=None)

    def save_checkpoint(epoch, path, loss_meter, inference_only=False):
        if rank != 0:
            return
        save_dict = {
            'encoder': encoder.state_dict() if not inference_only else None,
            'predictor': predictor.state_dict() if not inference_only else None,
            'opt': optimizer.state_dict(),
            'scaler': None if scaler is None else scaler.state_dict(),
            'target_encoder': target_encoder.state_dict(),
            'epoch': epoch,
            'loss': loss_meter.avg,
            'batch_size': batch_size,
            'world_size': world_size,
            'lr': lr,
        }
        os.makedirs(os.path.dirname(path), exist_ok=True)
        try:
            torch.save(save_dict, path)
        except Exception as e:
            logger.info(f'Encountered exception when saving checkpoint: {e}')

    logger.info('Initializing loader...')
    loader = iter(unsupervised_loader)

    #save_checkpoint(0, os.path.join(ckpt_dir, f'untrained.pth.tar'))

    if skip_batches > 0:
        logger.info(f'Skip {skip_batches} batches')
        unsupervised_sampler.set_epoch(start_epoch)
        for itr in range(skip_batches):
            if itr % 10 == 0:
                logger.info(f'Skip {itr}/{skip_batches} batches')
            try:
                udata = next(loader)
            except Exception:
                loader = iter(unsupervised_loader)
                udata = next(loader)

    if sync_gc:
        gc.disable()
        gc.collect()

    def save_fn(epoch, loss_meter):
        if rank != 0:
            return
        if epoch % checkpoint_freq == 0 or epoch == (num_epochs - 1):
            save_checkpoint(epoch + 1, latest_path, loss_meter)
            if epoch in save_epochs or epoch == (num_epochs - 1):
                save_every_file = f'{tag}-e{epoch}.pth.tar'
                save_every_path = os.path.join(ckpt_dir, save_every_file)
                save_checkpoint(epoch + 1, save_every_path, loss_meter, inference_only=False)#
                if epoch in eval_epochs or (eval_last and epoch == (num_epochs-1)):
                    for cmd in eval_cmds:
                        cmd_f = cmd.format(epoch=epoch, ckpt_path=save_every_path, run_id=run_id, project=tag, label_path=dataset_paths[0])
                        print(f"EXECUTING: {cmd_f}")
                        ret_cmd=os.system(cmd_f)
                        print(f"  Execution finished, return code {ret_cmd}")

    # -- TRAINING LOOP

    if model_architecture == "vjepa":
        vjepa_train_epoch(
            loader=loader,
            unsupervised_loader=unsupervised_loader, 
            unsupervised_sampler=unsupervised_sampler, 
            ipe=ipe, 
            cfgs_mask=cfgs_mask, 
            device=device, 
            batch_size=batch_size, 
            num_clips=num_clips, 
            scheduler=scheduler, 
            wd_scheduler=wd_scheduler, 
            target_encoder=target_encoder, 
            encoder=encoder, 
            predictor=predictor, 
            loss_exp=loss_exp, 
            dtype=dtype, 
            mixed_precision=mixed_precision, 
            scaler=scaler, 
            optimizer=optimizer, 
            reg_coeff=reg_coeff, 
            start_epoch=start_epoch,
            num_epochs=num_epochs, 
            warmup=warmup, 
            clip_grad=clip_grad, 
            momentum_scheduler=momentum_scheduler, 
            world_size=world_size, 
            csv_logger=csv_logger, 
            rank=rank, 
            run=run, 
            log_freq=log_freq,
            save_fn=save_fn,
        )
    else:
        vjepa2_train_epoch(
            loader=loader,
            unsupervised_sampler=unsupervised_sampler, 
            unsupervised_loader=unsupervised_loader, 
            start_epoch=start_epoch,
            num_epochs=num_epochs, 
            batch_size=batch_size, 
            dataset_fpcs=dataset_fpcs, 
            ipe=ipe, 
            device=device, 
            sync_gc=sync_gc, 
            GARBAGE_COLLECT_ITR_FREQ=GARBAGE_COLLECT_ITR_FREQ, 
            scheduler=scheduler, 
            wd_scheduler=wd_scheduler, 
            target_encoder=target_encoder, 
            encoder=encoder, 
            predictor=predictor, 
            loss_exp=loss_exp, 
            dtype=dtype, 
            mixed_precision=mixed_precision, 
            scaler=scaler, 
            optimizer=optimizer, 
            momentum_scheduler=momentum_scheduler, 
            csv_logger=csv_logger, 
            log_freq=log_freq, 
            run=run,
            save_fn=save_fn,
        )        

    if cache_dir is not None:
        shutil.rmtree(cache_dir, ignore_errors=True)
        logger.info(f"Removed cache dir {cache_dir}")
