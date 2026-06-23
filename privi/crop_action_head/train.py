# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
# Modified by Felix Benjamin Mueller, 2025

import datetime
from functools import partial
import json
import math
import os
from pathlib import Path
import secrets
import sys
import traceback
from typing import List
from torch.nn import functional as F

from einops import rearrange

from privi.crop_action_head.latents.generic import BaseLatents
from privi.crop_action_head.models.generic import get_classifier
from privi.utils.misc import setup_train, count_parameters
from privi.utils.wandb import StdoutLogger, WandbFanout

import logging
import pprint

import numpy as np

import torch
import torch.multiprocessing as mp

from privi.jepa.src.utils.schedulers import (
    WarmupCosineSchedule,
    CosineWDSchedule,
)
from privi.jepa.src.utils.logging import AverageMeter

logging.basicConfig(
    format="[%(asctime)s - %(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO
)
logger = logging.getLogger()

_GLOBAL_SEED = secrets.randbits(32)
np.random.seed(_GLOBAL_SEED)
torch.manual_seed(_GLOBAL_SEED)
torch.backends.cudnn.benchmark = True

pp = pprint.PrettyPrinter(indent=4)

DEBUG_BREAK = False

class Evaluator:
    eval_dataset: object
    label_names: List[str]

    @classmethod
    def from_checkpoint(cls, args, save_dir, ckpt_paths):

        try:
            mp.set_start_method("spawn")
        except Exception:
            pass

        if not torch.cuda.is_available():
            device = torch.device("cpu")
        else:
            device = torch.device("cuda:0")
            torch.cuda.set_device(device)

        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True  # if your input sizes are not wildly changing
        torch.set_float32_matmul_precision("high")  # PyTorch 2.x; lets TF32 engage

        args_data = args.get("data")
        dataset_mode = args_data.get("mode", "per_crop")  # per_crop, per_frame, video
        ignore_labels = args_data.get("ignore_labels", [])

        args_opt = args.get("optimization")
        val_batch_size = args_opt.get("val_batch_size")
        use_bfloat16 = args_opt.get("use_bfloat16")

        args_eval_args = args.get("eval", dict())

        if dataset_mode == "video":
            from privi.crop_action_head.latents.backbone import BackboneLatents

            backbone_type = args.get("backbone_type")

            args_data_aug = args.get("data_aug", dict())
            args_model = args.get("backbone", dict())

            latents = BackboneLatents(
                args_data=args_data,
                args_data_aug=args_data_aug,
                args_model=args_model,
                backbone_type=backbone_type,
                backbone_training_mode="frozen",
                batch_size=val_batch_size,
                val_batch_size=val_batch_size,
                limit_caching=20 if DEBUG_BREAK else -1,
                device=device,
            )

        obj = cls(latents, args_eval_args, use_bfloat16, save_dir)

        classifiers = []

        for ckpt_path in ckpt_paths:

            checkpoint = torch.load(ckpt_path, map_location=torch.device("cpu"))
            params = checkpoint.get("classifier_args", None)
            epoch = checkpoint["epoch"]

            if params is None:
                logger.warning("No head parameters in checkpoint, using values from eval cfg file")
                params = dict(
                    type=args.get("head_type", "vjepa"),
                    embed_dim_global=args_data.get("embed_dim_global"),
                    embed_dim_local=args_data.get("embed_dim_local"),
                    num_classes=obj.eval_dataset.num_classes,
                    num_scales=len(args_model["out_layers"]),
                    **args.get("head", dict()),
                )

            classifier = get_classifier(**params).to(device)

            pretrained_dict = checkpoint["classifier"]

            if "module." in list(pretrained_dict.keys())[0]:
                pretrained_dict = {k.replace("module.", ""): v for k, v in pretrained_dict.items()}
                logging.info("Removing module. from keys in pretrained_dict")

            msg = classifier.load_state_dict(pretrained_dict)
            logger.info(f"loaded pretrained classifier from epoch {epoch} with msg: {msg}")

            del checkpoint

            classifiers.append(classifier)

        obj.add_classifiers(classifiers)

        ignored_label_indices, _ = generate_ignored_label_indices(ignore_labels, obj, device)
        obj.add_loss(None, ignored_label_indices)

        return obj

    def __init__(self, latents: BaseLatents, eval_args, use_bfloat16, save_dir, run=None):

        self.eval_dataset_instances = {}
        for split in ["val", "train", "test"]:
            ds = latents.get_dataset(split)
            if ds is not None:
                self.eval_dataset_instances[split] = ds

        self.eval_dataset = next(iter(self.eval_dataset_instances.values()))
        self.label_names = self.eval_dataset.label_names

        self.latents = latents
        self.use_bfloat16 = use_bfloat16
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

        self.run = run
        self.classifiers = None
        self.loss_fn = None
        self.ignored_label_indices = None


    def add_classifiers(self, classifiers):

        self.classifiers = classifiers

        if self.run is None:
            # Fake wandb logger for subprocess calls, logs to stdout
            self.run = [StdoutLogger()] * len(classifiers)

    def add_loss(self, loss_fn, ignored_label_indices):
        self.loss_fn = loss_fn
        self.ignored_label_indices = ignored_label_indices

    def run_validation(self, split: str, epoch=None, step=None, total_train=None):

        try:
            latents_producer = self.latents.get_iter(split)
        except Exception:
            logger.info(f"Skip validaton for split {split}, no data provided")
            return

        for h in range(len(self.classifiers)):

            if epoch is not None:
                log_epoch = epoch + step / total_train
                epoch_suffix = f"_epoch{(log_epoch):05.2f}_step{step}"
            else:
                log_epoch = None
                epoch_suffix = ""

            logger.info(f"Running validation at log epoch {log_epoch}")

            dataset = self.eval_dataset_instances.get(split)
            dataset_evaluator = dataset.evaluator() if dataset is not None else None

            loss = self.run_validation_one_head(
                self.classifiers[h],
                latents_producer=latents_producer,
                loss_fn=self.loss_fn,
                ignored_label_indices=self.ignored_label_indices,
                dataset_evaluator=dataset_evaluator,
            )
            self.run[h].log(
                {
                    f"metrics/{split}_loss": loss,
                    "epoch": log_epoch,
                }
            )

            if dataset_evaluator is not None:
                results = dataset_evaluator.metrics()
                # Save full results to disk
                results_path = os.path.join(
                    self.save_dir,
                    f"head{h}_{split}_logits{epoch_suffix}_ava_results.json"
                )
                with open(results_path, "w") as f:
                    json.dump(results, f, indent=4)
                # Log filtered subset to wandb
                wandb_metrics = dataset_evaluator.wandb_log_metrics(results, split, log_epoch, h)
                self.run[h].log(wandb_metrics)

    def eval_one_ckpt(self, head_idx, split, log_epoch, logit_name):
        try:
            latents_producer = self.latents.get_iter(split)
        except Exception:
            logger.info(f"Skip validaton for split {split}, no data provided")
            return

        classifier = self.classifiers[head_idx]

        logger.info(f"Running validation for {logit_name}")

        dataset = self.eval_dataset_instances.get(split)
        dataset_evaluator = dataset.evaluator() if dataset is not None else None

        loss = self.run_validation_one_head(
            classifier,
            latents_producer=latents_producer,
            loss_fn=self.loss_fn,
            ignored_label_indices=self.ignored_label_indices,
            dataset_evaluator=dataset_evaluator,
        )

        if dataset_evaluator is not None:
            results = dataset_evaluator.metrics()
            # Save full results to disk
            results_path = os.path.join(
                self.save_dir,
                f"{logit_name}_ava_results.json"
            )
            with open(results_path, "w") as f:
                json.dump(results, f, indent=4)
            # Log filtered subset to wandb
            wandb_metrics = dataset_evaluator.wandb_log_metrics(results, split, log_epoch, head_idx)
            self.run[head_idx].log(wandb_metrics)

    def run_validation_one_head(
        self, model, latents_producer, loss_fn, ignored_label_indices, dataset_evaluator=None
    ):

        loss_meter = AverageMeter()

        for itr, data in enumerate(latents_producer):

            if DEBUG_BREAK and itr > 2:
                break

            loss, outputs, batch_size, local_data = run_one_step(
                model,
                data,
                training=False,
                use_bfloat16=self.use_bfloat16,
                loss_fn=loss_fn,
                ignored_label_indices=ignored_label_indices,
            )
            if loss is None:
                loss = 0.0
            else:
                loss = loss.item()

            if dataset_evaluator is not None:
                dataset_evaluator.add_batch(local_data, outputs)

            loss_meter.update(loss, n=batch_size)
            if DEBUG_BREAK or itr % 10 == 0:

                logger.info(
                    "[%s] EVAL [%5d/%5d] (loss: %.3f) [mem: %.2e]"
                    % (
                        datetime.datetime.now().isoformat(),
                        itr,
                        len(latents_producer),
                        loss,
                        torch.cuda.max_memory_allocated() / 1024.0**2,
                    )
                )

        return loss_meter.avg


class CropActionHeadTrainer:

    def __init__(self, args, ckpt_dir, run_id, group_id=None):
        super().__init__()

        pp.pprint(args)

        # ----------------------------------------------------------------------- #
        #  PASSED IN PARAMS FROM CONFIG FILE
        # ----------------------------------------------------------------------- #

        # -- DATA
        args_data = args.get("data")
        embed_dim_global = args_data.get("embed_dim_global")
        embed_dim_local = args_data.get("embed_dim_local")
        ignore_labels = args_data.get("ignore_labels", [])
        self.dataset_mode = args_data.get("mode", "per_crop")  # per_crop, per_frame, video

        if self.dataset_mode in ["video", "raw"]:

            args_data_aug = args.get("data_aug", dict())
            args_model = args.get("backbone", dict())
            self.backbone_type = args.get("backbone_type")
            self.backbone_training_mode = args_model.get("training_mode", "frozen")
        else:
            self.backbone_training_mode = "frozen"
            args_data_aug = dict()

        # -- OPTIMIZATION
        args_opt = args.get("optimization")
        self.batch_size = args_opt.get("batch_size")
        self.val_batch_size = args_opt.get("val_batch_size", self.batch_size)
        self.num_epochs = args_opt.get("num_epochs")
        wd = args_opt.get("weight_decay")
        self.lr = args_opt.get("lr")
        start_lr = args_opt.get("start_lr", self.lr)
        final_lr = args_opt.get("final_lr")
        warmup = args_opt.get("warmup")
        loss_fn_str = args_opt.get("loss_fn", "cross_entropy")
        batch_adjust_lr = args_opt.get("batch_adjust_lr", False)
        self.clip_grad_norm_value = args_opt.get("clip_grad_norm_value", 1.0)

        self.use_bfloat16 = args_opt.get("use_bfloat16")
        if self.use_bfloat16 and not args_model["dtype"] == "bfloat16":
            logger.warning(
                f"Inefficient setting: head uses bfloat16, but backbone uses {args_model['dtype']}"
            )

        if self.backbone_training_mode != "frozen" and args_model["dtype"] == "float16":
            raise ValueError(f"Cannot train fp16 backbone, grad scaling not implemented")

        if self.backbone_training_mode != "frozen":
            backbone_start_lr = args_opt.get("backbone_start_lr", start_lr)
            backbone_lr = args_opt.get("backbone_lr", self.lr)
            backbone_final_lr = args_opt.get("backbone_final_lr", final_lr)
            backbone_wd = args_opt.get("backbone_wd", wd)

        if batch_adjust_lr:
            self.lr = self.lr * math.sqrt(self.batch_size / 8.0)
            start_lr = start_lr * math.sqrt(self.batch_size / 8.0)
            final_lr = final_lr * math.sqrt(self.batch_size / 8.0)
            logger.info(
                f"Adjusted lr to {self.lr} and start_lr to {start_lr} and final_lr to {final_lr} based on batch size {self.batch_size}"
            )

        if self.dataset_mode == "video":
            # When using the backbone, train several classifiers in the same run, else in separate processes
            self.n_classifiers = int(args_opt.get("n_classifiers", 1))
            self.buffer_size = int(args_opt.get("buffer_size", 8))
        else:
            self.n_classifiers = 1
            self.buffer_size = 0

        head_type = args.get("head_type", "vjepa")
        args_head = args.get("head", dict())

        args_eval_args = args.get("eval", dict())

        # -- EXPERIMENT-ID/TAG (optional)
        args_meta = args.get("meta", dict())
        project_id = args_meta.get("project_id", "default")
        save_dir = args_meta.get("save_dir", "./checkpoints/")
        self.val_every_n_samples = args_meta.get("val_every_n_samples", 5000)
        # multi-classifier settings

        args["global_seed"] = _GLOBAL_SEED
        print("SEED", _GLOBAL_SEED)

        # ----------------------------------------------------------------------- #

        try:
            mp.set_start_method("spawn")
        except Exception:
            pass

        if not torch.cuda.is_available():
            self.device = torch.device("cpu")
        else:
            self.device = torch.device("cuda:0")
            torch.cuda.set_device(self.device)

        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True  # if your input sizes are not wildly changing
        torch.set_float32_matmul_precision("high")  # PyTorch 2.x; lets TF32 engage

        self.world_size, self.rank = 1, 0  # init_distributed()

        logger.info(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES')}")
        logger.info(f"Cuda devices visible: {torch.cuda.device_count()}")

        self.ckpt_dir = ckpt_dir

        # -- log/checkpointing paths

        if self.rank == 0:

            os.makedirs(self.ckpt_dir, exist_ok=True)

            args["slurm_id"] = os.getenv("SLURM_JOB_ID", "noslurm")

            self.run = WandbFanout(
                self.n_classifiers, project_id, group_id, run_id, args, save_dir, disabled=DEBUG_BREAK
            )

        self.ckp_stem = "head"
        config_file = os.path.join(self.ckpt_dir, f"{self.ckp_stem}_r{self.rank}.yaml")
        self.latest_path = os.path.join(self.ckpt_dir, f"{self.ckp_stem}-latest.pth.tar")

        with open(config_file, "w") as f:
            f.write(pprint.pformat(args))

        ### Init data

        if self.dataset_mode == "video":
            params = dict(
                args_data=args_data,
                args_data_aug=args_data_aug,
                args_model=args_model,
                backbone_type=self.backbone_type,
                backbone_training_mode=self.backbone_training_mode,
                batch_size=self.batch_size,
                val_batch_size=self.val_batch_size,
                limit_caching=2 if DEBUG_BREAK else -1,
                device=self.device,
            )

            if self.n_classifiers > 1:
                from privi.crop_action_head.latents.multihead_backbone import (
                    MultiheadBackboneLatents,
                )

                self.latents = MultiheadBackboneLatents(
                    **params, n_classifiers=self.n_classifiers, buffer_size=self.buffer_size
                )
            else:
                from privi.crop_action_head.latents.backbone import BackboneLatents

                self.latents = BackboneLatents(**params)

        elif self.dataset_mode == "raw":

            params = dict(
                args_data=args_data,
                args_data_aug=args_data_aug,
                args_model=args_model,
                backbone_type=self.backbone_type,
                backbone_training_mode=self.backbone_training_mode,
                batch_size=self.batch_size,
                val_batch_size=self.val_batch_size,
                limit_caching=2 if DEBUG_BREAK else -1,
                device=self.device,
            )

            assert self.n_classifiers == 1, "Raw video only works with one classifier!"

            from privi.crop_action_head.latents.video import VideoLatents

            self.latents = VideoLatents(**params)


        ipe = len(self.latents.get_iter("train"))

        if self.backbone_training_mode != "frozen":
            assert self.dataset_mode == "video", "Need dataset_mode video for backbone training"

            self.backbone_optimizer, self.backbone_scheduler, self.backbone_wd_scheduler = (
                init_opt(
                    classifier=self.latents.backbone,
                    wd=backbone_wd,
                    start_lr=backbone_start_lr,
                    ref_lr=backbone_lr,
                    final_lr=backbone_final_lr,
                    iterations_per_epoch=ipe,
                    warmup=warmup,
                    num_epochs=self.num_epochs,
                )
            )
            print(
                f"BACKBONE PARAMETERS (trainable): {count_parameters(self.latents.backbone) / 1000 / 1000:.2f}M"
            )

        else:
            self.backbone = None

        self.evaluator = Evaluator(
            self.latents,
            args_eval_args,
            use_bfloat16=self.use_bfloat16,
            save_dir=self.ckpt_dir,
            run=self.run,
        )

        # Initialize model

        if self.dataset_mode == "video":
            num_scales = len(args_model.get("out_layers", [None]))
        else:
            num_scales = 1

        self.classifier_args = dict(
            type=head_type,
            embed_dim_global=embed_dim_global,
            embed_dim_local=embed_dim_local,
            num_classes=self.evaluator.eval_dataset.num_classes,
            num_scales=num_scales,
            **args_head,
        )

        self.classifiers = [
            get_classifier(
                **self.classifier_args
            ).to(self.device)
            for _ in range(self.n_classifiers)
        ]

        print(f"HEAD PARAMETERS: {count_parameters(self.classifiers[0]) / 1000 / 1000:.2f}M")

        self.evaluator.add_classifiers(self.classifiers)

        self.total_steps = ipe * self.num_epochs
        self.ipe = ipe

        # -- optimizer and scheduler
        self.optimizers = []
        self.schedulers, self.wd_schedulers = [], []
        for head in self.classifiers:
            opt, sch, wdsch = init_opt(
                classifier=head,
                wd=wd,
                start_lr=start_lr,
                ref_lr=self.lr,
                final_lr=final_lr,
                iterations_per_epoch=ipe,
                warmup=warmup,
                num_epochs=self.num_epochs,
            )
            self.optimizers.append(opt)
            self.schedulers.append(sch)
            self.wd_schedulers.append(wdsch)

        self.setup_loss(loss_fn_str, ignore_labels)
        self.evaluator.add_loss(self.loss_fn, self.ignored_label_indices)

        # -- load training checkpoint
        self.start_epoch = 0


    def setup_loss(self, loss_fn_str, ignore_labels):

        assert not ignore_labels or loss_fn_str in [
            "cross_entropy",
            "asl_ls",
            "eqlv2"
        ], "Ignoring labels is only supported for cross entropy loss"

        self.ignored_label_indices, weights = generate_ignored_label_indices(
            ignore_labels, self.evaluator, self.device
        )

        if loss_fn_str == "cross_entropy":
            if self.evaluator.eval_dataset.multi_label:
                self.loss_fn = torch.nn.BCEWithLogitsLoss(weight=weights, reduction="none")
            else:

                def _loss_fn(pred, gt):
                    B = pred.shape[0]
                    assert gt.shape[0] == B

                    pred = rearrange(pred, "batch views classes -> (batch views) classes")

                    if len(gt.shape) == 3:
                        # single label one hot encoded
                        gt = torch.argmax(gt, dim=-1)
                    gt = rearrange(gt, "batch views -> (batch views)")

                    loss = F.cross_entropy(pred, gt, reduction="none")

                    loss = rearrange(loss, "(batch views) -> batch views", batch=B)

                    return loss

                self.loss_fn = _loss_fn

        elif loss_fn_str == "eqlv2":
            from privi.modules.eqlv2_loss import EQLv2

            # works for single- and multi-label

            loss_module = EQLv2(
                num_classes=self.evaluator.eval_dataset.num_classes,
                class_weight=weights,
            ).to(self.device)

            def _loss_fn(pred, gt):

                B = pred.shape[0]
                assert gt.shape[0] == B

                pred = rearrange(pred, "batch views classes -> (batch views) classes")

                if len(gt.shape) == 2:
                    # single label annotations-> make to fake multi-label
                    gt = torch.nn.functional.one_hot(gt, num_classes=self.evaluator.eval_dataset.num_classes)
                gt = rearrange(gt, "batch views classes -> (batch views) classes")

                loss = loss_module(pred, gt, expand_target=False)

                loss = rearrange(loss, "(batch views) -> batch views", batch=B)

                return loss

            self.loss_fn = _loss_fn

        elif loss_fn_str == "focal" and not self.evaluator.eval_dataset.multi_label:

            def focal_loss(inputs, targets, alpha=1.0, gamma=2.0, reduction="none"):
                logp = F.cross_entropy(inputs, targets, reduction="none")
                p = torch.exp(-logp)
                loss = alpha * (1 - p) ** gamma * logp
                return loss.mean() if reduction == "mean" else loss
            
            self.loss_fn = focal_loss

        elif loss_fn_str == "focal" and self.evaluator.eval_dataset.multi_label:
            from torchvision.ops import sigmoid_focal_loss

            self.loss_fn = partial(sigmoid_focal_loss, reduction="none")
        else:
            raise ValueError(f"Unknown loss function: {loss_fn_str}. Supported: 'cross_entropy'")

    def save_checkpoint(self, head_idx, epoch):
        save_dict = {
            "classifier": self.classifiers[head_idx].state_dict(),
            "classifier_args": self.classifier_args,
            "opt": self.optimizers[head_idx].state_dict(),
            "epoch": epoch,
            "batch_size": self.batch_size,
            "world_size": self.world_size,
            "lr": self.lr,
            "backbone": (
                self.latents.backbone.state_dict()
                if self.backbone_training_mode != "frozen"
                else None
            ),
        }
        if self.rank == 0:
            torch.save(
                save_dict,
                os.path.join(self.ckpt_dir, f"{self.ckp_stem}-head{head_idx}-latest.pth.tar"),
            )
            torch.save(
                save_dict,
                os.path.join(
                    self.ckpt_dir, f"{self.ckp_stem}-head{head_idx}-epoch{epoch:05.2f}.pth.tar"
                ),
            )

    def train(self):

        logger.info(
            f"Starting training with {self.n_classifiers} heads and buffer size {self.buffer_size}"
        )

        print("FlashAttention enabled:      ", torch.backends.cuda.flash_sdp_enabled())
        print("Memory-efficient enabled:    ", torch.backends.cuda.mem_efficient_sdp_enabled())
        print("Math (standard) enabled:     ", torch.backends.cuda.math_sdp_enabled())
        print("cuDNN SDPA enabled:          ", torch.backends.cuda.cudnn_sdp_enabled())

        samples_seen = 0
        eval_ctr = 0
        for epoch in range(self.start_epoch, self.num_epochs):
            logger.info("[%s] Epoch %d", datetime.datetime.now().isoformat(), epoch + 1)

            samples_seen, loss_meters, eval_ctr = self.train_epoch(epoch, samples_seen, eval_ctr)

            

            # save checkpoints per head
            for h in range(self.n_classifiers):
                self.run.log(
                    h,
                    {
                        f"metrics/train_loss": loss_meters[h].avg,
                        "epoch": epoch + 1,
                    },
                )

                self.save_checkpoint(h, epoch+1)


    def train_epoch(self, epoch, samples_seen, eval_ctr):

        loss_meters = [AverageMeter() for _ in range(self.n_classifiers)]

        latents_producer = self.latents.get_iter("train")
        train_iter = iter(latents_producer)
        total_backbone_batches = len(latents_producer)

        step = 0
        while True:
            # the iterator yields shuffled batches round robin for all classifiers
            # if n_classifiers > 1. Otherwise its just a regular iterator
            for h in range(self.n_classifiers):
                try:
                    batch = next(train_iter)
                except StopIteration:
                    if epoch == self.num_epochs -1 :
                        # reduce number of offloads
                        if eval_ctr % 2 == 0:
                            self.evaluator.run_validation("val", epoch, step, total_backbone_batches)
                            self.latents.offload("val")
                            self.evaluator.run_validation("test", epoch, step, total_backbone_batches)
                        else:
                            self.evaluator.run_validation("test", epoch, step, total_backbone_batches)
                            if "test" in self.evaluator.eval_dataset_instances:
                                self.latents.offload("test")
                            self.evaluator.run_validation("val", epoch, step, total_backbone_batches)
                        samples_seen = 0
                        eval_ctr += 1
                    return samples_seen, loss_meters, eval_ctr

                self.classifiers[h].step(step + epoch * self.ipe, self.total_steps)
                self.schedulers[h].step()
                self.wd_schedulers[h].step()
                if self.backbone_training_mode != "frozen":
                    self.backbone_scheduler.step()
                    self.backbone_wd_scheduler.step()

                loss, _, batch_size, _ = run_one_step(
                    self.classifiers[h],
                    batch,
                    training=True,
                    use_bfloat16=self.use_bfloat16,
                    loss_fn=self.loss_fn,
                    ignored_label_indices=self.ignored_label_indices,
                )

                loss.backward()
                params = list(self.classifiers[h].parameters())
                if self.backbone_training_mode != "frozen":
                    params += list(self.latents.backbone.parameters())
                grad_norm = torch.nn.utils.clip_grad_norm_(params, self.clip_grad_norm_value)

                self.optimizers[h].step()
                if self.backbone_training_mode != "frozen":
                    self.backbone_optimizer.step()

                self.optimizers[h].zero_grad(set_to_none=True)
                if self.backbone_training_mode != "frozen":
                    self.backbone_optimizer.zero_grad(set_to_none=True)

                loss = loss.item()
                loss_meters[h].update(loss, n=batch_size)

                if step % 10 == 0:
                    self.run.log(
                        h,
                        {
                            "grad_norm": grad_norm,
                            "lr": self.optimizers[h].param_groups[0]["lr"],
                            "wd": self.optimizers[h].param_groups[0].get("weight_decay", 0.0),
                            "cuda_mem": torch.cuda.max_memory_allocated() / 1024.0**2,
                            "loss": loss,
                        },
                    )

            if DEBUG_BREAK or step % 100 == 0:
                logger.info(
                    "TRAIN [%5d/%5d] [mem: %.2e]"
                    % (
                        step,
                        total_backbone_batches,
                        torch.cuda.max_memory_allocated() / 1024.0**2,
                    )
                )

            step += 1
            samples_seen += self.batch_size

            if (DEBUG_BREAK and step > 50) or samples_seen > self.val_every_n_samples or (step == 200 and epoch == 0):
                # reduce number of offloads
                if eval_ctr % 2 == 0:
                    self.evaluator.run_validation("val", epoch, step, total_backbone_batches)
                    self.latents.offload("val")
                    self.evaluator.run_validation("test", epoch, step, total_backbone_batches)
                else:
                    self.evaluator.run_validation("test", epoch, step, total_backbone_batches)
                    if "test" in self.evaluator.eval_dataset_instances:
                        self.latents.offload("test")
                    self.evaluator.run_validation("val", epoch, step, total_backbone_batches)
                log_epoch = epoch + step / total_backbone_batches
                for h in range(self.n_classifiers):
                    self.save_checkpoint(h, log_epoch)
                samples_seen = 0
                eval_ctr += 1

            if DEBUG_BREAK and step > 50:
                return samples_seen, loss_meters, eval_ctr


def init_opt(
    classifier,
    iterations_per_epoch,
    start_lr,
    ref_lr,
    warmup,
    num_epochs,
    wd=1e-6,
    final_wd=1e-6,
    final_lr=0.0,
):
    param_groups = [
        {
            "params": (
                p
                for n, p in classifier.named_parameters()
                if ("bias" not in n) and (len(p.shape) != 1)
            )
        },
        {
            "params": (
                p for n, p in classifier.named_parameters() if ("bias" in n) or (len(p.shape) == 1)
            ),
            "WD_exclude": True,
            "weight_decay": 0,
        },
    ]

    logger.info("Using AdamW")
    optimizer = torch.optim.AdamW(param_groups)
    scheduler = WarmupCosineSchedule(
        optimizer,
        warmup_steps=int(warmup * iterations_per_epoch),
        start_lr=start_lr,
        ref_lr=ref_lr,
        final_lr=final_lr,
        T_max=int(num_epochs * iterations_per_epoch),
    )
    wd_scheduler = CosineWDSchedule(
        optimizer, ref_wd=wd, final_wd=final_wd, T_max=int(num_epochs * iterations_per_epoch)
    )
    return optimizer, scheduler, wd_scheduler


def generate_ignored_label_indices(ignore_labels, evaluator, device):

    if ignore_labels:

        weights = torch.ones(
            evaluator.eval_dataset.num_classes, dtype=torch.float32, device=device
        )
        assert evaluator.eval_dataset.num_classes == len(
            evaluator.label_names
        ), f"Number of classes {evaluator.eval_dataset.num_classes} does not match number of label names {len(evaluator.label_names)}"
        ignored_label_indices = []
        for i, name in enumerate(evaluator.label_names):
            if name in ignore_labels:
                print(f"Ignoring label {name} with index {i}")
                weights[i] = 0.0
                ignored_label_indices.append(i)
    else:
        weights = None
        ignored_label_indices = None

    return ignored_label_indices, weights


def run_one_step(
    model,
    batch,
    training: bool,
    use_bfloat16: bool,
    loss_fn,
    ignored_label_indices,
    view_agg_fn="mean",
):
    model.train(mode=training)

    local_data = batch["local"]
    global_data = batch["global"]

    local_latents = local_data["patch_tokens"]
    batch_size, num_crops, num_views = local_latents[0].shape[:3]

    label = local_data["label"]
    local_latents = [
        rearrange(
            l,
            "b crops views c t h w -> (b views) crops c t h w",
        )
        for l in local_latents
    ]

    bbox = local_data["bbox_xyxy_rel"]
    present_crops = local_data["present_crops"]

    if num_views > 1:
        # this solution means that current all left, center, and right views interact with
        # the respective left, center, and right views of every crop. This is probably not what we want.
        # But a better solution is non-trivial
        # bbox = repeat(bbox, "b crops dim -> (b views) crops dim", views=num_views)
        # present_crops_w_views = repeat(
        #    present_crops, "b crops -> (b views) crops", views=num_views
        # )

        bbox = (
            bbox.unsqueeze(1)
            .expand(batch_size, num_views, *bbox.shape[1:])
            .reshape(batch_size * num_views, *bbox.shape[1:])
        )
        present_crops_w_views = (
            present_crops.unsqueeze(1)
            .expand(batch_size, num_views, -1)
            .reshape(batch_size * num_views, -1)
        )
    else:
        present_crops_w_views = present_crops

    if global_data is not None:
        global_latents = [
            rearrange(
                l,
                "b crops views c t h w -> (b views) crops c t h w",
            )
            for l in global_data["patch_tokens"]
        ]
    else:
        global_latents = None

    ctx = torch.inference_mode if not training else torch.enable_grad
    with (
        ctx(),
        torch.amp.autocast("cuda", dtype=torch.bfloat16 if use_bfloat16 else torch.float32),
    ):
        outputs = model(
            x_local=local_latents,
            x_global=global_latents,
            bbox=bbox,
            present_crops=present_crops_w_views,
        )

    outputs = rearrange(
        outputs,
        "(b views) crops classes -> b crops views classes",
        b=batch_size,
        views=num_views,
    )

    if view_agg_fn == "mean":
        mean_outputs = outputs.mean(dim=2)  # Average over views
    elif view_agg_fn == "add":
        mean_outputs = outputs.sum(dim=2)  # Average over views
    else:
        raise ValueError(f"Unknown view agg func {view_agg_fn}")


    if loss_fn is not None:
        loss = loss_fn(mean_outputs, label)

        loss = loss * present_crops.float().unsqueeze(-1)  # Apply present crops mask
        loss = loss.mean()
    else:
        loss = None

    with torch.no_grad():
        outputs = outputs[
            present_crops.bool()
        ]  # reduce output to present crops only, shape [B*crops, views, classes]

        if ignored_label_indices is not None:
            mask = torch.ones_like(outputs, dtype=torch.bool)
            mask[:, :, ignored_label_indices] = False

            # this returns a new tensor, leaves `outputs` intact under the hood
            outputs = outputs.masked_fill(~mask, 0)

    return loss, outputs, batch_size, local_data


def main(cfg, ckpt_dir, run_id, group_id, debug_break=False):
    global DEBUG_BREAK
    DEBUG_BREAK = debug_break
    # Initialize the training process
    trainer = CropActionHeadTrainer(
        args=cfg,
        ckpt_dir=ckpt_dir,
        run_id=run_id,
        group_id=group_id,
    )
    trainer.train()



def main_eval(cfg, cfg_path):
    save_dir = os.path.join(cfg["meta"]["save_dir"], Path(cfg["ckpts"]).stem + "_" + Path(cfg_path).stem)
    os.makedirs(save_dir, exist_ok=True)

    with open(os.path.join(save_dir, "head_r0.yaml"), "w") as f:
        import yaml
        yaml.dump(cfg, f)

    if cfg.get("head_epochs", None) is not None:
        ckpts = [
            os.path.join(
                cfg["ckpts"],
                f"head-head{idx}-epoch{epoch:05.2f}.pth.tar",
            )
            for idx, epoch in enumerate(cfg["head_epochs"])
        ]
    else:
        import glob 
        ckpts = glob.glob(f"{cfg['ckpts']}/*-epoch*.pth.tar")

    pprint.pprint(ckpts)

    evaluator = Evaluator.from_checkpoint(cfg, save_dir, ckpts)
    
    for split in ["val", "test"]:
        for idx, ckpt_path in enumerate(ckpts):
        
            try:
                
                head_parts = Path(ckpt_path).name.removesuffix(".pth.tar").split("-")
                log_epoch = float(head_parts[-1].removeprefix("epoch"))
                logit_name = f"{head_parts[1]}_{split}_logits_{head_parts[-1]}_step0"
            except Exception as e:
                print("FAIL")
                print(ckpt_path, flush=True)
                raise e
        
            evaluator.eval_one_ckpt(
                head_idx=idx,
                split=split,
                log_epoch=log_epoch,
                logit_name=logit_name,
            )
        evaluator.latents.offload(split)


if __name__ == "__main__":
    import argparse
    from privi.utils.config import apply_overrides

    parser = argparse.ArgumentParser(description="Train a crop action head")
    parser.add_argument("--config", type=str, required=False, help="Path to config file")
    parser.add_argument("--eval", action="store_true")
    parser.add_argument(
        "-o",
        "--override",
        nargs=2,
        action="append",
        metavar=("KEY", "VALUE"),
        help="Override a config entry, e.g. -o data.path.train /new/train",
    )
    args = parser.parse_args()

    if args.config is None:
        # for interactive debugging
        args.config = "configs/frozen_head/chimpact.yaml"
        args.override = [
            ("meta.n_trials", 1),
            ("data.num_workers", 0),
            ("optimization.batch_size", 2),
            ("optimization.val_batch_size", 4),
            ("optimization.n_classifiers", 2),
            ("optimization.buffer_size", 16),
            ("data.memory_caching", False),
        ]
        DEBUG_BREAK = True

    print(f"Using config file: {args.config}")

    # Load config file
    import yaml

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    overrides: list[tuple[str, str]] = []
    if args.override:
        overrides = [(k, v) for k, v in args.override]

    # then:
    apply_overrides(cfg, overrides)

    if args.eval:
        main_eval(cfg, args.config)

    else:

        # Only run test eval when in eval mode
        if 'data' in cfg and 'paths' in cfg['data'] and 'test_local' in cfg['data']['paths']:
            del cfg['data']['paths']['test_local']

        args_meta = cfg.get("meta", dict())

        if "project_id" not in args_meta:
            args_meta["project_id"] = Path(args.config).parent.name

        project_id = args_meta["project_id"]
        save_dir = args_meta.get("save_dir", "./checkpoints/")

        mp.set_start_method("spawn", force=True)

        n_trials = args_meta.get("n_trials", 1)
        if n_trials > 1 and not DEBUG_BREAK:
            group_id, base_ckpt_dir = setup_train(project_id, save_dir, Path(args.config).stem, None)

            processes = []

            for trial in range(n_trials):
                run_id = f"{group_id}-{trial+1:02d}"
                ckpt_dir = os.path.join(base_ckpt_dir, run_id)

                cfg["meta"]["run_id"] = run_id
                logger.info(f"Running trial {trial + 1}/{n_trials} with run_id {run_id}")

                p = mp.Process(
                    target=main,
                    args=(cfg, ckpt_dir, run_id, group_id, DEBUG_BREAK),
                )
                p.start()
                processes.append(p)

            for p in processes:
                p.join()
                if p.exitcode != 0:
                    logger.error(f"Trial process (PID {p.pid}) exited with code {p.exitcode}")
        else:
            run_id, ckpt_dir = setup_train(project_id, save_dir, Path(args.config).stem, None)
            main(cfg, ckpt_dir, run_id, group_id=run_id, debug_break=DEBUG_BREAK)
