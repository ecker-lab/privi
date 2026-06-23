
import itertools
import time
from einops import rearrange
import torch
from tqdm import tqdm
from privi.backbones.generic import get_backbone, reshape_latents
from privi.crop_action_head.latents.backbone import make_dataloader
from privi.crop_action_head.latents.generic import BaseLatents, LatentsIter
from torch.utils.data import Dataset, DataLoader
import logging

import numpy as np
import secrets

import gc, ctypes, os

from privi.utils.misc import normalize_dtype, timer

logger = logging.getLogger(__name__)

class _VideoLatentsIter(LatentsIter):

    def __init__(self, backbone_latents: "VideoLatents", local_loader, global_loader, device):
        self.local_loader = local_loader
        self.global_loader = global_loader
        self.backbone_latents = backbone_latents
        self.device = device

    def __iter__(self):
        for l, g in zip(self.local_loader, self.global_loader):
            assert g is None, "global context not implemented"

            buffer, label, clip_indices, info = l

            sample = {}

            sample["label"] = label.unsqueeze(1).to(self.device, non_blocking=True)
            sample["bbox_xyxy_rel"] = info["bbox_xyxy_rel"].unsqueeze(1)
            sample["present_crops"] = torch.ones(sample["label"].shape[:2], dtype=torch.bool, device=self.device)
            sample["index"] = info["index"].unsqueeze(1).to(self.device)
            sample["patch_tokens"] = [
                torch.stack([torch.stack(row, dim=1) for row in buffer], dim=1).to(self.device) # batch, crops, views, channel, time, height, width
                # Not 100% sure about crop vs view though 
            ]

            yield {"local": sample, "global": None}

    def __len__(self):
        return len(self.local_loader)

    def num_crops(self):
        return self.local_loader.dataset.num_crops()

class VideoLatents(BaseLatents):

    """
    Fake latent representation to just pass through videos to a X3D head

    """
    backbone: torch.nn.Module
    cached_eval: bool = False # whether eval sets have been cached in RAM

    def __init__(self, args_data, args_data_aug, args_model, backbone_type, backbone_training_mode, batch_size, val_batch_size, limit_caching, device, **kwargs):
        super().__init__(args_data)

        self.resolution = args_data.get("resolution", 224)
        self.frames_per_clip = args_data.get("frames_per_clip", 16)
        self.batch_size = batch_size
        self.val_batch_size = val_batch_size
        self.limit_caching = limit_caching
        self.backbone_training_mode = backbone_training_mode
        self.backbone_type = backbone_type
        self.device = device
        self.dtype_cache = normalize_dtype(args_model["dtype"])

        self.setup_data(args_data, args_data_aug)

    def setup_data(self, args_data, args_data_aug):

        base_path = args_data.get("data_path")
        num_workers = args_data.get("num_workers", 12)
        num_views_per_segment_train = args_data.get("num_views_per_segment_train", 1)
        num_views_per_segment_val = args_data.get("num_views_per_segment_val", 1)
        num_segments = 1
        dataset_type = "VideoDataset"
        frame_step = 4
        clip_duration = None
        normalize = args_data.get("normalize", "default")
        eval_resize_mode = args_data.get("eval_resize_mode", "crop")
        crop_to_bboxes = args_data.get("crop_to_bboxes", True)
        temporal_crop_frames = args_data.get("temporal_crop_frames", None)
        use_caching = args_data.get("use_caching", False)

        ar_range = args_data_aug.get('random_resize_aspect_ratio', [3/4, 4/3])
        rr_scale = args_data_aug.get('random_resize_scale', [0.08, 1.0])
        motion_shift = args_data_aug.get('motion_shift', False)
        reprob = args_data_aug.get('reprob', 0.25)
        use_aa = args_data_aug.get('auto_augment', False)
        random_horizontal_flip = args_data_aug.get('random_horizontal_flip', False)
        crop_increase_factor = args_data_aug.get('crop_increase_factor', 0.25)
        aug_mode = args_data_aug.get("aug_mode", None)

        cache_dir = None if not use_caching else os.environ.get('LOCAL_TMPDIR', "/local/u12272/")

        loaders = {}
        for split in ["train", "val", "test"]:
            for stream in ["global", "local"]:
                path = self.data_paths.get(f"{split}_{stream}", None)
                if path is not None:

                    _, loaders[f"{split}_{stream}"] = make_dataloader(
                        dataset_type=dataset_type,
                        root_path=[path],
                        resolution=self.resolution,
                        frames_per_clip=self.frames_per_clip,
                        frame_step=frame_step,
                        eval_duration=clip_duration,
                        num_segments=num_segments,
                        num_views_per_segment=num_views_per_segment_train if split == "train" else num_views_per_segment_val,
                        allow_segment_overlap=True,
                        batch_size=self.batch_size if split == "train" else 1 * self.batch_size,
                        world_size=1, #self.world_size, # multi-GPU training disabled for head training
                        rank=0, #self.rank,
                        training=(split == "train"),
                        num_workers=num_workers,
                        repetitions_per_epoch=1,
                        video_base_path=base_path,
                        random_resize_aspect_ratio=ar_range,
                        random_resize_scale=rr_scale,
                        reprob=reprob,
                        auto_augment=use_aa,
                        motion_shift=motion_shift,
                        random_horizontal_flip=random_horizontal_flip,
                        crop_to_bboxes=crop_to_bboxes,
                        normalize=normalize,
                        eval_resize_mode=eval_resize_mode,
                        temporal_crop_frames=temporal_crop_frames,
                        crop_increase_factor=crop_increase_factor,
                        shuffle=(split == "train"),
                        aug_mode=aug_mode,
                        cache_dir=cache_dir,
                    )

                    print(
                        f"Created dataloader for {split} {stream}: {len(loaders[f'{split}_{stream}'].dataset)} samples"
                    )

        if "train_global" in loaders:
            ipe = len(loaders["train_local"])
            ipe_global = len(loaders["train_global"])
            assert (
                ipe == ipe_global
            ), f"Number of iterations per epoch for local and global streams must match, got {ipe} and {ipe_global}"

        self.loaders = loaders

    def infer_backbone(self, data, training):

        self.backbone.train(mode=training)

        buffer, label, clip_indices, info = data

        # buffer is list[list[Tensor]] with shapes matching [num_clips][num_views]
        num_clips = len(buffer)
        num_views_per_clip = len(buffer[0])
        clips_cpu = torch.cat([torch.cat(row, dim=0) for row in buffer], dim=0).pin_memory()
        clips = clips_cpu.to(self.device, non_blocking=True)

        self.backbone.train(mode=training)

        if training and not self.backbone_training_mode == "frozen":
            outputs = self.backbone(clips)
        else:
            with torch.no_grad():
                outputs = self.backbone(clips)
        # outputs: {layer_idx: tensor[(temporal*spatial*batch), tokens, dim]}
        
        latents = [
            rearrange(
                o.contiguous(),
                "(temporal spatial b) tokens dim -> b (temporal spatial) tokens dim",
                temporal=num_clips,
                spatial=num_views_per_clip,
            )
            for _, o in outputs.items()
        ]

        sample = {}

        sample["label"] = label.unsqueeze(1).to(self.device, non_blocking=True)
        sample["bbox_xyxy_rel"] = info["bbox_xyxy_rel"].unsqueeze(1)
        sample["present_crops"] = torch.ones(sample["label"].shape[:2], dtype=torch.bool, device=self.device)
        sample["index"] = info["index"].unsqueeze(1).to(self.device)

        rehaped_latents = [reshape_latents(self.backbone_type, l) for l in latents]
        cls_tokens = [r[0] for r in rehaped_latents if r[0] is not None]
        patch_tokens = [r[1].unsqueeze(1) for r in rehaped_latents if r[1] is not None]

        sample["patch_tokens"] = patch_tokens

        return sample
    
    def offload(self, split: str):
        pass

    def get_iter(self, split: str):

        if split not in ["train", "val", "test"]:
            raise ValueError(f"Invalid split: {split}")
        
        local_loader = self.loaders[f"{split}_local"]
        global_loader = self.loaders.get(f"{split}_global", itertools.repeat(None))

        return _VideoLatentsIter(
            backbone_latents=self,
            local_loader=local_loader,
            global_loader=global_loader,
            device=self.device
        )