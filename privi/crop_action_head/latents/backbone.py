
import itertools
import time
from einops import rearrange
import torch
from tqdm import tqdm
from privi.backbones.generic import get_backbone, reshape_latents
from privi.crop_action_head.latents.generic import BaseLatents, LatentsIter
from torch.utils.data import Dataset, DataLoader
import logging

import numpy as np
import secrets

import gc, ctypes, os

from privi.utils.misc import normalize_dtype, timer

logger = logging.getLogger(__name__)

def _malloc_trim():
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)  # Linux only; harmless if it fails elsewhere
    except Exception:
        pass

def _free_everything():
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()  # GPU; doesn't affect CPU RSS but good hygiene
    except Exception:
        pass
    _malloc_trim()


class CachedLatentsDataset(Dataset):
    """
    Holds precomputed samples in memory as big tensors and exposes the same format
    as latent_dataset/vjepa.py, including num_crops().
    Expects:
      - patch_tokens: list[Tensor] with shape [N, crops, views, c, t, h, w]
      - label: Tensor [N, ...]
      - bbox_xyxy_rel: Tensor [N, crops, 4]
      - present_crops: Bool Tensor [N, crops]
      - index: Tensor [N, crops]
    """
    def __init__(self, patch_tokens, label, bbox_xyxy_rel, present_crops, index, offload=False):
        self.patch_tokens = patch_tokens
        self.label = label
        self.bbox_xyxy_rel = bbox_xyxy_rel
        self.present_crops = present_crops
        self.index = index
        self._num_crops = int(self.present_crops.sum().item())

        self.do_offload = offload
        self.tmpdir = os.path.join(os.environ.get('LOCAL_TMPDIR', ""), secrets.token_hex(16))
        if self.do_offload:
            print(f"OFFLOAD DIR {self.tmpdir}")
        os.makedirs(self.tmpdir, exist_ok=False)
        self.n_patch_tokens = len(patch_tokens)
        self.data_on_disk = False

        memory_consumption = 0
        for pt in self.patch_tokens:
            memory_consumption += pt.element_size() * pt.nelement()
        logger.info(f"Memory consumption for dataset: {memory_consumption / 1024**2:.2f} MB, dtype {self.patch_tokens[0].dtype}")

    def offload(self):
        if not self.do_offload:
            return
        if not self.data_on_disk:
            print("OFFLOADING")
            with timer("offload patch_tokens"):
                try:
                    for i in range(self.n_patch_tokens):
                        np.save(os.path.join(self.tmpdir, f"patch_tokens.{i:03d}.npy"), self.patch_tokens[i].detach().cpu().contiguous().numpy(), allow_pickle=False)
                except OSError as e:
                    # Not enough disk space
                    print(f"Loading to local ssd yielded {e}. Offloading to $TMPDIR instead")
                    self.tmpdir = os.path.join(os.environ['TMPDIR'], secrets.token_hex(16))
                    os.makedirs(self.tmpdir, exist_ok=False)
                    print(f"NEW OFFLOAD DIR {self.tmpdir}")
                    for i in range(self.n_patch_tokens):
                        np.save(os.path.join(self.tmpdir, f"patch_tokens.{i:03d}.npy"), self.patch_tokens[i].detach().cpu().contiguous().numpy(), allow_pickle=False)
            self.data_on_disk = True
        self.patch_tokens = None
        _free_everything()

    def reload(self):
        with timer("Reload patch tokens"):
            self.patch_tokens = []
            for i in range(self.n_patch_tokens):
                arr = np.load(os.path.join(self.tmpdir, f"patch_tokens.{i:03d}.npy"), allow_pickle=False)
                self.patch_tokens.append(torch.from_numpy(arr))  # shares memory; cheap

    def __len__(self):
        return int(self.label.shape[0])

    def __getitem__(self, idx):
        if self.patch_tokens is None:
            self.reload()

        return {
            "patch_tokens": [pt[idx] for pt in self.patch_tokens],  # [crops, views, c, t, h, w]
            "label": self.label[idx],                                # [...]
            "bbox_xyxy_rel": self.bbox_xyxy_rel[idx],                # [crops, 4]
            "present_crops": self.present_crops[idx],                # [crops]
            "index": self.index[idx],                                # [crops]
        }

    def num_crops(self):
        return self._num_crops
    
class _BackboneLatentsIter(LatentsIter):

    def __init__(self, backbone_latents: "BackboneLatents", local_loader, global_loader, training):
        self.local_loader = local_loader
        self.global_loader = global_loader
        self.backbone_latents = backbone_latents
        self.training = training

    def __iter__(self):
        for l, g in zip(self.local_loader, self.global_loader):

            l = self.backbone_latents.infer_backbone(l, training=self.training)
            g = self.backbone_latents.infer_backbone(g, training=self.training) if g is not None else None
            yield {"local": l, "global": g}

    def __len__(self):
        return len(self.local_loader)

    def num_crops(self):
        return self.local_loader.dataset.num_crops()
    
def make_dataloader(
    root_path,
    batch_size,
    world_size,
    rank,
    dataset_type="chimpact",
    resolution=224,
    frames_per_clip=16,
    frame_step=4,
    num_segments=8,
    eval_duration=None,
    num_views_per_segment=1,
    allow_segment_overlap=True,
    training=False,
    num_workers=12,
    subset_file=None,
    repetitions_per_epoch=1,
    random_resize_aspect_ratio=(0.75, 4 / 3),
    random_resize_scale=(0.08, 1.0),
    reprob=0.25,
    auto_augment=True,
    motion_shift=False,
    random_horizontal_flip=False,
    normalize="default",
    eval_resize_mode="crop",
    shuffle=False,
    aug_mode=None,
    **kwargs,
):
    from privi.jepa.evals.video_classification_frozen.utils import make_transforms
    from privi.datasets.action.transforms import TransformNTimes

    # Make Video Transforms
    _num_views_per_segment = num_views_per_segment
    if training:
        # For training, we apply the same transform multiple times to get multiple views
        num_views_per_segment = 1
    transform = make_transforms(
        training=training,
        num_views_per_clip=num_views_per_segment,
        random_horizontal_flip=random_horizontal_flip,
        random_resize_aspect_ratio=random_resize_aspect_ratio,
        random_resize_scale=random_resize_scale,
        reprob=reprob,
        auto_augment=auto_augment,
        motion_shift=motion_shift,
        crop_size=resolution,
        normalize=normalize,
        eval_resize_mode=eval_resize_mode,
        aug_mode=aug_mode,
    )
    if training:
        # If training, apply the transform multiple times to get multiple views
        transform = TransformNTimes(transform, _num_views_per_segment)

    from privi.datasets.generic import make_videodataset
    dataset, data_loader, dist_sampler = make_videodataset(
        dataset_type=dataset_type,
        label_path=root_path[0],
        batch_size=batch_size,
        frames_per_clip=frames_per_clip,
        frame_step=frame_step,
        duration=eval_duration, # Also train duration, bad parameter name
        num_clips=num_segments,
        allow_clip_overlap=allow_segment_overlap,
        transform=transform,
        num_workers=num_workers,
        world_size=world_size,
        rank=rank,
        drop_last=training,
        repetitions_per_epoch=repetitions_per_epoch,
        shuffle=shuffle,
        **kwargs)

    return dataset, data_loader

class BackboneLatents(BaseLatents):

    """
    Extract latent representations from video clips using a backbone model.

    If backbone_training_mode is not "frozen", this will pass gradients on to self.backbone to allow for training it.

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
        self.setup_backbone(
            args_model=args_model,
            backbone_type=backbone_type,
            resolution=self.resolution,
            frames_per_clip=self.frames_per_clip,
            backbone_training_mode=backbone_training_mode)

        mem_cache = args_data.get("memory_caching", False)
        if  mem_cache == "offload" or bool(mem_cache):
            self.cache_eval_sets(offload=mem_cache == "offload")

    def setup_backbone(self, args_model, backbone_type, resolution, frames_per_clip, backbone_training_mode):

        args_model["resolution"] = resolution
        args_model["frames_per_clip"] = frames_per_clip
        args_model["training_mode"] = backbone_training_mode

        self.backbone = get_backbone(
            backbone_name=backbone_type,
            cfg=args_model,
            device=self.device,
        )

    def setup_data(self, args_data, args_data_aug):

        base_path = args_data.get("data_path")
        num_workers = args_data.get("num_workers", 12)
        num_views_per_segment_train = args_data.get("num_views_per_segment_train", 1)
        num_views_per_segment_val = args_data.get("num_views_per_segment_val", 1)
        num_segments = 1
        dataset_type = args_data.get("dataset_type", "chimpact")
        pred_bboxes_path = args_data.get("pred_bboxes_path", None)
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
        self.datasets = {}
        for split in ["train", "val", "test"]:
            for stream in ["global", "local"]:
                path = self.data_paths.get(f"{split}_{stream}", None)
                if path is not None:

                    dataset, loader = make_dataloader(
                        dataset_type=dataset_type,
                        root_path=[path],
                        pred_bboxes_path=pred_bboxes_path,
                        resolution=self.resolution,
                        frames_per_clip=self.frames_per_clip,
                        frame_step=frame_step,
                        eval_duration=clip_duration,
                        num_segments=num_segments,
                        num_views_per_segment=num_views_per_segment_train if split == "train" else num_views_per_segment_val,
                        allow_segment_overlap=True,
                        batch_size=self.batch_size if split == "train" else 1 * self.batch_size,
                        world_size=1, #self.world_size, # multi-GPU disabled for head training
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
                    loaders[f"{split}_{stream}"] = loader
                    if stream == "local":
                        self.datasets[split] = dataset

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

    def get_dataset(self, split: str):
        return self.datasets.get(split, None)

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
    
    @torch.inference_mode()
    def cache_eval_sets(self, offload=False):
        """
        Run val/test once through the (frozen) backbone, cache samples in RAM as big tensors,
        and replace the corresponding dataloaders with in-memory datasets. 
        """

        def _infer_to_cpu(batched):
            sample = self.infer_backbone(batched, training=False)
            patch_tokens = [t.detach().cpu().to(self.dtype_cache).contiguous() for t in sample["patch_tokens"]]
            label = sample["label"].detach().cpu().contiguous()
            bbox = sample["bbox_xyxy_rel"].detach().cpu().contiguous()
            present = sample["present_crops"].detach().cpu().contiguous()
            index = sample["index"].detach().cpu().contiguous()
            return patch_tokens, label, bbox, present, index

        def _cache_stream(split: str, stream: str):
            key = f"{split}_{stream}"
            if key not in self.loaders:
                return

            loader = self.loaders[key]
            total = len(loader.dataset)  # number of samples
            if total == 0:
                return

            pt_buf = None
            label_buf = bbox_buf = present_buf = index_buf = None
            write_ptr = 0

            logger.info(f"Started to load {split}-{stream}")

            start_time = time.perf_counter()
            for i, batch in enumerate(tqdm(loader)):
                pt, lab, bbox, present, index = _infer_to_cpu(batch)
                B = lab.shape[0]

                if pt_buf is None:
                    # Preallocate buffers using shapes from first batch
                    pt_buf = [torch.empty((total, *p.shape[1:]), dtype=p.dtype) for p in pt]
                    label_buf = torch.empty((total, *lab.shape[1:]), dtype=lab.dtype)
                    bbox_buf = torch.empty((total, *bbox.shape[1:]), dtype=bbox.dtype)
                    present_buf = torch.empty((total, *present.shape[1:]), dtype=present.dtype)
                    index_buf = torch.empty((total, *index.shape[1:]), dtype=index.dtype)

                sl = slice(write_ptr, write_ptr + B)
                for li in range(len(pt)):
                    pt_buf[li][sl] = pt[li]
                label_buf[sl] = lab
                bbox_buf[sl] = bbox
                present_buf[sl] = present
                index_buf[sl] = index
                write_ptr += B

                if self.limit_caching > 0 and i > self.limit_caching:
                    print("Only limited caching for debug")
                    break

            print(f"Loading {split}-{stream} took {time.perf_counter()-start_time:.3f} s")

            # Replace loader with in-memory dataset
            ds = CachedLatentsDataset(pt_buf, label_buf, bbox_buf, present_buf, index_buf, offload=offload)
            if offload:
                ds.offload()
            del self.loaders[key]
            self.loaders[key] = DataLoader(ds, batch_size=self.val_batch_size, shuffle=False, num_workers=0, pin_memory=True)
            logger.info(f"Cached {key}: {len(ds)} items, {ds.num_crops()} present crops")

        for split in ["val", "test"]:
            _cache_stream(split, "local")
            _cache_stream(split, "global")

        _free_everything()

        self.cached_eval = True

    def offload(self, split: str):
        if self.cached_eval:
            if (gl := self.loaders.get(f"{split}_local", None)) is not None:
                gl.dataset.offload()
            if (gl := self.loaders.get(f"{split}_global", None)) is not None:
                gl.dataset.offload()

    def get_iter(self, split: str):

        if split not in ["train", "val", "test"]:
            raise ValueError(f"Invalid split: {split}")
        
        local_loader = self.loaders[f"{split}_local"]
        global_loader = self.loaders.get(f"{split}_global", itertools.repeat(None))

        if self.cached_eval and split != "train":
            return LatentsIter(local_loader, global_loader, device=self.device)
        else:
            return _BackboneLatentsIter(
                backbone_latents=self,
                local_loader=local_loader,
                global_loader=global_loader,
                training=(split == "train")
            )