import logging
import abc

from privi.datasets.action.video_dataset import VideoDataset, worker_init_fn
import torch

logger = logging.getLogger(__name__)

class Evaluator(abc.ABC):

    @abc.abstractmethod
    def add_batch(self, labels, logits):
        ...

    @abc.abstractmethod
    def metrics(self) -> dict[str, float]:
        ...

    @abc.abstractmethod
    def wandb_log_metrics(self, 
            results: list[str, float], 
            split: str, 
            log_epoch, classifier_id):
        ...
    

def make_videodataset(
    dataset_type,
    label_path,
    batch_size,
    frames_per_clip=8,
    frame_step=4,
    num_clips=1,
    random_clip_sampling=True,
    allow_clip_overlap=False,
    filter_short_videos=False,
    filter_long_videos=int(10**9),
    transform=None,
    shared_transform=None,
    rank=0,
    world_size=1,
    datasets_weights=None,
    collator=None,
    drop_last=True,
    num_workers=10,
    pin_mem=True,
    duration=None,
    log_dir=None,
    repetitions_per_epoch=1,
    shuffle=True,
    **kwargs,
):
    from privi.jepa.src.datasets.utils.weighted_sampler import DistributedWeightedSampler

    if dataset_type == "chimpact":
        from privi.datasets.eval.chimpact import ChimpACT
        dataset_cls = ChimpACT
    elif dataset_type == "pretrain":
        from privi.datasets.pretrain import PretrainDataset
        dataset_cls = PretrainDataset
    elif dataset_type == "chimpbehave":
        from privi.datasets.eval.chimpbehave import ChimpBehave
        dataset_cls = ChimpBehave
    elif dataset_type == "baboonland":
        from privi.datasets.eval.baboonland import BaboonLand
        dataset_cls = BaboonLand
    elif dataset_type == "panaf500":
        from privi.datasets.eval.panaf500 import PanAf500
        dataset_cls = PanAf500

    dataset = dataset_cls(
        label_path=label_path,
        datasets_weights=datasets_weights,
        frames_per_clip=frames_per_clip,
        frame_step=frame_step,
        num_clips=num_clips,
        random_clip_sampling=random_clip_sampling,
        allow_clip_overlap=allow_clip_overlap,
        filter_short_videos=filter_short_videos,
        filter_long_videos=filter_long_videos,
        duration=duration,
        shared_transform=shared_transform,
        transform=transform,
        **kwargs,
    )

    if False:
        dataset_ = torch.utils.data.Subset(dataset, range(0, 100))
        print(f"Using subset of {len(dataset_)} samples")
        dataset_.multi_label = dataset.multi_label
        dataset = dataset_

    logger.info("VideoDataset dataset created")
    if repetitions_per_epoch > 1:
        dataset = torch.utils.data.ConcatDataset([dataset] * repetitions_per_epoch)

    if datasets_weights is not None:
        dist_sampler = DistributedWeightedSampler(
            dataset.sample_weights, num_replicas=world_size, rank=rank, shuffle=shuffle
        )
    else:
        dist_sampler = torch.utils.data.distributed.DistributedSampler(
            dataset, num_replicas=world_size, rank=rank, shuffle=shuffle
        )

    data_loader = torch.utils.data.DataLoader(
        dataset,
        collate_fn=collator,
        sampler=dist_sampler,
        batch_size=batch_size,
        drop_last=drop_last,
        pin_memory=pin_mem,
        num_workers=num_workers,
        persistent_workers=num_workers > 0,
        multiprocessing_context="spawn" if num_workers > 0 else None,
        worker_init_fn=worker_init_fn if num_workers > 0 else None,
    )
    logger.info("VideoDataset unsupervised data loader created")

    return dataset, data_loader, dist_sampler