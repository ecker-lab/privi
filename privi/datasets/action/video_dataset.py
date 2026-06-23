# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
# Modified by Felix Benjamin Mueller, 2025

from collections import defaultdict
import json
import os
import pathlib

from logging import getLogger
import random

import numpy as np
import pandas as pd

import torch

from privi.datasets.action.decoding import loadvideo_decord, resolve_video


_GLOBAL_SEED = 0
logger = getLogger()

def worker_init_fn(_):
    import os, faulthandler
    faulthandler.enable()
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    try:
        import cv2; cv2.setNumThreads(0)
    except Exception:
        pass

def load_ava_dataset(data_path, temporal_crop_frames, multi_label=True):

    if isinstance(data_path, str):
        data_path = pathlib.Path(data_path)

    # Load AVA dataset
    names=["filename", "frame", "x_min", "y_min", "x_max", "y_max", "label", "entity_id", "confidence"]
    data = pd.read_csv(
        data_path,
        header=None,
        delimiter=",",)
    
    if data.shape[1] == 8:
        data.columns = names[:-1]
        data["confidence"] = 1.0
    else:
        data.columns = names

    data = data.astype({
            "filename": str,
            "frame": int,
            "x_min": float,
            "y_min": float,
            "x_max": float,
            "y_max": float,
            "label": int,
            "entity_id": int,
            "confidence": float,
        }
    )

    with open(data_path.parent / "action_list.json", "r") as f:
        action_list = json.load(f)

    print(f"Loaded {len(data)} rows from {data_path}")

    drop_keys = ["filename", "frame", "entity_id", "x_min", "y_min", "x_max", "y_max"]
    group_keys = drop_keys + ["confidence"]

    annot_joint = data[group_keys].drop_duplicates(subset=drop_keys)
    annot_joint = annot_joint.set_index(["filename", "frame", "entity_id"])

    duplicates_in_frames = annot_joint[annot_joint.index.duplicated(keep=False)]
    if len(duplicates_in_frames) > 0:
        print("Warning: duplicates in gt_short", duplicates_in_frames.index.drop_duplicates())

    annot_joint = annot_joint[~annot_joint.index.duplicated(keep="last")]
    annot_joint = annot_joint.reset_index()

    print(
        f"Got {len(annot_joint)} samples from {data_path} after removing duplicates and merging multi-labels"
    )

    samples = annot_joint["filename"].to_list()

    action_label2idx = {action["id"]: i for i, action in enumerate(action_list)}
    action_label2idx[-1] = np.nan  # For no action label

    annot_joint_ = annot_joint.reset_index().rename(columns={"index": "orig_idx"})[
        ["orig_idx", "filename", "frame", "entity_id"]
    ]

    # 2. Perform a single merge on the join keys
    merged = annot_joint_.merge(
        data[["filename", "frame", "entity_id", "label"]],
        on=["filename", "frame", "entity_id"],
        how="left",
    )

    # 3. Drop rows with no label match and map labels to indices
    merged = merged.dropna(subset=["label"])
    merged["action_idx"] = merged["label"].map(action_label2idx)
    merged = merged.dropna(subset=["action_idx"])
    merged["action_idx"] = merged["action_idx"].astype(int)

    # 4. Prepare the label matrix
    N = len(annot_joint)
    K = len(action_label2idx) - 1 # Exclude the no-action label
    labels = np.zeros((N, K), dtype=np.float32)

    

    # 5. Vectorized assignment
    rows = merged["orig_idx"].to_numpy(dtype=int)
    cols = merged["action_idx"].to_numpy(dtype=int)
    labels[rows, cols] = 1

    print("Num of samples with no labels:", np.sum(labels.sum(axis=1) == 0))
    print("Num of samples with at least one label", np.sum(labels.sum(axis=1) >= 1))

    if not multi_label:
        assert int(labels.sum()) == N, f"Expected one label per sample {N}, but got {labels.sum()} labels"
        labels = np.argmax(labels, axis=1)
        assert len(labels) == N

    bboxes_list = [
        [row[["x_min", "y_min", "x_max", "y_max"]].to_list()] for _, row in annot_joint.iterrows()
    ]

    time_intervals = [
        (row["frame"] - temporal_crop_frames // 2, row["frame"] + temporal_crop_frames // 2)
        for _, row in annot_joint.iterrows()
    ]

    #bbox_format = "xyxy_relative" if annot_joint[["x_min", "y_min", "x_max", "y_max"]].max().max() <= 2.0 else "tlwh_absolute"
    #print("BBOX FORMAT:", bbox_format)

    ret = dict(
        samples=samples,
        labels=labels,
        bboxes_list=bboxes_list,
        time_intervals=time_intervals,
        bbox_format="xyxy_relative", #bbox_format,
        multi_label=True,
        label_names=[action["name"] for action in action_list],
        data=data,
        annot_joint=annot_joint,
    )

    return ret


def load_csv_dataset(data_path):
    """ "
    Each data_path's entry must be an absolute path to a csv file. We interpret the all video paths
    in the csv file as relative to data_path's parent directory. If bboxes are available, they
    should be stored in a file with the same stem as the csv file but with a '_bboxes.npy' suffix.
    The bboxes file should contain a numpy array with the following fields:
        - bboxes: np.array of shape (N, 4) where N is the number of bboxes. Each bbox is represented as
            [x1, y1, x2, y2] where (x1, y1) is the top-left corner and (x2, y2) is the bottom-right
            corner of the bbox.
        - sample_idx2bbox_idx: np.array of shape (M,) where M is the number of samples in the csv file.
            This array maps each sample to the corresponding bbox indices in the bboxes array. The
            i-th element of this array is the index in the bboxes array corresponding to the i-th
            sample in the csv file.
    """

    ret = dict()

    data = pd.read_csv(data_path, header=None, delimiter=" ")

    ret["samples"] = list(data.values[:, 0])

    if len(data.values[0]) > 2:
        ret["multi_label"] = True
        ret["labels"] = data.values[:, 1:].astype(np.float32)
    else:
        ret["labels"] = list(data.values[:, 1])

    return ret


def crop_to_bbox(buffer, sample_bboxes, bbox_format, crop_increase_factor):

    if len(sample_bboxes) > 0:
        bbox_idx = np.random.randint(len(sample_bboxes))
        sample_bbox = sample_bboxes[bbox_idx]
    else:
        sample_bbox = None

    img_height, img_width = buffer.shape[1:3]

    if sample_bbox is not None:
        if bbox_format == "xyxy_relative":
            width = sample_bbox[2] - sample_bbox[0]  # relative width
            height = sample_bbox[3] - sample_bbox[1]  # relative height
            width_increase = width * crop_increase_factor / 2
            height_increase = height * crop_increase_factor / 2

            # Compute expanded bbox coordinates while keeping them between 0 and 1
            rel_x1 = max(0.0, sample_bbox[0] - width_increase)
            rel_y1 = max(0.0, sample_bbox[1] - height_increase)
            rel_x2 = min(1.0, sample_bbox[2] + width_increase)
            rel_y2 = min(1.0, sample_bbox[3] + height_increase)

            # Convert relative coordinates to absolute pixel values

            abs_x1 = int(rel_x1 * img_width)
            abs_y1 = int(rel_y1 * img_height)
            abs_x2 = int(rel_x2 * img_width)
            abs_y2 = int(rel_y2 * img_height)

            rel_bbox = np.array(sample_bbox)
        elif bbox_format == "tlwh_absolute":
            width_increase = sample_bbox[2] * crop_increase_factor / 2
            height_increase = sample_bbox[3] * crop_increase_factor / 2

            rel_bbox = np.array(
                [
                    sample_bbox[0] / img_width,  # x1 relative
                    sample_bbox[1] / img_height,  # y1 relative
                    (sample_bbox[0] + sample_bbox[2]) / img_width,  # x2 relative
                    (sample_bbox[1] + sample_bbox[3]) / img_height,  # y2 relative
                ]
            )

            # Compute expanded bbox coordinates in absolute coordinates
            abs_x1 = int(max(0, sample_bbox[0] - width_increase))
            abs_y1 = int(max(0, sample_bbox[1] - height_increase))
            abs_x2 = int(min(buffer.shape[2], sample_bbox[0] + sample_bbox[2] + width_increase))
            abs_y2 = int(min(buffer.shape[1], sample_bbox[1] + sample_bbox[3] + height_increase))
        else:
            raise ValueError(f"Unsupported bbox format: {bbox_format}")

        return buffer[:, abs_y1:abs_y2, abs_x1:abs_x2, :], rel_bbox

    return buffer, None

class VideoDataset(torch.utils.data.Dataset):
    """Video classification dataset."""

    label_names: list[str] = None  # Optional, used for evaluation

    def __init__(
        self,
        label_path,
        datasets_weights=None,
        frames_per_clip=16,
        frame_step=4,
        num_clips=1,
        transform=None,
        shared_transform=None,
        random_clip_sampling=True,
        allow_clip_overlap=False,
        filter_short_videos=False,
        filter_long_videos=int(10**9),
        duration=None,  # duration in seconds
        video_base_path=None,
        crop_increase_factor=0.25,
        temporal_crop_frames=None,
        cache_dir=None,
        crop_to_bboxes=True,
        prob_crop_to_bboxes=1.0,
        dataset_type=None,
    ):

        # For Explanation of most of this params, see loadvideo_decord
        self.datasets_weights = datasets_weights
        self.frames_per_clip = frames_per_clip
        self.frame_step = max(frame_step, 1)
        self.num_clips = num_clips
        self.transform = transform
        self.shared_transform = shared_transform
        self.random_clip_sampling = random_clip_sampling
        self.allow_clip_overlap = allow_clip_overlap
        self.filter_short_videos = filter_short_videos
        self.filter_long_videos = filter_long_videos
        self.duration = duration

        if temporal_crop_frames is None:
            temporal_crop_frames = self.frames_per_clip * self.frame_step
        else:
            print(f"Using temporal crop frames: {temporal_crop_frames}")

        print("allowing clip overlap:", allow_clip_overlap)

        self.center_frame_only = frame_step == 0
        if self.center_frame_only:
            print(f"Using center frame only, {self.frames_per_clip=}, {self.frame_step=}")

        self.crop_increase_factor = crop_increase_factor

        assert video_base_path is not None, "video_base_path must be set"

        self.cache_dir = pathlib.Path(cache_dir) if cache_dir is not None else None
        if self.cache_dir is not None:
            os.makedirs(self.cache_dir, exist_ok=True)

        self.base_dir = pathlib.Path(video_base_path)

        dataPath = pathlib.Path(label_path)

        if dataset_type == "ava":
            ret = load_ava_dataset(dataPath, temporal_crop_frames=temporal_crop_frames)

        elif dataset_type == "ava_sl":
            ret = load_ava_dataset(dataPath, temporal_crop_frames=temporal_crop_frames, multi_label=False)

        elif dataset_type == "csv":
            ret = load_csv_dataset(dataPath)

        else:
            raise ValueError(f"Unsupported file format: {dataset_type}")

        self.label_parser_ret = ret
        self.samples = ret["samples"]  # list of relative video paths
        self.labels = ret[
            "labels"
        ]  # # list of labels or a np.array of shape (N, num_classes) where N is the number of samples
        self.bboxes_list = ret.get(
            "bboxes_list", []
        )  # list of bboxes for each sample (list[list[list[float]]]) or empty
        self.time_intervals = ret.get(
            "time_intervals", []
        )  # list of time intervals for each sample (list[tuple[int, int]]) or empty
        self.bbox_format = ret.get("bbox_format", None)  # 'xyxy_relative' or 'tlwh_absolute'
        self.num_samples_per_dataset = [len(self.samples)]

        if not crop_to_bboxes:
            print("Forgetting bounding boxes, as bbox cropping is disabled")
            self.bboxes_list = []

        self.prob_crop_to_bboxes = prob_crop_to_bboxes

        # [Optional] Weights for each sample to be used by downstream
        # weighted video sampler
        self.sample_weights = None
        if self.datasets_weights is not None:
            self.sample_weights = []
            for dw, ns in zip(self.datasets_weights, self.num_samples_per_dataset):
                self.sample_weights += [dw / ns] * ns

        if self.bboxes_list:
            assert len(self.bboxes_list) == len(
                self.samples
            ), f"{len(self.bboxes_list)=} {len(self.samples)=}"

    def cache_sample(self, index):
        sample = self.samples[index]
        cached_file = resolve_video(sample, self.cache_dir, self.base_dir)  # caches the video
        # Return the size of the cached file
        return os.path.getsize(cached_file)

    def __getitem__(self, index):
        sample = self.samples[index]

        # Keep trying to load videos until you find a valid sample
        loaded_video = False
        while not loaded_video:
            try:
                if self.time_intervals:
                    interval = self.time_intervals[index]
                else:
                    interval = None
                buffer, clip_indices = loadvideo_decord(
                    resolve_video(sample, self.cache_dir, self.base_dir),
                    interval,
                    filter_long_videos=self.filter_long_videos,
                    filter_short_videos=self.filter_short_videos,
                    frames_per_clip=self.frames_per_clip,
                    frame_step=self.frame_step,
                    duration=self.duration,
                    num_clips=self.num_clips,
                    random_clip_sampling=self.random_clip_sampling,
                    allow_clip_overlap=self.allow_clip_overlap,
                )  # [T H W 3]
            except Exception as e:
                logger.warning(f"Error loading video {sample=}: {e.__class__} {e}")
                buffer = []
                clip_indices = []
            loaded_video = len(buffer) > 0
            if not loaded_video:
                logger.warning(f"Error loading video {sample=}, trying again")
                index = np.random.randint(self.__len__())
                sample = self.samples[index]

        # Label/annotations for video
        label = self.labels[index]

        def split_into_clips(video):
            """Split video into a list of clips"""
            fpc = self.frames_per_clip
            nc = self.num_clips
            return [video[i * fpc : (i + 1) * fpc] for i in range(nc)]

        if self.bboxes_list and np.random.rand(1).item() < self.prob_crop_to_bboxes:
            try:
                sample_bboxes = self.bboxes_list[index]
            except Exception as e:
                logger.warning(
                    f"Error loading bboxes for video {sample=}: {e.__class__} {e}, {index=}"
                )
                raise e

            buffer, xyxy_rel_bbox = crop_to_bbox(
                buffer, sample_bboxes, self.bbox_format, self.crop_increase_factor
            )
        else:
            xyxy_rel_bbox = np.array([0.0, 0.0, 1.0, 1.0])

        # Parse video into frames & apply data augmentations
        if self.shared_transform is not None:
            buffer = self.shared_transform(buffer)
        buffer = split_into_clips(buffer)

        try:
            if self.transform is not None:
                buffer = [self.transform(clip) for clip in buffer]
        except ZeroDivisionError as e:
            logger.warning(f"Error transforming video {index=} {sample=}, {xyxy_rel_bbox=}, {buffer=}: {e.__class__} {e}, {index=}")
            raise e

        if self.center_frame_only:
            # If center frame only, then take the center frame of each clip
            buffer = [
                np.repeat(clip[clip.shape[0] // 2 : clip.shape[0] // 2 + 1], clip.shape[0])
                for clip in buffer
            ]


        return (
            buffer,
            label,
            clip_indices,
            {"index": int(index), "bbox_xyxy_rel": xyxy_rel_bbox, "filename": sample},
        )

    def __len__(self):
        return len(self.samples)

    def num_crops(self):
        return len(self)
