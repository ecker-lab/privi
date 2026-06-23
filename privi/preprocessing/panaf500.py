#!/usr/bin/env python3
"""
Convert the raw PanAf500 dataset into fixed-length, single-action, cropped
video clips for action recognition.

Pipeline (per split):
  1. Read the per-video PanAf annotations and reconstruct per-ape tracks.
  2. Cut each track at every behaviour change so every segment has a single
     action label (``split_tracks_by_action``).
  3. Slice each single-action segment into fixed-length clips with a fixed
     stride (``split_track``).
  4. For each clip, compute an enclosing bounding box (with a 25% margin)
     around the ape across the clip and write a spatially-cropped video
     (``process_video_segment``).
  5. Write a ``<name>_<split>.csv`` mapping clip filename -> action label.

Label convention
-----------------
PanAf annotates 10 behaviours with ``no_action`` at index 0 (see
``PANAF_IDX2ACTIONS``). The published action-recognition dataset has 9 classes:
``no_action`` clips are dropped and the remaining labels are shifted down by one
so that ``walking`` becomes 0, ..., ``sitting_on_back`` becomes 8. This matches
``privi/datasets/eval/panaf500.py`` (``num_classes = 9``). Both steps are applied
when the CSV is written (see ``convert_split``).

Expected raw dataset layout (``--dataset_path``)::

    panaf500/
      annotations/{train,validation,test}/<video_id>.json
      videos/<video_id>.mp4

Output layout (``--output_dir``)::

    <output_dir>/
      train.csv               # columns: video_filename label  (space separated, no header)
      val.csv
      test.csv
      videos/
        train/<video_id>_<ape_id>_<start_frame>.mp4
        val/<video_id>_<ape_id>_<start_frame>.mp4
        test/<video_id>_<ape_id>_<start_frame>.mp4

Example::

    python -m privi.preprocessing.panaf500 \
        --dataset_path /data/public/panaf/panaf500 \
        --output_dir   /data/output/panaf500_ar
"""

import os
import json

import torch
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

import cv2
import torchvision.transforms.functional as F
from decord import VideoReader
import fire


# PanAf behaviour taxonomy. Index 0 (``no_action``) is dropped for the published
# 9-class action-recognition dataset; see the module docstring.
PANAF_IDX2ACTIONS = [
    'no_action',
    'walking',
    'standing',
    'sitting',
    'climbing_up',
    'hanging',
    'climbing_down',
    'running',
    'camera_interaction',
    'sitting_on_back',
]

PANAF_ACTIONS2IDX = {action: idx for idx, action in enumerate(PANAF_IDX2ACTIONS)}

# Frame rate of the written clip videos.
OUTPUT_FPS = 30

# Map raw split names to the names used in the output layout.
OUTPUT_SPLIT_NAMES = {'train': 'train', 'validation': 'val', 'test': 'test'}


class panaf500:
    """Read the raw PanAf500 annotations and reconstruct per-ape tracks.

    Only the methods needed for dataset conversion are implemented (track
    extraction and video-path lookup).
    """

    def __init__(self, path) -> None:
        self.base_path = Path(path)
        self._splits = ['train', 'validation', 'test']

        self.split_ids = {}
        for split_name in self._splits:
            files = os.listdir(os.path.join(self.base_path, 'annotations', split_name))
            ids = [os.path.splitext(file)[0] for file in files if file.endswith('.json')]
            ids.sort()
            self.split_ids[split_name] = ids

    def list_split(self, split_name: str):
        """Return the sorted list of video ids belonging to ``split_name``."""
        return self.split_ids[split_name]

    def which_split(self, video_id: str):
        """Return the name of the split that contains ``video_id``."""
        for split in self._splits:
            if video_id in self.list_split(split):
                return split

    def get_video_path(self, video_id: str):
        """Return the absolute path of the source ``.mp4`` for ``video_id``."""
        return os.path.join(self.base_path, 'videos', f'{video_id}.mp4')

    def get_tracks_for_split(self, split_name, min_length_frames=0, gap_threshold=10):
        """Return tracks for every video in a split, skipping videos with none.

        Args:
            split_name: split to process (``train``/``validation``/``test``).
            min_length_frames: drop tracks shorter than this many frames.
            gap_threshold: maximum frame gap tolerated within a single track
                (larger gaps start a new track).

        Returns:
            tuple ``(tracks, ids)`` where ``tracks`` is a list of per-video track
            lists and ``ids`` is the corresponding list of video ids.
        """
        tracks = []
        ids = []
        for video_id in self.list_split(split_name):
            this_tracks = self.get_tracks_for_video(video_id, min_length_frames, gap_threshold)

            if not this_tracks:
                # skip videos with no tracks
                continue

            ids.append(video_id)
            tracks.append(this_tracks)

        return tracks, ids

    def get_tracks_for_video(self, video_id, min_length_frames=0, gap_threshold=10):
        """Build the list of per-ape tracks for a single video.

        The ``start_frame`` is 0-based, not 1-based as in the PanAf annotations.

        Args:
            video_id: the video id.
            min_length_frames: drop tracks shorter than this many frames.
            gap_threshold: the maximum gap (in frames) between two detections of
                the same ape that is still considered the same track.

        Returns:
            list of dict, one per track, with keys:
                'ape_id' (int): the ape id.
                'start_frame' (int): 0-based frame where the track starts.
                'bboxes' (torch.Tensor): per-frame bounding boxes. Gaps inside a
                    track are filled by repeating the last seen bbox/label until
                    the ape is detected again.
                'labels' (torch.Tensor): per-frame action labels (PanAf indices).
        """
        with open(os.path.join(self.base_path, 'annotations', self.which_split(video_id), f'{video_id}.json'), 'r') as f:
            data = json.load(f)

        finished_tracks = []
        tracks = {}  # data structure: {ape_id: [(frame_id, bbox, behaviour) for each frame in track]}
        for frame in data['annotations']:
            frame_id = int(frame['frame_id'])
            for det in frame['detections']:
                if det['ape_id'] not in tracks:
                    tracks[det['ape_id']] = []
                elif tracks[det["ape_id"]][-1][0] < frame_id - gap_threshold:
                    finished_tracks.append((det["ape_id"], tracks[det['ape_id']]))
                    tracks[det['ape_id']] = []
                else:
                    for id_ in range(tracks[det["ape_id"]][-1][0] + 1, frame_id):
                        # add dummy entries for missing frames
                        tracks[det["ape_id"]].append((id_, *tracks[det["ape_id"]][-1][1:]))
                tracks[det['ape_id']].append((int(frame['frame_id']), det['bbox'], det['behaviour']))
        for ape_id, track in tracks.items():
            finished_tracks.append((ape_id, track))

        out_tracks = []
        for ape_id, track in finished_tracks:
            if len(track) < min_length_frames:
                continue

            bboxes = torch.empty((len(track), 4))
            labels = torch.empty((len(track),), dtype=torch.long)

            for idx, (frame_id, bbox, behaviour) in enumerate(track):
                bboxes[idx] = torch.tensor(bbox)
                labels[idx] = PANAF_ACTIONS2IDX[behaviour]

            out_tracks.append({
                'ape_id': ape_id,
                'start_frame': track[0][0] - 1,  # PanAf counts from 1, we use 0-based indexing
                'bboxes': bboxes,
                'labels': labels,
            })
        return out_tracks


def split_tracks_by_action(ids, tracks):
    """Cut tracks at every behaviour change into single-action segments.

    Args:
        ids: list of video ids (parallel to ``tracks``).
        tracks: list of per-video track lists, as returned by
            ``panaf500.get_tracks_for_split``.

    Returns:
        list of dict, one per single-action segment, with keys ``video_id``,
        ``ape_id``, ``start_frame``, ``label`` (PanAf index), ``bboxes`` and
        ``no_frames``.
    """
    flat_tracks = []

    for video_id, this_tracks in zip(ids, tracks):
        for track in this_tracks:
            previous_cut_idx = 0
            for idx in range(len(labels := track["labels"])):
                if idx + 1 >= len(labels) or labels[idx] != labels[idx + 1]:
                    # label change, set a cut here
                    label = labels[idx]
                    assert (labels[previous_cut_idx:idx + 1] == label).all()
                    flat_tracks.append({
                        "video_id": video_id,
                        "ape_id": track["ape_id"],
                        "start_frame": track["start_frame"] + previous_cut_idx,
                        "label": label.item(),
                        "bboxes": track["bboxes"][previous_cut_idx:idx + 1],
                        "no_frames": idx + 1 - previous_cut_idx,
                    })
                    previous_cut_idx = idx + 1

    return flat_tracks


def split_track(track, split_len=64, stride=16):
    """Slice a single-action segment into fixed-length, overlapping clips.

    Args:
        track: a single-action segment from ``split_tracks_by_action``.
        split_len: clip length in frames.
        stride: step between consecutive clip start frames.

    Returns:
        list of clip dicts (same schema as the input, with ``no_frames`` equal
        to ``split_len``).
    """
    result = []
    for i in range(0, track["no_frames"] - split_len + 1, stride):
        result.append({
            "video_id": track["video_id"],
            "ape_id": track["ape_id"],
            "start_frame": track["start_frame"] + i,
            "label": track["label"],
            "bboxes": track["bboxes"][i:i + split_len],
            "no_frames": split_len,
        })
    return result


def get_enclosing_bbox(bboxes):
    """Return the box enclosing all ``bboxes``, enlarged by 25% on each side.

    Args:
        bboxes: ``(n, 4)`` tensor of ``[x_min, y_min, x_max, y_max]`` boxes.

    Returns:
        list ``[x_min, y_min, x_max, y_max]`` of the padded enclosing box.
    """
    bboxes_np = bboxes.numpy()

    x_min = bboxes_np[:, 0].min()
    y_min = bboxes_np[:, 1].min()
    x_max = bboxes_np[:, 2].max()
    y_max = bboxes_np[:, 3].max()

    # Increase size by 25% in each direction
    width = x_max - x_min
    height = y_max - y_min
    x_min -= width * 0.25
    y_min -= height * 0.25
    x_max += width * 0.25
    y_max += height * 0.25

    return [x_min, y_min, x_max, y_max]


def process_video_segment(dataset, row, output_dir):
    """Write a spatially-cropped clip video for a single clip row.

    The clip is cropped to ``row.enclosing_bbox`` (clipped to image bounds) and
    written to ``<output_dir>/<row.video_filename>`` at ``OUTPUT_FPS``.

    Args:
        dataset: a ``panaf500`` instance (used to resolve the source video path).
        row: a row from the per-split clip dataframe (see ``convert_split``).
        output_dir: base output directory for this split.
    """
    video_path = dataset.get_video_path(row.video_id)
    out_path_video = Path(output_dir) / row.video_filename

    vr = VideoReader(video_path)
    frames = vr.get_batch(range(row.start_frame, row.start_frame + row.no_frames)).asnumpy()

    # Bbox clipped to image boundaries
    height, width = frames[0].shape[:2]
    x1, y1, x2, y2 = map(int, row.enclosing_bbox)
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(width, x2)
    y2 = min(height, y2)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(str(out_path_video), fourcc, OUTPUT_FPS, (x2 - x1, y2 - y1))

    for frame in frames:
        frame_tensor = torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0
        cropped_frame = F.crop(frame_tensor, y1, x1, y2 - y1, x2 - x1)
        cropped_frame = (cropped_frame.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        cropped_frame_bgr = cv2.cvtColor(cropped_frame, cv2.COLOR_RGB2BGR)
        video_writer.write(cropped_frame_bgr)

    video_writer.release()


def convert_split(dataset, split, output_dir, clip_len=16, stride=16, gap_threshold=1):
    """Convert one split into cropped clip videos plus a label CSV.

    Args:
        dataset: a ``panaf500`` instance.
        split: split name (``train``/``validation``/``test``).
        output_dir: base output directory. The label CSV is written to
            ``<output_dir>/<out_split>.csv`` and clips to
            ``<output_dir>/videos/<out_split>/`` (created by the caller), where
            ``out_split`` is the output name from ``OUTPUT_SPLIT_NAMES``.
        clip_len: clip length in frames.
        stride: step between consecutive clip start frames.
        gap_threshold: maximum frame gap tolerated within a track.
    """
    out_split = OUTPUT_SPLIT_NAMES[split]
    tracks, ids = dataset.get_tracks_for_split(split, min_length_frames=clip_len, gap_threshold=gap_threshold)
    flat_tracks = split_tracks_by_action(ids, tracks)
    print(f"[{split}] {len(flat_tracks)} single-action segments")

    split_tracks = []
    for track in flat_tracks:
        if track["no_frames"] > clip_len:
            split_tracks.extend(split_track(track, clip_len, stride))
    print(f"[{split}] {len(split_tracks)} clips of length {clip_len}")

    df = pd.DataFrame(split_tracks)

    # Drop no_action clips and shift labels down by one -> published 9-class
    # taxonomy (walking=0, ..., sitting_on_back=8). See module docstring.
    df = df[df["label"] != PANAF_ACTIONS2IDX["no_action"]].reset_index(drop=True)
    df["label"] = df["label"] - 1

    df["enclosing_bbox"] = df["bboxes"].map(get_enclosing_bbox)
    df["video_filename"] = df.apply(
        lambda r: f"videos/{out_split}/{r.video_id}_{r.ape_id}_{r.start_frame}.mp4", axis=1
    )

    df["label_str"] = df["label"].map(lambda x: PANAF_IDX2ACTIONS[x + 1])
    print(f"[{split}] label distribution:\n{df['label_str'].value_counts().to_string()}")

    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"writing {split} clips"):
        process_video_segment(dataset, row, output_dir)

    csv_path = Path(output_dir) / f"{out_split}.csv"
    df.to_csv(csv_path, index=False, header=False, sep=" ", columns=["video_filename", "label"])
    print(f"[{split}] wrote {csv_path}")


def main(
    output_dir,
    dataset_path,
    splits=("train", "validation", "test"),
    clip_len=16,
    stride=None,
    gap_threshold=1,
):
    """Convert the raw PanAf500 dataset into cropped action-recognition clips.

    Args:
        output_dir: base output directory. Writes ``<split>.csv`` and
            ``videos/<split>/*.mp4`` for each split (``validation`` is named
            ``val`` in the output; see ``OUTPUT_SPLIT_NAMES``).
        dataset_path: path to the raw PanAf500 dataset. Must contain ``annotations/<split>/*.json``
            and ``videos/*.mp4``.
        splits: which splits to convert.
        clip_len: clip length in frames.
        stride: step between consecutive clip start frames (defaults to
            ``clip_len``, i.e. non-overlapping clips).
        gap_threshold: maximum frame gap tolerated within a track.
    """
    if stride is None:
        stride = clip_len
    if isinstance(splits, str):
        splits = (splits,)

    dataset = panaf500(path=dataset_path)
    output_dir = Path(output_dir)

    for split in splits:
        (output_dir / "videos" / OUTPUT_SPLIT_NAMES[split]).mkdir(parents=True, exist_ok=True)
        convert_split(dataset, split, output_dir, clip_len=clip_len, stride=stride, gap_threshold=gap_threshold)


if __name__ == '__main__':
    fire.Fire(main)
