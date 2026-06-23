from pathlib import Path
import shutil
import cv2
from filelock import FileLock
import numpy as np
import logging
import os
from decord import VideoReader, cpu

logger = logging.getLogger(__name__)

def load_video_cv2(fname, all_indices):
    cap = cv2.VideoCapture(fname)

    buffer = []

    for i in all_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if not ret:
            raise ValueError(f"Error reading frame {i} from {fname=}")
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        buffer.append(frame)
    buffer = np.stack(buffer, axis=0)

    return buffer


def resolve_video(sample: str, cache_dir: Path, base_dir: Path) -> str:
    """ "
    Given a relative video sample path, resolve it to an absolute path. If a cache directory is provided,
    the video will be copied to the cache directory if it does not already exist there and the path to
    the cached video will be returned. If the cache directory is None, the original video path will be
    returned.

    Args:
        sample (str): Relative path to the video sample.
        cache_dir (Path or None): Directory to cache the video. If None, no caching is performed.
        base_dir (Path): Base directory where the video sample is located.

    Returns:
        str: Absolute path to the video sample, either from the cache directory or the base directory.
    """

    if cache_dir is not None:
        # print(f'Caching video {sample=}')
        fname = cache_dir / sample
        if not os.path.exists(fname):
            try:
                os.makedirs(os.path.dirname(fname), exist_ok=True)
                with FileLock(str(fname) + ".lock"):
                    if not os.path.exists(fname):
                        tmp_path = str(fname) + ".tmp"
                        shutil.copy2(base_dir / sample, tmp_path)
                        os.rename(tmp_path, fname)
            except Exception as e:
                #logger.warning(f"Error caching video {sample=}, {e.__class__} {e}")
                fname = str(base_dir / sample)

    else:
        fname = str(base_dir / sample)

    return str(fname)


def loadvideo_decord(
    fname: str,
    interval,
    filter_long_videos,
    filter_short_videos,
    frames_per_clip,
    frame_step,
    duration,
    num_clips,
    random_clip_sampling,
    allow_clip_overlap,
):
    """Load video content using Decord, falling back to OpenCV if Decord fails.

    color axis are in RGB order.
    
    Args:
        fname (str): Absolute path to the video sample.
        interval (tuple): Start and end indices for sampling frames. All temporal sampling is done within this interval.
        filter_long_videos (int): Maximum allowed size of the video file in bytes.
        filter_short_videos (bool): If True, skip videos shorter than the clip length.
        frames_per_clip (int): Number of frames to sample per clip.
        frame_step (int): Step size for sampling frames.
        duration (float): Duration of the clip in seconds. If not None, it overrides frame_step.
        num_clips (int): Number of clips to sample from the video (temporal sampling).
        random_clip_sampling (bool): If True, sample a random window of frames. Only relevant if vidoe length > num_clips * clip length
        allow_clip_overlap (bool): If True, allow overlap between clips. Only relevant if video length < num_clips * clip length.
        
    Returns:
        tuple: A tuple containing:
            - buffer (np.ndarray): Array of sampled frames for all clips (flattened).
            - clip_indices (list[np.array]): List of frame indices for each sampled clip."""


    if not os.path.exists(fname):
        logger.warn(f"video path not found {fname=}")
        return [], None

    _fsize = os.path.getsize(fname)
    if _fsize < 1 * 1024:  # avoid hanging issue
        logger.warn(f"video too short {fname=}")
        return [], None
    if _fsize > filter_long_videos:
        logger.warn(f"skipping long video of size {_fsize=} (bytes)")
        return [], None

    try:
        vr = VideoReader(fname, num_threads=-1, ctx=cpu(0))
    except Exception as e:
        logger.warn(f"Error loading video {fname=}, {e.__class__} {e}")
        return [], None

    fpc = frames_per_clip
    fstp = frame_step
    if duration is not None:
        try:
            fps = vr.get_avg_fps()
            fstp = int(duration * fps / fpc)
        except Exception as e:
            logger.warn(e)
            logger.warn(f"Error getting fps for video {fname=}, using frame_step={fstp}, duration={duration}, fpc={fpc}")
    clip_len = int(fpc * fstp)

    if interval is not None:
        start_interval, end_interval = interval
        start_interval = max(0, start_interval)
        end_interval = min(len(vr), end_interval)
        sample_len = end_interval - start_interval
        start_offset = start_interval
    else:
        sample_len = len(vr)
        start_offset = 0

    if filter_short_videos and sample_len < clip_len:
        logger.warn(f"skipping video of length {sample_len} out of {len(vr)}: {fname=}")
        return [], None

    vr.seek(0)  # Go to start of video before sampling frames

    if num_clips == 1 or sample_len <= clip_len:
        # either sample a random window of clip_len frames or the center clip_len frames
        if sample_len <= clip_len:
            end_indx = sample_len
        elif random_clip_sampling:
            end_indx = np.random.randint(clip_len, sample_len)
        else:
            end_indx = clip_len + (sample_len - clip_len) // 2
        start_indx = max(0, end_indx - clip_len)

        frames_to_select = fpc if sample_len >= clip_len else sample_len// fstp
        indices = np.linspace(start_indx, end_indx, frames_to_select)
        indices = np.clip(indices, 0, sample_len - 1).astype(np.int64)
        
        if len(indices) < fpc:
            # If the video is shorter than the clip length, repeat the last frame
            indices = np.concatenate((indices, np.ones(fpc - len(indices)) * (sample_len - 1))).astype(np.int64) 

        indices += start_offset

        # if we wanted several clips, but the video is too short, repeat one clip
        clip_indices = [indices] * num_clips

    else:
        start_idxs = np.linspace(0, sample_len - clip_len, num=num_clips)
        start_idxs = np.clip(start_idxs, 0, sample_len - clip_len).astype(np.int64)

        clip_indices = []
        for i, start_idx in enumerate(start_idxs):
            end_idx = start_idx + clip_len
            indices = np.linspace(start_idx, end_idx, fpc)
            indices = np.clip(indices, 0, sample_len - 1).astype(np.int64)
            assert len(indices) == fpc, f"Expected {fpc} frames, got {len(indices)}"

            indices += start_offset

            clip_indices.append(indices)


    all_indices = np.concatenate(clip_indices).tolist()
    clip_indices = np.stack(clip_indices, axis=0)  # (num_clips, fpc)

    try:
        buffer = vr.get_batch(all_indices).asnumpy()
    except Exception as e:
        buffer = load_video_cv2(fname, all_indices)
        logger.info(f"Error loading video {fname=}: {e.__class__} {e}, fallback to cv2")

    return buffer, clip_indices
