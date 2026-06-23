#!/usr/bin/env python3
"""
Produce video snippets (spatial + temporal crops) from a CSV.

CSV columns: snippet_id, video_filename, start_time, end_time, x1, y1, x2, y2
  - Bbox is normalized to [0, 1] in source coordinates.
  - The first two underscore-separated components of `snippet_id` become
    subdirectories in the output tree, e.g.
      yt_0001_00510_0 -> <save_dir>/yt/0001/yt_0001_00510_0.mp4

Frames are loaded with decord. Snippets sharing a source video are processed
in chunks: their target frame indices (at 30 fps, nearest-source-frame) are
unioned and read with a single VideoReader.get_batch() call, then sliced,
cropped, resized per snippet and piped to ffmpeg for libx264 encoding.
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

from collections import defaultdict
from functools import partial
from multiprocessing import Pool
from pathlib import Path
import secrets
import subprocess

import av
import cv2
import numpy as np
import pandas as pd
from decord import VideoReader, cpu
from tqdm import tqdm
import fire

cv2.setNumThreads(1)


FFMPEG = "ffmpeg"
TARGET_FPS = 30


def snippet_out_path(save_dir: Path, snippet_id: str) -> Path:
    parts = snippet_id.split("_")
    if len(parts) == 4:
        ds, video_id, frame, ind = parts
    elif len(parts) == 5:
        ds1, ds2, video_id, frame, ind = parts
        ds = f"{ds1}_{ds2}"
    else:
        raise ValueError(f"illegal id: {snippet_id}")
    return save_dir / ds / video_id / f"{snippet_id}.mp4"


def compute_target_indices(snippet, source_fps, n_source_frames):
    duration = snippet["end_time"] - snippet["start_time"]
    n_out = int(duration * TARGET_FPS)
    if n_out < 1:
        return None
    ks = np.arange(n_out)
    times = snippet["start_time"] + ks / TARGET_FPS
    idx = np.rint(times * source_fps).astype(int)
    idx = np.clip(idx, 0, n_source_frames - 1)
    return idx


def resolve_input(video_base_path, video_filename, fallback_video_base_path):
    in_path = Path(video_base_path) / video_filename
    if in_path.exists():
        return in_path, None
    webm = in_path.with_suffix(".webm")
    if webm.exists():
        return webm, None
    in_path = Path(fallback_video_base_path) / video_filename
    if in_path.exists():
        print(f'Using fallback for video {video_filename}')
        return in_path, "fallback"
    webm = in_path.with_suffix(".webm")
    if webm.exists():
        print(f'Using fallback for video {video_filename}')
        return webm, "fallback"
    return None, None


def encode_snippet(rgb_frames, out_path: Path, threads: int, preset: str) -> int:
    T, H, W, _ = rgb_frames.shape
    tmp_path = out_path.with_suffix(f".tmp.{secrets.token_hex(4)}.mp4")
    cmd = [
        FFMPEG,
        "-hide_banner",
        "-loglevel", "error",
        "-y",
        "-f", "rawvideo",
        "-pix_fmt", "rgb24",
        "-s", f"{W}x{H}",
        "-r", str(TARGET_FPS),
        "-i", "pipe:0",
        "-an",
        "-map_metadata", "-1",
        "-map_chapters", "-1",
        "-fflags", "+bitexact",
        "-flags:v", "+bitexact",
        "-c:v", "libx264",
        "-preset", preset,
        "-crf", "18",
        "-pix_fmt", "yuv420p",
        "-threads", str(threads),
        str(tmp_path),
    ]
    proc = subprocess.run(cmd, input=rgb_frames.tobytes(), capture_output=True)
    if proc.returncode != 0:
        print(proc.stderr.decode(errors="replace"))
        tmp_path.unlink(missing_ok=True)
        return proc.returncode
    os.rename(tmp_path, out_path)
    return 0


class FrameReader:
    """Random-access frame reader with a decord backend and a PyAV fallback.

    Some source videos are encoded in ways decord refuses to open, or that it
    opens but chokes on mid-decode (odd codecs, broken headers, AV1 with the
    bundled libav, ...). For those we transparently fall back to PyAV, which
    binds the conda ffmpeg and handles a wider codec set (including AV1 via
    libdav1d).

    Surface used by process_chunk:
      - reader.fps                -> average fps (float)
      - len(reader)               -> total frame count
      - reader.frame_shape()      -> (H, W, 3)
      - reader.get_batch(indices) -> (N, H, W, 3) uint8 RGB ndarray
      - reader.close()
    """

    def __init__(self, path, num_threads):
        self.path = path
        self._num_threads = num_threads
        self._decord = None
        self._pyav_container = None
        self._pyav_stream = None
        self._pyav_iter = None
        self._pyav_pos = 0
        try:
            vr = VideoReader(str(path), num_threads=num_threads, ctx=cpu(0))
            self._fps = float(vr.get_avg_fps())
            self._total = len(vr)
            self._decord = vr
            self.backend = "decord"
        except Exception as e:
            self._open_pyav(reason=f"open failed: {type(e).__name__}: {e}")

    def __len__(self):
        return self._total

    @property
    def fps(self):
        return self._fps

    def _open_pyav(self, reason):
        container = av.open(str(self.path))
        if not container.streams.video:
            container.close()
            raise RuntimeError(
                f"both decord and pyav failed to read {self.path} "
                f"(decord: {reason}; pyav: no video stream)"
            )
        stream = container.streams.video[0]
        try:
            stream.thread_type = "AUTO"
            stream.thread_count = max(1, self._num_threads)
        except Exception:
            pass
        self._pyav_container = container
        self._pyav_stream = stream
        self._pyav_pos = 0
        self._pyav_iter = container.decode(stream)
        self.backend = "pyav"
        rate = stream.average_rate or stream.guessed_rate
        self._fps = float(rate) if rate else 0.0
        total = stream.frames or 0
        if total <= 0:
            # Some containers don't populate stream.frames; estimate from
            # duration so process_chunk doesn't bail on n_source_frames <= 0.
            if stream.duration and stream.time_base and self._fps:
                total = int(float(stream.duration * stream.time_base) * self._fps)
            elif container.duration and self._fps:
                total = int((container.duration / 1_000_000.0) * self._fps)
        self._total = total
        print(f"    decord failed ({reason}); falling back to pyav "
              f"({self._total} frames, {self._fps:.2f} fps) for {self.path}",
              flush=True)

    def frame_shape(self):
        if self._decord is not None:
            try:
                return tuple(self._decord[0].shape)
            except Exception as e:
                # First actual decode — the decord constructor only parses the
                # container header, so threaded-decoder errors surface here.
                self._decord = None
                self._open_pyav(
                    reason=f"frame_shape failed: {type(e).__name__}: {e}"
                )
        cc = self._pyav_stream.codec_context
        return (int(cc.height), int(cc.width), 3)

    def get_batch(self, indices) -> np.ndarray:
        if self._decord is not None:
            try:
                return self._decord.get_batch(indices).asnumpy()
            except Exception as e:
                # decord opened the file but choked on decode (commonly an
                # internal libav error). Drop it and re-read this batch from a
                # fresh pyav container, which seeks from the start.
                self._decord = None
                self._open_pyav(
                    reason=f"get_batch failed: {type(e).__name__}: {e}"
                )
        return self._read_pyav(indices)

    def _read_pyav(self, indices) -> np.ndarray:
        # Decode each wanted frame once (in ascending order), then reassemble
        # in the caller's requested order. PyAV gives us a forward iterator
        # over decoded frames; to rewind we seek the container to 0 and start
        # a fresh decode iterator.
        want = sorted({int(i) for i in indices})
        frames_by_idx = {}
        for target in want:
            if target < self._pyav_pos:
                self._pyav_container.seek(0)
                self._pyav_pos = 0
                self._pyav_iter = self._pyav_container.decode(self._pyav_stream)
            while self._pyav_pos <= target:
                try:
                    frame = next(self._pyav_iter)
                except StopIteration:
                    break
                cur = self._pyav_pos
                self._pyav_pos += 1
                if cur == target:
                    frames_by_idx[target] = frame.to_ndarray(format="rgb24")
            if target not in frames_by_idx:
                raise RuntimeError(
                    f"pyav could not read frame {target} from {self.path}"
                )
        return np.stack([frames_by_idx[int(i)] for i in indices], axis=0)

    def close(self):
        if self._pyav_container is not None:
            try:
                self._pyav_container.close()
            except Exception:
                pass
            self._pyav_container = None
            self._pyav_stream = None
            self._pyav_iter = None
        self._decord = None


def process_chunk(chunk, save_dir, video_base_path, fallback_video_base_path, short_edge_px, threads, preset):
    results = []
    video_filename = chunk[0]["video_filename"]
    save_dir_p = Path(save_dir)

    pending = []
    for s in chunk:
        out_path = snippet_out_path(save_dir_p, s["snippet_id"])
        if out_path.exists():
            results.append((s["snippet_id"], "existed"))
        else:
            pending.append((s, out_path))

    if not pending:
        return results

    in_path, log_notice = resolve_input(video_base_path, video_filename, fallback_video_base_path=fallback_video_base_path)
    if in_path is None:
        for s, _ in pending:
            results.append((s["snippet_id"], "not_found"))
        return results

    try:
        vr = FrameReader(in_path, num_threads=threads)
        source_fps = vr.fps
        n_source_frames = len(vr)
        H_src, W_src, _ = vr.frame_shape()
    except Exception as e:
        print(f"Failed to open {in_path}: {e}")
        for s, _ in pending:
            results.append((s["snippet_id"], "failed_probe"))
        return results

    try:
        return _process_opened(
            vr, pending, results, source_fps, n_source_frames,
            H_src, W_src, in_path, short_edge_px, threads, preset,
        )
    finally:
        vr.close()


def _process_opened(vr, pending, results, source_fps,
                    n_source_frames, H_src, W_src, in_path, short_edge_px,
                    threads, preset):
    if source_fps <= 0 or n_source_frames <= 0:
        for s, _ in pending:
            results.append((s["snippet_id"], "failed_probe"))
        return results

    snippet_plans = []
    union = set()
    for s, out_path in pending:
        idx = compute_target_indices(s, source_fps, n_source_frames)
        if idx is None:
            results.append((s["snippet_id"], "too_short"))
            continue
        x1 = int(round(s["x1"] * W_src))
        y1 = int(round(s["y1"] * H_src))
        x2 = int(round(s["x2"] * W_src))
        y2 = int(round(s["y2"] * H_src))
        x1, x2 = max(0, min(x1, x2)), min(W_src, max(x1, x2))
        y1, y2 = max(0, min(y1, y2)), min(H_src, max(y1, y2))
        if (x2 - x1) < 2 or (y2 - y1) < 2:
            results.append((s["snippet_id"], "bad_bbox"))
            continue
        snippet_plans.append({
            "snippet": s,
            "out_path": out_path,
            "idx": idx,
            "crop": (y1, y2, x1, x2),
        })
        union.update(int(i) for i in idx)

    if not snippet_plans:
        return results

    union_sorted = sorted(union)
    pos = {idx: i for i, idx in enumerate(union_sorted)}


    try:
        frames = vr.get_batch(union_sorted)
    except Exception as e:
        print(f"get_batch failed for {in_path}: {e}")
        for p in snippet_plans:
            results.append((p["snippet"]["snippet_id"], "failed_decode"))
        return results

    for p in snippet_plans:
        s = p["snippet"]
        out_path = p["out_path"]
        y1, y2, x1, x2 = p["crop"]
        rows = [pos[int(i)] for i in p["idx"]]
        snippet_frames = frames[rows, y1:y2, x1:x2, :]
        T, h, w, _ = snippet_frames.shape

        short = min(h, w)
        if short > short_edge_px:
            scale = short_edge_px / short
            new_h = int(round(h * scale))
            new_w = int(round(w * scale))
        else:
            new_h, new_w = h, w
        new_h -= new_h % 2
        new_w -= new_w % 2
        if new_h < 2 or new_w < 2:
            results.append((s["snippet_id"], "bad_bbox"))
            continue

        if (new_h, new_w) != (h, w):
            resized = np.empty((T, new_h, new_w, 3), dtype=np.uint8)
            for t in range(T):
                resized[t] = cv2.resize(
                    snippet_frames[t], (new_w, new_h), interpolation=cv2.INTER_AREA
                )
            snippet_frames = resized
        elif not snippet_frames.flags["C_CONTIGUOUS"]:
            snippet_frames = np.ascontiguousarray(snippet_frames)

        out_path.parent.mkdir(parents=True, exist_ok=True)
        ret = encode_snippet(snippet_frames, out_path, threads, preset)
        results.append((s["snippet_id"], ret))

    #print('.', flush=True)

    return results


def main(
    csv_path,
    save_dir,
    video_base_path,
    fallback_video_base_path=None,
    short_edge_px=384,
    preset="medium",
    n_snippets_per_chunk=1,
    check_existence=True,
    n_samples=None,
    n_processes=8,
    threads_per_process=1,
):
    df = pd.read_csv(csv_path)
    snippets = df.to_dict(orient="records")
    print(f"Loaded {len(snippets)} snippets from {csv_path}")

    if check_existence:
        save_dir_p = Path(save_dir)
        snippets = [
            s for s in snippets
            if not snippet_out_path(save_dir_p, s["snippet_id"]).exists()
        ]
        print(f"Filtered to {len(snippets)} unprocessed snippets")

    if n_samples is not None:
        snippets = snippets[:n_samples]

    groups = defaultdict(list)
    for s in snippets:
        groups[s["video_filename"]].append(s)
    chunks = []
    for group in groups.values():
        for i in range(0, len(group), n_snippets_per_chunk):
            chunks.append(group[i:i + n_snippets_per_chunk])
    print(f"Grouped into {len(chunks)} chunks across {len(groups)} source videos")

    worker = partial(
        process_chunk,
        save_dir=str(save_dir),
        video_base_path=str(video_base_path),
        fallback_video_base_path=str(fallback_video_base_path),
        short_edge_px=short_edge_px,
        threads=threads_per_process,
        preset=preset,
    )

    print(f"Processing with {n_processes} processes ({threads_per_process} threads each)")
    with Pool(processes=n_processes) as pool:
        for _ in tqdm(pool.imap_unordered(worker, chunks), total=len(chunks)):
            pass


if __name__ == "__main__":
    fire.Fire(main)
