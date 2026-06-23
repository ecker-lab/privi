from datetime import datetime
import random
import logging
import os

import time, sys
from contextlib import contextmanager

@contextmanager
def timer(label: str = "", stream=sys.stderr):
    t0 = time.perf_counter()
    try:
        yield
    finally:
        dt = time.perf_counter() - t0
        msg = f"{label} took {dt:.3f}s" if label else f"{dt:.3f}s"
        print(msg, file=stream, flush=True)

def normalize_dtype(dtype_like):
    import torch
    
    if isinstance(dtype_like, torch.dtype) or dtype_like is None:
        return dtype_like
    if isinstance(dtype_like, str):
        m = {
            "float32": torch.float32, "fp32": torch.float32, "f32": torch.float32,
            "float16": torch.float16, "fp16": torch.float16, "f16": torch.float16, "half": torch.float16,
            "bfloat16": torch.bfloat16, "bf16": torch.bfloat16,
            "float64": torch.float64, "fp64": torch.float64, "f64": torch.float64, "double": torch.float64,
        }
        try:
            return m[dtype_like.lower()]
        except KeyError:
            raise ValueError(f"Unknown dtype string: {dtype_like!r}")
    raise TypeError(f"Unsupported dtype spec: {type(dtype_like)}")

def setup_train(project_id, save_dir, cfg_name, resume_id=None):

    logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")

    timestamp = datetime.now().strftime("%y%m%d-%H%M")
    run_id = f"{cfg_name}-{timestamp}" if resume_id is None else resume_id

    ckpt_dir = f"{save_dir}/checkpoints/{project_id}/{run_id}/"

    while True:
        try:
            os.makedirs(ckpt_dir)
            break
        except FileExistsError:
            run_id = f"{cfg_name}-{timestamp}-{random.randint(0, 10000)}"
            ckpt_dir = f"{save_dir}/checkpoints/{project_id}/{run_id}/"
            logging.warning(f"Checkpoint directory {ckpt_dir} already exists. Trying again with a different run_id.")

    return run_id, ckpt_dir

def count_parameters(m: "torch.nn.Module", only_trainable: bool = True):
    """
    Returns the total number of parameters used by `m` (only counting
    shared parameters once); if `only_trainable` is True, then only
    includes parameters with `requires_grad = True`
    """
    parameters = list(m.parameters())
    if only_trainable:
        parameters = [p for p in parameters if p.requires_grad]
    unique = {p.data_ptr(): p for p in parameters}.values()
    return sum(p.numel() for p in unique)

def timestamp(mode="long"):
    if mode == "short":
        return datetime.now().strftime("%y%m%d-%H%M")
    elif mode == "short_seconds":
        return datetime.now().strftime("%y%m%d-%H%M%S")
    else:
        return datetime.now().strftime("%Y-%m-%dT%H-%M-%S%Z")


