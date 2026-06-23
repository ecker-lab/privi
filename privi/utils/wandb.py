# wbipc.py
import os
from pprint import pprint
import time
import queue
import traceback
import multiprocessing as mp
from typing import List

# Messages are small dicts:
#   {"type": "log", "data": {...}, "step": int, "commit": bool}
#   {"type": "summary", "data": {...}}
#   {"type": "artifact", "name": str, "type": str, "files": [paths], "metadata": dict}
#   {"type": "close"}

def _wandb_worker(head_idx, proj, group, run_name, cfg, save_dir, q: mp.Queue, disabled):
    import wandb
    os.environ.setdefault("WANDB_CONSOLE","off")  # quiet logs in worker
    if disabled:
        os.environ["WANDB_DISABLED"] = "true"
    run = wandb.init(
        project=proj, group=group, name=run_name, config=cfg, dir=save_dir, mode="disabled" if disabled else "online"
        # mode="offline",  # if you need it sometimes
    )

    try:
        while True:
            try:
                msg = q.get(timeout=1.0)
            except queue.Empty:
                continue

            t = msg.get("type")

            if t == "log":
                data = msg["data"]
                step = msg.get("step")
                commit = msg.get("commit", None)
                # Allow caller to control step/commit; both are optional
                if step is None and commit is None:
                    wandb.log(data)                  # default behavior
                elif step is None:
                    wandb.log(data, commit=commit)
                elif commit is None:
                    wandb.log(data, step=step)
                else:
                    wandb.log(data, step=step, commit=commit)

            elif t == "summary":
                for k, v in msg["data"].items():
                    run.summary[k] = v

            elif t == "artifact":
                name = msg["name"]
                art_type = msg["type"]
                metadata = msg.get("metadata")
                files = msg.get("files", [])
                art = wandb.Artifact(name=name, type=art_type, metadata=metadata)
                for f in files:
                    art.add_file(f)
                run.log_artifact(art)

            elif t == "close":
                break

    except Exception:
        # If a worker dies, we still try to close the run so W&B marks it ended
        traceback.print_exc()
    finally:
        try:
            wandb.finish()
        except Exception:
            pass

class StdoutLogger:
    """One W&B run in its own process."""
    def __init__(self):
        pass
        
    def log(self, data: dict, step: int | None = None, commit: bool | None = None):
        pprint(data)

class WandbLogger:
    """One W&B run in its own process."""
    def __init__(self):
        self.q = mp.get_context("spawn").Queue(maxsize=500)
        

    def log(self, data: dict, step: int | None = None, commit: bool | None = None):
        self.q.put({"type": "log", "data": data, "step": step, "commit": commit})

    def summary(self, data: dict):
        self.q.put({"type": "summary", "data": data})

    def artifact(self, name: str, art_type: str, files: list[str], metadata: dict | None = None):
        self.q.put({"type": "artifact", "name": name, "type": art_type, "files": files, "metadata": metadata})
        

class WandbFanout:
    """
    Manages N logger processes (one per head). Call .log(head, {...}, step)
    from your training process. No wandb runs in the trainer process.
    """

    def __init__(self, n_heads: int, project: str, group: str, base_run: str, base_cfg: dict, save_dir: str, disabled=False):
        self.processes = []
        self.runs = []

        for h in range(n_heads):
            run_name = f"{base_run}-head{h:02d}"
            cfg = dict(base_cfg)
            cfg["active_head_idx"] = h

            self.runs.append(WandbLogger())

            self.processes.append(mp.get_context("spawn").Process(
                target=_wandb_worker,
                args=(h, project, group, run_name, cfg, save_dir, self.runs[-1].q, disabled),
                daemon=True,
            ))
            self.processes[-1].start()

    def __getitem__(self, idx: int) -> WandbLogger:
        return self.runs[idx]
    
    def __len__(self):
        return len(self.runs)

    def log(self, head_idx: int, data: dict, step: int | None = None, commit: bool | None = None):
        self.runs[head_idx].log(data, step=step, commit=commit)

    def summary(self, head_idx: int, data: dict):
        self.runs[head_idx].summary(data)

    def close(self, wait: bool = True, timeout: float = 60.0):
        for q, p in zip(self.runs, self.processes):
            try:
                q.put({"type": "close"})
            except Exception:
                pass
            if wait:
                p.join(timeout=timeout)
                if p.is_alive():
                    p.terminate()
