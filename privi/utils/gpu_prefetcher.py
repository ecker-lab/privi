import torch


def _move_to_device(obj, device):
    if torch.is_tensor(obj):
        return obj.to(device, non_blocking=True)
    elif isinstance(obj, dict):
        return {k: _move_to_device(v, device) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        seq = [_move_to_device(x, device) for x in obj]
        return type(obj)(seq)  # keep list vs tuple
    else:
        return obj  # e.g., ints, strings

class GPUPrefetcher:
    def __init__(self, dataloader, device):
        self.loader = dataloader
        self.device = device
        self.stream = torch.cuda.Stream(device=device)
        self.iter = None
        self.next_batch = None

    def __iter__(self):
        self.iter = iter(self.loader)
        self._preload()
        return self

    def __next__(self):
        if self.next_batch is None:
            raise StopIteration
        # Make default stream wait until prefetch stream finished moving tensors
        torch.cuda.current_stream(self.device).wait_stream(self.stream)
        batch = self.next_batch
        self.next_batch = None
        self._preload()
        return batch

    def _preload(self):
        try:
            batch = next(self.iter)
        except StopIteration:
            self.next_batch = None
            return
        with torch.cuda.stream(self.stream):
            self.next_batch = _move_to_device(batch, self.device)
