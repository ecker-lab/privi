
import os
from typing import Dict
from privi.utils.gpu_prefetcher import GPUPrefetcher

class LatentsIter:
    """"
    Simple base implementation, may be overwritten
    """

    def __init__(self, local_loader, global_loader, device):
        self.local_loader = local_loader
        self.global_loader = global_loader
        self.prefetch_to = device

    def __iter__(self):
        local_loader = self.local_loader
        global_loader = self.global_loader

        if self.prefetch_to is not None:
            local_loader = GPUPrefetcher(local_loader, self.prefetch_to)
            if hasattr(global_loader, "__len__"):  # if it’s a real loader, not repeat(None)
                global_loader = GPUPrefetcher(global_loader, self.prefetch_to)

        for l, g in zip(local_loader, global_loader):
            yield {"local": l, "global": g}

    def __len__(self):
        return len(self.local_loader)
    
    def num_crops(self):
        return self.local_loader.dataset.num_crops()

class BaseLatents:
    """
    The different latents classes implement how the latent representations used for head training are produced.

    This base class does some generic preprocessing of paths to datasets that are relevant for all kinds of latents.
    """
    data_paths : Dict[str, str]

    def __init__(self, args_data):

        data_paths = args_data.get("paths")

        self.data_paths = data_paths

    def get_iter(self, split: str) -> LatentsIter:
        raise NotImplementedError()