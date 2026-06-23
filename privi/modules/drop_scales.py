import torch
from torch import nn



class DropScales(nn.Module):
    """
    Drop entire scales (lists entries) with stochastic depth semantics,
    but WITHOUT rescaling survivors. One mask per batch (not per sample).

    Args:
      drop_prob: float or list/tuple of per-scale drop probs in [0,1).
      force_keep_one: ensure at least one scale is kept each forward.
    """
    def __init__(self, drop_prob=0.1, force_keep_one=True):
        super().__init__()
        if isinstance(drop_prob, (list, tuple)):
            self.register_buffer("drop_prob", torch.tensor(drop_prob, dtype=torch.float32))
        else:
            self.register_buffer("drop_prob", torch.tensor([float(drop_prob)], dtype=torch.float32))
        self.force_keep_one = force_keep_one

    def update_drop_prob(module, step, total_steps, target_probs):
        """Linearly increase drop probs from 0 -> target_probs over first 40% of training."""
        ramp_steps = int(0.4 * total_steps)
        factor = min(1.0, step / ramp_steps)
        if not isinstance(target_probs, (list, tuple)):
            target_probs = [target_probs]
        new_p = [p * factor for p in target_probs]
        module.drop_prob.copy_(torch.tensor(new_p, device=module.drop_prob.device))

    def extra_repr(self):
        return f"drop_prob={self.drop_prob.tolist()}, force_keep_one={self.force_keep_one}"

    def forward(self, feats):
        """
        feats: list of tensors, one per scale, each [B, N_i, D] or [B, ..., D].
               (We just pass them through or remove them; no shape assumptions.)
        Returns: list of kept tensors (subset of feats).
        """
        if (not self.training) or self.drop_prob.max().item() <= 0.0:
            return feats

        S = len(feats)
        assert S >= 1

        # device: use first tensor's device
        device = feats[0].device

        # broadcast probs to S
        if self.drop_prob.numel() == 1:
            p = self.drop_prob.expand(S)  # same p for all scales
        else:
            assert self.drop_prob.numel() == S, "len(drop_prob) must equal num scales"
            p = self.drop_prob

        # sample one mask for the whole batch
        u = torch.rand(S, device=device)
        keep = (u >= p)  # [S] bool

        if self.force_keep_one and not keep.any():
            keep[u.argmax()] = True

        # return only the kept scales (no rescaling)
        return [f for f, k in zip(feats, keep.tolist()) if k]