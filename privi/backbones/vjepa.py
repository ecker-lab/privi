from typing import Iterable, Optional, Sequence
import torch
from torch import nn

from privi.jepa.src.utils.model_loader import init_model

def set_trainable(m: nn.Module, trainable: bool, set_mode: bool = True):
    """Freeze/unfreeze a module and optionally flip its train/eval mode."""
    for p in m.parameters():
        p.requires_grad = trainable
    if set_mode:
        m.train(trainable)

def configure_vit_finetuning(
    encoder: nn.Module,
    mode: str = "frozen",                        # "frozen", "partial", "full"
    finetune_blocks: Optional[Iterable[int]] = None,  # block indices to train (e.g., {22, 23} for last two blocks)
    freeze_dropout_in_frozen_parts: bool = True,
):
    """
    Make a ViT encoder frozen/partial/full. Supports common ViT layouts:
      - encoder.patch_embed
      - encoder.pos_embed (Parameter or buffer)
      - encoder.cls_token / encoder.dist_token
      - encoder.blocks: nn.ModuleList of Transformer blocks
      - encoder.norm / encoder.fc_norm / head, depending on implementation
    """
    mode = mode.lower()

    # Helper: find optional pieces by attribute name
    def maybe_get(name):
        return getattr(encoder, name, None)

    # 1) Default: everything frozen + eval
    if mode == "frozen":
        set_trainable(encoder, False, set_mode=True)  # puts entire model in eval()
        # If you want to train only a *separate* classifier head outside `encoder`, handle that module separately.

    # 2) Full finetune: everything trainable + train
    elif mode == "full":
        set_trainable(encoder, True, set_mode=True)

    elif mode == "finetune":
        set_trainable(encoder, True, set_mode=True)

        t = getattr(encoder, "pos_embed", None)
        if isinstance(t, nn.Parameter):
            t.requires_grad = False

    # 3) Partial finetune: select blocks + some norms/head
    elif mode == "layer_finetune":
        if finetune_blocks is None:
            raise ValueError("For mode='partial', provide finetune_blocks, e.g. {23} for last block on ViT-L (24 blocks).")

        # First freeze *everything* and put frozen parts into eval() to stop dropout in them
        set_trainable(encoder, False, set_mode=freeze_dropout_in_frozen_parts)

        # Now mark selected blocks as trainable and set them to train mode
        assert hasattr(encoder, "blocks"), "Expected encoder.blocks (ModuleList of Transformer blocks)."
        num_blocks = len(encoder.blocks)
        # sanitize provided indices
        finetune_blocks = {i if i >= 0 else num_blocks + i for i in finetune_blocks}
        for i, blk in enumerate(encoder.blocks):
            if i in finetune_blocks:
                set_trainable(blk, True, set_mode=True)
            else:
                if freeze_dropout_in_frozen_parts:
                    blk.eval()  # keep frozen part deterministic

        # Typical ViT practice: also tune the final normalization(s) and the classification head
        # Different repos name these slightly differently; try common names:
        if maybe_get("norm") is not None:
            set_trainable(encoder.norm, True, set_mode=True)
        # if "pre_norm" in also_tune and maybe_get("pre_norm") is not None:
        #     set_trainable(encoder.pre_norm, True, set_mode=True)
        # if "post_norm" in also_tune and maybe_get("post_norm") is not None:
        #     set_trainable(encoder.post_norm, True, set_mode=True)
        # if "fc_norm" in also_tune and maybe_get("fc_norm") is not None:
        #     set_trainable(encoder.fc_norm, True, set_mode=True)
        # if "head" in also_tune and maybe_get("head") is not None:
        #     set_trainable(encoder.head, True, set_mode=True)

    else:
        raise ValueError(f"Unknown training mode: {mode}")

    # --- Edge cases: positional embeddings & tokens (no grads usually; handled at resize/load time) ---
    # By default we keep pos_embed/tokens frozen; you almost never want to train them for last-layer FT.
    # But we ensure they are parameters with requires_grad=False.
    # for name in ("pos_embed", "cls_token", "dist_token"):
    #     t = getattr(encoder, name, None)
    #     if isinstance(t, nn.Parameter):
    #         t.requires_grad = False

    # Patch embedding usually stays frozen for “last-layer FT”.
    # if hasattr(encoder, "patch_embed"):
    #     set_trainable(encoder.patch_embed, mode == "full", set_mode=(mode != "partial"))

    return encoder

class VJEPA(torch.nn.Module):

    def __init__(self, cfg: dict, device):
        super(VJEPA, self).__init__()

        self.training_mode = cfg.get("training_mode", "frozen")

        encoder = init_model(
            crop_size=cfg.get("resolution", 224),
            device=device,
            pretrained=cfg["pretrained_path"],
            model_name=cfg.get("model_name", 'vit_large'),
            patch_size=cfg.get("patch_size", 16),
            tubelet_size=cfg.get("tubelet_size", 2),
            frames_per_clip=cfg.get("frames_per_clip", 8),
            uniform_power=cfg.get("uniform_power", True),
            checkpoint_key=cfg.get("checkpoint_key", 'target_encoder'),
            use_SiLU=cfg.get("use_silu", False),
            tight_SiLU=cfg.get("tight_silu", False),
            use_sdpa=cfg.get("use_sdpa", True),
            out_layers=cfg.get("out_layers", [23]),
        )

        configure_vit_finetuning(encoder, 
                                 self.training_mode, 
                                 cfg.get("finetune_layers", []), 
                                 freeze_dropout_in_frozen_parts=True)

        self.cfg = cfg
        self.device = device
        self.encoder = encoder

        which_dtype = self.cfg.get("dtype", "float32").lower()
        if which_dtype.lower() == 'bfloat16':
            self.dtype = torch.bfloat16
            self.mixed_precision = True
        elif which_dtype.lower() == 'float16':
            self.dtype = torch.float16
            self.mixed_precision = True
        else:
            self.dtype = torch.float32
            self.mixed_precision = False

        
        
    def forward(self, x):
        """

        Args:
            x (torch.Tensor): (batch, channels, frames, height, width)

        Returns:
            dict[torch.Tensor]: {out_layer: (batch, n_tokens, d_model)}
                The final representation of the sequence (usually [batch, 1568, 1024])

        """

        

        with torch.amp.autocast("cuda", enabled=self.mixed_precision, dtype=self.dtype):
            output = self.encoder.forward(x)
            
        return {self.cfg["out_layers"][i]: output[i] for i in range(len(self.cfg["out_layers"]))}

    def train(self, mode=True):
        """
        Set the training mode of the model.
        Args:
            mode (bool): If True, set the model to training mode, otherwise set it to evaluation mode.
        """
        if self.training_mode == "frozen":
            mode = False
        self.encoder.train(mode)
        return self