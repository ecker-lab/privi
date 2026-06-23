
from einops import rearrange


def get_backbone(backbone_name, cfg: dict, device):
    """
    Get the backbone model based on the provided name and configuration.

    Args:
        backbone_name (str): Name of the backbone model.
        cfg (dict): Configuration dictionary containing model parameters.
        device: Device to load the model onto (e.g., 'cuda' or 'cpu').

    Returns:
        torch.nn.Module: The initialized backbone model.
    """
    if backbone_name == "vjepa":
        from privi.backbones.vjepa import VJEPA
        return VJEPA(cfg, device)
    else:
        raise ValueError(f"Unknown backbone: {backbone_name}")



def reshape_latents(backbone_type, latents):
    """ Reshape the latents based on the backbone model.

    Note that DINOv2 orders width and height as (width, height) while VJEPA uses (time, height, width).

    Args:
        backbone_name (str): Name of the backbone model.
        latents (torch.Tensor): Latent representations from the model of shape (*, num_tokens, latent_dim).

    Returns:
        tuple: A tuple containing:
            - cls_token (torch.Tensor): Class token of shape (*, 1, latent_dim) or None if not applicable.
            - patch_tokens (torch.Tensor): Patch tokens of shape (*, latent_dim, time, height, width). time=1 for image models.
    """

    if backbone_type == "vjepa":
        num_tokens, latent_dim = latents.shape[-2], latents.shape[-1]
        assert num_tokens in [8 * 14 * 14, 8 * 24 * 24], f"""VJEPA latents should have 8 * 14 * 14 or 8 * 24 * 24 patch tokens. got {num_tokens}."""

        wh = 14 if num_tokens == 8 * 14 * 14 else 24

        patch_tokens = rearrange(latents, "... (time height width) latent_dim -> ... latent_dim time height width", time=8, width=wh, height=wh)

        return None, patch_tokens
    else:
        raise ValueError(f"Unknown backbone type: {backbone_type}. Cannot reshape latents.")