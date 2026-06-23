import torch
from privi.vjepa2.src.models.vision_transformer import vit_giant_xformers_rope, vit_huge_rope, vit_large_rope, vit_large

def load_pretrained_vjepa_pt_weights(model, pretrained_weights):
    # Load weights of the VJEPA2 encoder
    # The PyTorch state_dict is already preprocessed to have the right key names
    pretrained_dict = torch.load(pretrained_weights, weights_only=True, map_location="cpu")["target_encoder"]
    pretrained_dict = {k.replace("module.", ""): v for k, v in pretrained_dict.items()}
    pretrained_dict = {k.replace("backbone.", ""): v for k, v in pretrained_dict.items()}
    msg = model.load_state_dict(pretrained_dict, strict=False)
    print("Pretrained weights found at {} and loaded with msg: {}".format(pretrained_weights, msg))


class VJEPA2(torch.nn.Module):

    def __init__(self, cfg, device):
        """"

        VJEPA2 uses the same normalization values (ImageNet) as the original VJEPA model.

        """
        super(VJEPA2, self).__init__()

        self.cfg = cfg
        self.device = device

        model_cfg = dict(
            img_size=(cfg["resolution"], cfg["resolution"]), 
            num_frames=cfg["frames_per_clip"],
            uniform_power=True,) # pretty sure that this is what was trained
        
        model_cfg.update(cfg)

        if cfg["model_name"] == "vit_giant_xformers_rope":
            self.model = vit_giant_xformers_rope(**model_cfg)
        elif cfg["model_name"] == "vit_huge_rope":
            self.model = vit_huge_rope(**model_cfg)
        elif cfg["model_name"] == "vit_large_rope":
            self.model = vit_large_rope(**model_cfg)
        elif cfg["model_name"] == "vit_large":
            self.model = vit_large(**model_cfg)
        else:
            raise ValueError(f"Unknown model name: {cfg['model_name']}")

        self.model.to(device).eval()
        load_pretrained_vjepa_pt_weights(self.model, cfg["pretrained_path"])

        self.out_layers = cfg["out_layers"]

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
            x (torch.Tensor): (batch, channels, frames, heigh, width)

        Returns:
            torch.Tensor: (batch, tokens, output_dim)
                The final representation of the sequence (usually [batch, 16*(256+1)=4112, 1024])

        """
        batch_size, channels, frames, height, width = x.shape

        with torch.amp.autocast("cuda", enabled=self.mixed_precision, dtype=self.dtype):
            image_features = self.model(x)

        return {l: o for l, o in zip(self.out_layers, image_features)}

if __name__ == "__main__":
    import pprint

    cfg = dict(
        model_name="vit_huge_rope", #"vit_large_rope",
        resolution=256,
        frames_per_clip=16,
        pretrained_path="data/vjepa2_checkpoints/vith.pt"
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = VJEPA2(cfg, device)
    pprint.pprint(model)
    
    # Example input tensor
    x = torch.randn(2, 3, 16, 256, 256).to(device)  # (batch, channels, frames, height, width)
    
    output = model(x)
    print({idx: o.shape for idx, o in output.items()})  # Should print the shape of the output tensor