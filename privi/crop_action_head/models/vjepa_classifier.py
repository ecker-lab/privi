
from torch import nn
from einops import rearrange



class VJEPAClassifier(nn.Module):
    """
    Wrapper around the AttentiveClassifier to accept the right forward parameters.
    """

    def __init__(self, embed_dim_global, embed_dim_local, num_classes, num_scales, latent_dim=None, **kwargs):
        super().__init__()
        from privi.jepa.src.models.attentive_pooler import AttentiveClassifier

        if latent_dim is not None:
            self.proj = nn.Linear(embed_dim_local, latent_dim)
        else:
            self.proj = nn.Identity()
            latent_dim = embed_dim_local
        self.classifier = AttentiveClassifier(embed_dim=latent_dim, num_classes=num_classes, **kwargs)
        

    def forward(self, x_local, **kwargs):
        x_local = x_local[0]  # Assuming x_local is a list of features, take the first one
        x_local = rearrange(x_local, "B 1 C T H W -> B (T H W) C")  # Flatten time, height, width
        x_local = self.proj(x_local)
        return self.classifier(x_local).unsqueeze(1)  # Add crops dimension, shape (B, 1, num_classes)
    
    def step(*args, **kwargs):
        pass

def main(config: str):

    import yaml
    from privi.utils.misc import count_parameters

    with open(config) as fp:
        args = yaml.safe_load(fp)

    args_head = args.get("head", dict())
    args_data = args.get("data")
    embed_dim_global = args_data.get("embed_dim_global")
    embed_dim_local = args_data.get("embed_dim_local")

    classifier_args = dict(
            embed_dim_global=embed_dim_global,
            embed_dim_local=embed_dim_local,
            num_classes=23,
            num_scales=1,
            **args_head,
    )
    
    model = VJEPAClassifier(**classifier_args)

    print(f"BACKBONE PARAMETERS (trainable): {count_parameters(model) / 1000 / 1000:.2f}M")

if __name__ == "__main__":
    from fire import Fire
    Fire(main)