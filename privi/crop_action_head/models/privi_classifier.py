import math
from einops import rearrange, repeat
from torch import nn
import torch

from privi.jepa.src.models.utils.modules import CrossAttentionBlock
from privi.jepa.src.utils.tensors import trunc_normal_

import torch
import torch.nn as nn

from privi.modules.drop_scales import DropScales


class PriViClassifier(nn.Module):
    """
    Our attentive classifier head
    """

    def __init__(
        self,
        num_classes,
        embed_dim_global,
        embed_dim_local,
        embed_dim=None,
        depth=1,
        num_heads=16,
        mlp_ratio=4.0,
        qkv_bias=True,
        norm_layer="LN",
        init_std=0.02,
        query_init_path=None,
        query_token_mode="per_class",
        grid_size=None,
        input_projection="learned",
        attn_drop=0.0,
        proj_drop=0.0,
        downsample_local=None,
        num_scales=1,
        norm_input=False,
        scale_drop=0.0,
        use_local_encoding=True,
        use_ca=False,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.scale_drop = scale_drop

        if embed_dim is None:
            embed_dim = embed_dim_local

        if embed_dim_global is None:
            embed_dim_global = embed_dim_local
            self.proj_global = nn.Identity()
        else:
            self.proj_global = nn.Linear(embed_dim_global, embed_dim)
        
        self.proj_local = nn.ModuleList(
            nn.Linear(embed_dim_local, embed_dim) if input_projection != "none" else nn.Identity()
            for _ in range(num_scales))
        
        self.in_norm_local = nn.ModuleList(
            nn.LayerNorm(embed_dim_local) if norm_input else nn.Identity()
            for _ in range(num_scales)
        )

        self.drop_scales = DropScales(drop_prob=scale_drop)
        self.scale_drop_target = scale_drop

        self.query_token_mode = query_token_mode
        if query_token_mode == "per_class":
            self.num_queries = num_classes
        elif query_token_mode == "single":
            self.num_queries = 1
        else:
            raise ValueError(f"Unknown query_token_mode: {query_token_mode}")

        self.downsample_local = downsample_local
        self.printed_local_shape = False

        self.query_tokens = nn.Parameter(torch.zeros(1, self.num_queries, embed_dim))

        self.global_context_encoding = nn.Parameter(torch.zeros(1, embed_dim, 1, 1, 1))
        self.global_roi_encoding = nn.Parameter(torch.zeros(1, embed_dim, 1, 1, 1))

        self.use_local_encoding = use_local_encoding
        if use_local_encoding:
            self.local_encoding =  nn.ParameterList(
                nn.Parameter(torch.zeros(1, 1, embed_dim))
                for _ in range(num_scales))
        else:
            self.local_encoding = [
                torch.zeros(1, 1, embed_dim)
                for _ in range(num_scales)]
            for i, b in enumerate(self.local_encoding):
                self.register_buffer(f"local_encoding{i}", b)

        if norm_layer == "LN":
            norm_layer = nn.LayerNorm
        else:
            raise ValueError(f"Unknown norm_layer: {norm_layer}")

        from privi.jepa.src.models.utils.modules import Block
        self.blocks = nn.ModuleList(
            [
                Block(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    norm_layer=norm_layer,
                    drop=proj_drop,
                    attn_drop=attn_drop,
                )
                for _ in range(depth)
            ]
        )

        self.use_ca = use_ca
        if self.use_ca:
            self.ca_block = CrossAttentionBlock(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                norm_layer=norm_layer,)
        

        self.classifier = nn.Linear(embed_dim, num_classes)

        self.init_std = init_std
        trunc_normal_(self.query_tokens, std=self.init_std)
        trunc_normal_(self.global_context_encoding, std=self.init_std)
        trunc_normal_(self.global_roi_encoding, std=self.init_std)
        for e in self.local_encoding:
            trunc_normal_(e, std=self.init_std)
        self.apply(self._init_weights)
        self._rescale_blocks()

        if query_init_path is not None:
            print(f"Loading query tokens from {query_init_path}")
            query_tokens = torch.load(query_init_path).unsqueeze(0)  # Add batch dimension
            assert query_tokens.shape == self.query_tokens.shape, f"Expected query tokens shape {self.query_tokens.shape}, got {query_tokens.shape}"
            self.query_tokens.data.copy_(query_tokens)

        if input_projection == "gaussian_random":
            assert len(self.proj_local) == 1
            if not isinstance(self.proj_global, nn.Identity):
                self.proj_global.weight.data.normal_()
                self.proj_global.weight.data /= math.sqrt(embed_dim)
                self.proj_global.bias.data.zero_()
                self.proj_global.weight.requires_grad = False
                self.proj_global.bias.requires_grad = False
            self.proj_local[0].weight.data.normal_()
            self.proj_local[0].weight.data /= math.sqrt(embed_dim)
            self.proj_local[0].bias.data.zero_()
            self.proj_local[0].weight.requires_grad = False
            self.proj_local[0].bias.requires_grad = False

    def _rescale_blocks(self):
        def rescale(param, layer_id):
            param.div_(math.sqrt(2.0 * layer_id))

        if self.blocks is not None:
            for layer_id, layer in enumerate(self.blocks, 1):
                rescale(layer.attn.proj.weight.data, layer_id + 1)
                rescale(layer.mlp.fc2.weight.data, layer_id + 1)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=self.init_std)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=self.init_std)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def step(self, step, total_steps):

        self.drop_scales.update_drop_prob(step, total_steps, self.scale_drop_target)

    def rasterize_boxes_to_mask(self, boxes: torch.Tensor, S: int = 32) -> torch.BoolTensor:
        """
        
        Args:
            boxes: Tensor of shape (N,4) in relative coords [xmin, ymin, xmax, ymax]
            S: output mask size (default 32)

        Returns: BoolTensor of shape (N, S, S)
        """
        # boxes on device
        device = boxes.device
        N = boxes.shape[0]

        # 1) Precompute the cell edges in [0,1]
        edges = torch.linspace(0, 1, S + 1, device=device)         # shape (S+1,)
        cell_x1, cell_x2 = edges[:-1], edges[1:]                   # each shape (S,)
        cell_y1, cell_y2 = edges[:-1], edges[1:]                   # reuse for y

        # 2) Expand box coords to (N,1) so we can broadcast against (S,)
        x1 = boxes[:, 0].unsqueeze(1)  # (N,1)
        y1 = boxes[:, 1].unsqueeze(1)
        x2 = boxes[:, 2].unsqueeze(1)
        y2 = boxes[:, 3].unsqueeze(1)

        # 3) Compute per‐box × per‐column intersection widths: (N,S)
        inter_w = (torch.min(x2, cell_x2)
                - torch.max(x1, cell_x1)).clamp(min=0)         # (N,S)

        # 4) Compute per‐box × per‐row intersection heights: (N,S)
        inter_h = (torch.min(y2, cell_y2)
                - torch.max(y1, cell_y1)).clamp(min=0)         # (N,S)

        # 5) Outer‐product to get per‐box × per‐cell intersection area: (N,S,S)
        #    inter_w[:, :, None] is (N,S,1), inter_h[:, None, :] is (N,1,S)
        inter_area = inter_w[:, :, None] * inter_h[:, None, :]

        # 6) Threshold: cell_area = (1/S)*(1/S), so 50% is 0.5*(1/S^2)
        threshold = 0.5 * (1.0 / S) * (1.0 / S)

        # 7) Build boolean mask
        mask = inter_area > threshold      # (N,S,S), dtype=torch.bool

        return mask

    def forward(self, x_local, x_global, bbox, *args, **kwargs):
        """"

        This classifier cannot deal with multiple crops per image, so crops must be 1.

        Args:
            x_local list(torch.Tensor): Local features of shape (B, crops, channels, time, height, width)
            x_global list(torch.Tensor): Global features of shape (B, crops, channels, time, height, width),
            bbox (torch.Tensor): Bounding boxes of shape (B, crops, 4)
        """
        
        B = x_local[0].shape[0]

        q = self.query_tokens.expand(B, -1, -1)

        if self.use_ca:
            x = [] # CA to queries last
        else:
            x = [q] # SA with all queries

        x_local = self.drop_scales(x_local)

        for idx, xl in enumerate(x_local):
            xl = rearrange(xl, "B 1 C T H W -> B C T H W")
            if self.downsample_local is not None:
                xl = nn.functional.avg_pool3d(xl, kernel_size=self.downsample_local)
                if not self.printed_local_shape:
                    print(f"Downsampled local features to shape {xl.shape}")
                    self.printed_local_shape = True
            xl = rearrange(xl, "B C T H W -> B (T H W) C")  # Flatten time, height, width
            xl = self.in_norm_local[idx](xl)
            xl = self.proj_local[idx](xl)
            if self.use_local_encoding:
                xl = xl + self.local_encoding[idx]
            x.append(xl)
        
        if x_global:
            _, crops, C, T, H, W = x_global[0].shape
            assert H == W, "Global features must be square (H == W)"

            assert len(x_global) == 1, "cannot deal with multiple layers as input yet"
            x_global = x_global[0]
            x_global = rearrange(x_global, "B 1 C T H W -> B C T H W")

            mask = rearrange(self.rasterize_boxes_to_mask(
                rearrange(bbox, "B 1 D -> B D"), S=H).long(), "B H W -> B 1 1 H W")
            encoding = mask * self.global_roi_encoding + (1 - mask) * self.global_context_encoding # (B, C, 1, H, W)
            x_global = rearrange(x_global, "B C T H W -> B (T H W) C")
            x_global = self.proj_global(x_global)  # Project to embed_dim
            x_global = rearrange(x_global, "B (T H W) C -> B C T H W", T=T, H=H, W=W)  # Reshape back to (B, C, T, H, W)
            x_global = x_global + encoding
            
            x_global = rearrange(x_global, "B C T H W -> B (T H W) C")  # Flatten time, height, width

            x.append(x_global)
        
        x = torch.cat(x, dim=1)

        for block in self.blocks:
            x = block(x, mask=None)
            x = torch.nn.functional.relu(x)

        if self.use_ca:
            x = self.ca_block(q, x)
            x = torch.nn.functional.relu(x)
        else:
            # Extract the query tokens for classification
            x = x[:, :q.shape[1], :]

        # x (B, num_classes, D), weights (num_classes, D)

        if self.query_token_mode == "per_class":
            # Each query token corresponds to a class

            # Apply linear classifier separately to each query token, i.e. each query token corresponds to a class
            logits = torch.einsum("bfk, fk -> bf", x, self.classifier.weight)
            assert logits.shape == (B, self.query_tokens.shape[1]), f"Expected logits shape (B, num_classes), got {logits.shape}"

            logits = logits + self.classifier.bias

        elif self.query_token_mode == "single":
            # Single query token for all classes
            logits = self.classifier(x[:, 0, :])
        else:
            raise ValueError(f"Unknown query_token_mode: {self.query_token_mode}")

        return logits.unsqueeze(1)  # Add crops dimension, shape (B, 1, num_classes)