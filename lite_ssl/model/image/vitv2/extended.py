from functools import partial

import torch
import torch.nn as nn

from lite_ssl.config import logger
from lite_ssl.model.image.vitv2.transformer import ViTv2
from lite_ssl.layers import (
    DINOHead,
    IdentityHead,
    NestedTensorBlock as Block,
    MemEffAttention,
)


class ProjViTv2(ViTv2):
    def __init__(
        self,
        *args,
        # ADDED PARAMETERS
        cva_head_proj=None,
        project_latent=False,
        project_patch=False,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        if project_patch:  # for ddp (no patch proj-no mask token needed ddp angy)
            self.norm_patch = nn.LayerNorm(
                self.embed_dim, eps=1e-6
            )  # a different norm layer for the patch embedding
            self.mask_token = nn.Parameter(torch.empty(1, self.embed_dim))
        else:
            self.mask_token = None

        # Construct projection head
        logger.info(f"{(cva_head_proj is not None)=}, {project_latent=}")
        if (cva_head_proj is not None) and project_latent:
            self.cva_module_proj = DINOHead(in_dim=self.embed_dim, **cva_head_proj)
        else:
            self.cva_module_proj = IdentityHead()

        # Initialize the model's weights
        self.init_weights()


def proj_vitv2_tiny(patch_size=16, num_register_tokens=0, **kwargs):
    if "init_values" not in kwargs:
        kwargs["init_values"] = 0.1
    model = ProjViTv2(
        patch_size=patch_size,
        embed_dim=192,
        depth=12,
        num_heads=3,
        mlp_ratio=4,
        block_fn=partial(Block, attn_class=MemEffAttention),
        num_register_tokens=num_register_tokens,
        **kwargs,
    )
    return model


def proj_vitv2_small(patch_size=16, num_register_tokens=0, **kwargs):
    if "init_values" not in kwargs:
        kwargs["init_values"] = 0.1
    model = ProjViTv2(
        patch_size=patch_size,
        embed_dim=384,
        depth=12,
        num_heads=6,
        mlp_ratio=4,
        block_fn=partial(Block, attn_class=MemEffAttention),
        num_register_tokens=num_register_tokens,
        **kwargs,
    )
    return model


def proj_vitv2_base(patch_size=16, num_register_tokens=0, **kwargs):
    if "init_values" not in kwargs:
        kwargs["init_values"] = 0.1
    model = ProjViTv2(
        patch_size=patch_size,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        block_fn=partial(Block, attn_class=MemEffAttention),
        num_register_tokens=num_register_tokens,
        **kwargs,
    )
    return model


def proj_vitv2_large(patch_size=16, num_register_tokens=0, **kwargs):
    if "init_values" not in kwargs:
        kwargs["init_values"] = 1e-5
    model = ProjViTv2(
        patch_size=patch_size,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4,
        block_fn=partial(Block, attn_class=MemEffAttention),
        num_register_tokens=num_register_tokens,
        **kwargs,
    )
    return model


if __name__ == "__main__":
    _model = proj_vitv2_tiny(
        patch_size=16,
        num_register_tokens=4,
    ).cuda()
    _out = _model(torch.randn(2, 3, 224, 224, device="cuda"), last_self_attention=True)

    print({k: v.shape if v is not None else v for k, v in _out.items()})

    _l = _model.oc_encoder_global(_out["latent"].unflatten(0, (2, -1)))
    print(_l.shape)
