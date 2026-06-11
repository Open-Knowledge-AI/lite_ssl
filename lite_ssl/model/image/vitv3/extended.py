import torch
import torch.nn as nn


from lite_ssl.config import logger
from lite_ssl.model.image.vitv3.transformer import ViTv3
from lite_ssl.layers import DINOHead, IdentityHead


class ProjViTv3(ViTv3):
    def __init__(
        self,
        *,
        cva_head_proj=None,
        project_latent=False,
        project_patch=False,
        **kwargs,
    ):
        super().__init__(**kwargs)

        if project_patch:  # for ddp (no patch proj-no mask token needed ddp angy)
            self.mask_token = nn.Parameter(torch.empty(1, self.embed_dim))
        else:
            self.mask_token = None

        # Construct projection head
        logger.info(f"{(cva_head_proj is not None)=}, {project_latent=}")
        if (cva_head_proj is not None) and project_latent:
            self.cva_module_proj = DINOHead(in_dim=self.embed_dim, **cva_head_proj)
        else:
            self.cva_module_proj = IdentityHead()

        self.custom_keys_weight_decay_filter = [
            "pos_embed",
            "rope_embed",
            "cls_token",
            "register_tokens",
            "storage_tokens",
            "mask_token",
        ]

        self.init_weights()


def proj_vitv3_tiny(patch_size=16, n_storage_tokens=4, **kwargs):
    model = ProjViTv3(
        patch_size=patch_size,
        embed_dim=192,
        depth=12,
        num_heads=3,
        ffn_ratio=4,
        n_storage_tokens=n_storage_tokens,
        layerscale_init=0.1,
        **kwargs,
    )
    return model


def proj_vitv3_small(patch_size=16, n_storage_tokens=4, **kwargs):
    model = ProjViTv3(
        patch_size=patch_size,
        embed_dim=384,
        depth=12,
        num_heads=6,
        ffn_ratio=4,
        n_storage_tokens=n_storage_tokens,
        layerscale_init=0.1,
        **kwargs,
    )
    return model


def proj_vitv3_base(patch_size=16, n_storage_tokens=4, **kwargs):
    model = ProjViTv3(
        patch_size=patch_size,
        embed_dim=768,
        depth=12,
        num_heads=12,
        ffn_ratio=4,
        n_storage_tokens=n_storage_tokens,
        layerscale_init=0.1,
        **kwargs,
    )
    return model


def proj_vitv3_large(patch_size=16, n_storage_tokens=4, **kwargs):
    model = ProjViTv3(
        patch_size=patch_size,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        ffn_ratio=4,
        n_storage_tokens=n_storage_tokens,
        layerscale_init=1e-5,
        **kwargs,
    )
    return model
