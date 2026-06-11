from collections import defaultdict

from loguru import logger

import torch


def get_vit_lr_decay_rate(
    name, lr_decay_rate=1.0, num_layers=12, force_is_backbone=False, chunked_blocks=False
):
    """
    Calculate lr decay rate for different ViT blocks.
    Args:
        name (string): parameter name.
        lr_decay_rate (float): base lr decay rate.
        num_layers (int): number of ViT blocks.
    Returns:
        lr decay rate for the given parameter.
    """
    layer_id = num_layers + 1
    if name.startswith("backbone") or force_is_backbone:
        if (
            ".pos_embed" in name
            or ".patch_embed" in name
            or ".mask_token" in name
            or ".cls_token" in name
            or ".register_tokens" in name
        ):
            layer_id = 0
        elif force_is_backbone and (
            "pos_embed" in name
            or "patch_embed" in name
            or "mask_token" in name
            or "cls_token" in name
            or "register_tokens" in name
        ):
            layer_id = 0
        elif ".blocks." in name and ".residual." not in name:
            layer_id = int(name[name.find(".blocks.") :].split(".")[2]) + 1
        elif chunked_blocks and "blocks." in name and "residual." not in name:
            layer_id = int(name[name.find("blocks.") :].split(".")[2]) + 1
        elif "blocks." in name and "residual." not in name:
            layer_id = int(name[name.find("blocks.") :].split(".")[1]) + 1

    return lr_decay_rate ** (num_layers + 1 - layer_id)


def get_params_groups_with_decay(
    model, custom_keys_weight_decay, lr_decay_rate=1.0, patch_embed_lr_mult=1.0
):
    chunked_blocks = False
    if hasattr(model, "n_blocks"):
        logger.info("chunked fsdp")
        n_blocks = model.n_blocks
        chunked_blocks = model.chunked_blocks
    elif hasattr(model, "blocks"):
        logger.info("first code branch")
        n_blocks = len(model.blocks)
    elif hasattr(model, "backbone"):
        logger.info("second code branch")
        n_blocks = len(model.backbone.blocks)
    else:
        logger.info("else code branch")
        n_blocks = 0

    all_param_groups = []

    for name, param in model.named_parameters():
        name = name.replace("_fsdp_wrapped_module.", "")
        if not param.requires_grad:
            continue
        decay_rate = get_vit_lr_decay_rate(
            name,
            lr_decay_rate,
            num_layers=n_blocks,
            force_is_backbone=n_blocks > 0,
            chunked_blocks=chunked_blocks,
        )
        d = {
            "params": param,
            "is_last_layer": False,
            "is_cva_module_cls": False,
            "is_cva_module_proj": False,
            "is_online_probe": False,
            "lr_multiplier": decay_rate,
            "wd_multiplier": 1.0,
            "name": name,
        }

        if "last_layer" in name:
            d.update({"is_last_layer": True})

        if (
            name.endswith(".bias")
            or "norm" in name
            or "gamma" in name
            or any([name in custom for custom in custom_keys_weight_decay])
        ):
            d.update({"wd_multiplier": 0.0})

        if "patch_embed" in name:
            d.update({"lr_multiplier": d["lr_multiplier"] * patch_embed_lr_mult})

        if "cva_module_cls" in name:
            d.update({"is_cva_module_cls": True})

        if "cva_module_proj" in name:
            d.update({"is_cva_module_proj": True})

        if "online_probe" in name:
            d.update({"is_online_probe": True})

        all_param_groups.append(d)
        logger.info(
            f"""{name}: lr_multiplier: {d["lr_multiplier"]}, wd_multiplier: {d["wd_multiplier"]}"""
        )

    return all_param_groups


def fuse_params_groups(
    all_params_groups,
    keys=(
        "lr_multiplier",
        "wd_multiplier",
        "is_last_layer",
        "is_cva_module_cls",
        "is_cva_module_proj",
        "is_online_probe",
    ),
):
    fused_params_groups = defaultdict(lambda: {"params": []})
    for d in all_params_groups:
        identifier = ""
        for k in keys:
            identifier += k + str(d[k]) + "_"

        for k in keys:
            fused_params_groups[identifier][k] = d[k]
        fused_params_groups[identifier]["params"].append(d["params"])

    return fused_params_groups.values()


def init_optims_from_config(config, model):
    if not isinstance(model, (tuple, list)):
        model = [model]

    custom_keys_weight_decay = [
        "class_token",
        "position_embedding",
        "relative_position_bias_table",
    ]

    for sub_model in model:
        if hasattr(sub_model, "custom_keys_weight_decay_filter"):
            custom_keys_weight_decay.extend(
                [key for key in sub_model.custom_keys_weight_decay_filter]
            )

        if hasattr(sub_model, "no_weight_decay"):
            custom_keys_weight_decay.extend(
                [key for key in sub_model.custom_keys_weight_decay_filter]
            )

    # === Add layerwise + patch_embed LR scaling ===
    param_groups = []
    for m in model:
        param_groups.extend(
            get_params_groups_with_decay(
                m,
                custom_keys_weight_decay,
                patch_embed_lr_mult=getattr(config.opt, "patch_embed_lr_mult", 1.0),
                lr_decay_rate=getattr(config.opt, "layerwise_decay", 1.0),
            )
        )
    param_groups = fuse_params_groups(param_groups)

    for g in param_groups:
        g["foreach"] = True

    # === Instantiate optimizer ===
    if hasattr(torch.optim, config.opt.type):
        opt = getattr(torch.optim, config.opt.type)(param_groups, **config.opt.params)
    else:
        raise NotImplementedError(f"Unknown optimizer: {config.opt.type}")

    return opt
