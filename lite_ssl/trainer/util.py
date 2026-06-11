import math

import torch
import torch.nn as nn


def interpolate_pos_embed(pos_embed, orig_num_patches, new_num_patches, num_tokens=1):
    """
    Interpolate positional embeddings to match new number of patches.

    Args:
        pos_embed: Positional embedding tensor of shape [1, orig_num_patches + num_tokens, embed_dim]
        orig_num_patches: Original number of patches (H * W)
        new_num_patches: New number of patches (H' * W')
        num_tokens: Number of special tokens (CLS token, etc.)

    Returns:
        Interpolated positional embedding
    """
    if pos_embed.shape[1] == new_num_patches + num_tokens:
        return pos_embed

    # Separate tokens and patches
    extra_tokens = pos_embed[:, :num_tokens]  # CLS token, etc.
    pos_tokens = pos_embed[:, num_tokens:]  # Position tokens

    # Calculate original and new grid dimensions
    orig_grid_size = int(math.sqrt(orig_num_patches))
    new_grid_size = int(math.sqrt(new_num_patches))

    # Reshape to 2D grid [1, embed_dim, grid_size, grid_size]
    pos_tokens = pos_tokens.reshape(1, orig_grid_size, orig_grid_size, -1).permute(0, 3, 1, 2)

    # Interpolate using bicubic for better quality
    pos_tokens = nn.functional.interpolate(
        pos_tokens, size=(new_grid_size, new_grid_size), mode="bicubic", align_corners=False
    )

    # Reshape back to [1, new_num_patches, embed_dim]
    pos_tokens = pos_tokens.permute(0, 2, 3, 1).reshape(1, new_num_patches, -1)

    # Concatenate with extra tokens
    new_pos_embed = torch.cat([extra_tokens, pos_tokens], dim=1)

    return new_pos_embed


def get_orig_image_size_from_state_dict(state_dict, patch_size=16):
    """
    Try to infer original image size from state_dict by looking at pos_embed dimensions.

    Args:
        state_dict: Model state dictionary
        patch_size: Patch size used in the model

    Returns:
        Tuple of (image_size, num_patches) or None if cannot determine
    """
    pos_embed_keys = [k for k in state_dict.keys() if "pos_embed" in k]

    for key in pos_embed_keys:
        pos_embed = state_dict[key]
        if len(pos_embed.shape) == 3:  # [1, num_patches + num_tokens, embed_dim]
            num_patches_with_tokens = pos_embed.shape[1]
            # Assume 1 CLS token for standard ViT
            num_patches = num_patches_with_tokens - 1
            grid_size = int(math.sqrt(num_patches))
            if grid_size * grid_size == num_patches:
                image_size = grid_size * patch_size
                return image_size, num_patches

    return None, None


def load_and_interpolate_pos_embed(state_dict, model, pos_embed_keys=None):
    """
    Load and interpolate positional embeddings from state_dict to match model dimensions.

    Args:
        state_dict: Source state dictionary
        model: Target model
        pos_embed_keys: List of pos_embed keys to process. If None, auto-detect.

    Returns:
        Modified state_dict with interpolated positional embeddings
    """
    if pos_embed_keys is None:
        pos_embed_keys = [k for k in state_dict.keys() if "pos_embed" in k]

    if not pos_embed_keys:
        return state_dict

    model_patch_embed = model.patch_embed
    current_num_patches = model_patch_embed.num_patches

    for key in pos_embed_keys:
        if key not in state_dict:
            continue

        pos_embed = state_dict[key]

        # Determine original number of patches from the state_dict
        if hasattr(model, "num_tokens"):
            num_tokens = model.num_tokens
        else:
            # For standard ViT, assume 1 CLS token
            num_tokens = 1

        orig_num_patches_with_tokens = pos_embed.shape[1]
        orig_num_patches = orig_num_patches_with_tokens - num_tokens

        # Only interpolate if number of patches has changed
        if orig_num_patches != current_num_patches:
            print(f"Interpolating {key} from {orig_num_patches} to {current_num_patches} patches")
            interpolated_pos_embed = interpolate_pos_embed(
                pos_embed, orig_num_patches, current_num_patches, num_tokens
            )
            state_dict[key] = interpolated_pos_embed

    return state_dict


def filter_state_dict(state_dict, model, ignore_keys=None):
    """
    Filter state_dict to remove incompatible keys.

    Args:
        state_dict: Source state dictionary
        model: Target model
        ignore_keys: List of keys to ignore

    Returns:
        Filtered state_dict
    """
    if ignore_keys is None:
        ignore_keys = []

    model_state_dict = model.state_dict()
    filtered_state_dict = {}

    for k, v in state_dict.items():
        if k in ignore_keys:
            continue

        if k in model_state_dict:
            if v.shape == model_state_dict[k].shape:
                filtered_state_dict[k] = v
            else:
                print(f"Shape mismatch for {k}: {v.shape} vs {model_state_dict[k].shape}")
        else:
            print(f"Key {k} not found in model")

    return filtered_state_dict
