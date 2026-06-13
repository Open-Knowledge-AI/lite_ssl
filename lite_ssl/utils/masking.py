import random

import torch
import numpy as np


def complete_mask_randomly_np(mask, num_masking_patches, rng):
    flat = mask.reshape(-1)
    missing = num_masking_patches - flat.sum()

    if missing <= 0:
        return mask

    available = np.flatnonzero(~flat)
    chosen = rng.choice(available, size=missing, replace=False)
    flat[chosen] = True

    return mask


class SeedletMaskingGeneratorCPU:
    def __init__(
        self,
        input_size,
        num_masking_patches=None,
        min_num_patches=0,
        max_num_patches=None,
        min_aspect=0.3,
        max_aspect=3.33,
        max_tries=10,
    ):
        if isinstance(input_size, int):
            input_size = (input_size, input_size)

        self.h, self.w = input_size
        self.num_patches = self.h * self.w

        self.min_num_patches = min_num_patches
        self.num_masking_patches = num_masking_patches
        self.max_num_patches = max_num_patches or num_masking_patches

        self.log_min_aspect = np.log(min_aspect)
        self.log_max_aspect = np.log(max_aspect or 1 / min_aspect)

        self.max_tries = max_tries

    def __call__(self, num_masking_patches, starting_mask=None, rng=None):
        if rng is None:
            rng = np.random.default_rng()

        if starting_mask is None:
            mask = np.zeros((self.h, self.w), dtype=np.bool_)
        else:
            mask = starting_mask.copy()

        mask_count = mask.sum()

        while mask_count < num_masking_patches:
            max_mask = num_masking_patches - mask_count
            if self.max_num_patches is not None:
                max_mask = min(max_mask, self.max_num_patches)

            delta = self._mask(mask, max_mask, rng)
            if delta == 0:
                break

            mask_count += delta

        return complete_mask_randomly_np(mask, num_masking_patches, rng)

    def _mask(self, mask, max_mask_patches, rng):
        for _ in range(self.max_tries):
            target = rng.uniform(self.min_num_patches, max_mask_patches)
            aspect = np.exp(rng.uniform(self.log_min_aspect, self.log_max_aspect))

            h = int(round(np.sqrt(target * aspect)))
            w = int(round(np.sqrt(target / aspect)))

            if h <= 0 or w <= 0 or h >= self.h or w >= self.w:
                continue

            top = rng.integers(0, self.h - h + 1)
            left = rng.integers(0, self.w - w + 1)

            region = mask[top : top + h, left : left + w]
            newly = (~region).sum()

            if 0 < newly <= max_mask_patches:
                region[:] = True
                return newly

        return 0


def generate_masks(mask_generator, number_of_samples, mask_prob=0.5, per_sample_range=(0.1, 0.5)):
    num_masks = int(number_of_samples * mask_prob)
    num_tokens = mask_generator.num_patches
    prob_per_sample = np.linspace(*per_sample_range, num=num_masks)
    masks = [
        (
            mask_generator(num_masking_patches=int(prob_per_sample[i] * num_tokens))
            if i < num_masks
            else mask_generator(num_masking_patches=0)
        )
        for i in range(number_of_samples)
    ]
    random.shuffle(masks)
    masks = np.stack(masks, dtype=bool)
    masks = torch.from_numpy(masks).flatten(1, -1)

    return masks
