from functools import partial

import numpy as np
import torchvision.transforms as T

from ml_collections import ConfigDict
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import DataLoader, Subset, default_collate

from lite_ssl.util import STORE, DATA_TYPE, AUG_TYPE
from lite_ssl.utils.masking import generate_masks, SeedletMaskingGeneratorCPU
from lite_ssl.config import logger, PROCESSED_DATA_DIR, NUM_WORKERS, IN_TRAIN_DIR, IN_VAL_DIR


class MultiTransform:
    def __init__(self, transform, num_times=2):
        """
        Initializes the MultiTransform class.
        Args:
            transform: A transformation function to be applied to give multiple views `num_times`.
            num_times: The number of times the transformation should be applied to the input.
        """
        self.num_times = num_times
        self.transform = transform

    def __call__(self, x):
        return [self.transform(x) for _ in range(self.num_times)]


class LabelNoiseTransform:
    def __init__(self, noise, num_classes, seed):
        self.noise = noise
        self.num_classes = num_classes
        self.random_state = np.random.RandomState(seed)

    def __call__(self, target):
        if self.random_state.rand() < self.noise:
            # Randomly select a different class
            target = self.random_state.choice(list(set(range(self.num_classes)) - {target}), 1)[0]
        return target


def collate_and_generate_masks(batch, patch_size, mask_prob):
    scales, y = default_collate(batch)

    gl = 0
    temp = scales
    while isinstance(temp, list) or isinstance(temp, tuple):
        gl = len(temp)
        temp = temp[0]

    resolution = temp.shape[-2:]
    mask_generator = SeedletMaskingGeneratorCPU(
        input_size=(res // patch_size for res in resolution)
    )
    masks = generate_masks(mask_generator, len(batch) * gl, mask_prob=mask_prob)

    return scales, masks, y


def make_probe_datasets(val_dataset):
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    indices = list(range(len(val_dataset)))
    labels = (
        val_dataset.targets
        if not hasattr(val_dataset, "_samples")
        else [label for _, label in val_dataset._samples]
    )
    train_indices, val_indices = next(splitter.split(indices, labels))
    ptrain_dataset = Subset(val_dataset, train_indices)
    pval_dataset = Subset(val_dataset, val_indices)

    return ptrain_dataset, pval_dataset


def handle_subset(cfg, ds_name, train_dataset):
    if hasattr(cfg, "subset") and 0 < cfg.subset.pct < 100:
        splitter = StratifiedShuffleSplit(
            n_splits=1, test_size=cfg.subset.pct / 100, random_state=cfg.subset.subset_seed
        )
        indices = list(range(len(train_dataset)))

        if hasattr(train_dataset, "targets") or ds_name == "im10":
            labels = (
                train_dataset.targets
                if ds_name != "im10"
                else [label for _, label in train_dataset._samples]
            )
        else:
            labels = [-1 for _ in indices]

        for _, test_indices in splitter.split(indices, labels):
            indices = test_indices
        train_dataset = Subset(train_dataset, indices)
    return train_dataset


def handle_label_noise(cfg, ds_name, train_dataset):
    if hasattr(cfg, "label_noise") and cfg.label_noise.rate > 0:
        label_noise_transform = LabelNoiseTransform(
            cfg.label_noise.rate, cfg.num_classes, cfg.label_noise.noise_seed
        )
        if ds_name == "im10":
            noised_samples = [
                (path, label_noise_transform(label)) for path, label in train_dataset._samples
            ]
            train_dataset._samples = noised_samples
        else:
            train_dataset.targets = [
                label_noise_transform(label) for label in train_dataset.targets
            ]


def get_mean_std(cfg):
    ds_name = cfg.dataset.name

    if hasattr(cfg, "finetune") and cfg.finetune.enable:
        mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
        if hasattr(cfg.finetune, "mean") and hasattr(cfg.finetune, "std"):
            mean = cfg.finetune.mean
            std = cfg.finetune.std
    elif ds_name in ["c10", "c100"]:
        mean, std = (0.49139968, 0.48215827, 0.44653124), (0.24703233, 0.24348505, 0.26158768)
    elif ds_name in ["m", "fm", "km"]:
        mean, std = (0.1307,), (0.3081,)
    elif ds_name in ["im10", "stl", "in100", "in", "in21k", "coco-un"]:
        mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
    else:
        raise ValueError(f"Unknown dataset {ds_name}")

    return mean, std


def get_standard_transforms(cfg):
    ds_name = cfg.dataset.name

    val_transform = []

    if hasattr(cfg, "finetune") and cfg.finetune.enable:
        crop, resize = cfg.finetune.crop, cfg.finetune.resize

        train_transform = [
            T.RandomResizedCrop(crop, scale=(0.75, 1.0), antialias=True),
            T.RandomHorizontalFlip(),
        ]
        val_transform = [
            T.Resize(resize),
            T.CenterCrop(crop),
        ]
    elif ds_name in ["c10", "c100"]:
        train_transform = [
            T.RandomHorizontalFlip(),
            T.RandomCrop(32, padding=4),
        ]
    elif ds_name in ["stl"]:
        train_transform = [
            T.RandomHorizontalFlip(),
            T.RandomResizedCrop(
                size=96,
                scale=(0.32, 1.0),
            ),
        ]

        val_transform = [
            T.Resize(96),
        ]
    elif ds_name in ["m", "fm", "km"]:
        train_transform = [
            T.RandomCrop(28, padding=4),
        ]
    elif ds_name in ["im10", "in100", "in", "in21k"]:
        if ds_name != "im10":
            crop, resize = 224, 256
        else:
            crop, resize = 128, 160

        train_transform = [
            T.RandomResizedCrop(
                crop,
                scale=(0.3, 1.0),
            ),
            T.RandomHorizontalFlip(),
        ]
        val_transform = [
            T.Resize(resize),
            T.CenterCrop(crop),
        ]
    elif ds_name in ["coco-un"]:
        train_transform = [
            T.RandomResizedCrop(672, scale=(0.2, 1.0)),
            T.RandomHorizontalFlip(),
        ]
        val_transform = [
            T.Resize(672),
        ]
    else:
        raise ValueError(f"Unknown dataset {ds_name}")

    if hasattr(cfg.dataset, "aug"):
        for aug in cfg.dataset.aug:
            aug = ConfigDict(aug, convert_dict=True)
            try:
                aug_cls = STORE.get(AUG_TYPE, aug.type)
                train_transform.append(aug_cls(**aug.params))
            except KeyError:
                if hasattr(T, aug.type):
                    train_transform.append(getattr(T, aug.type)(**aug.params))
                else:
                    raise ValueError(f"Unknown augmentation {aug.type}")

    train_transform = T.Compose(train_transform + [T.ToTensor()])
    val_transform = T.Compose(val_transform + [T.ToTensor()])

    return train_transform, val_transform


def make_loaders(cfg):
    ds_name = cfg.dataset.name

    ds = STORE.get(DATA_TYPE, ds_name)

    mean, std = get_mean_std(cfg)
    train_transform, val_transform = get_standard_transforms(cfg)

    if hasattr(cfg.dataset, "multi") and cfg.dataset.multi.enable:
        aug_info = ConfigDict(cfg.dataset.multi, convert_dict=True)
        train_transform = MultiTransform(train_transform, num_times=aug_info.num_times)

    if hasattr(cfg.dataset, "con") and cfg.dataset.con.enable:
        params = cfg.dataset.con.params if hasattr(cfg.dataset.con, "params") else {}
        con_info = ConfigDict(cfg.dataset.con, convert_dict=True)
        train_transform = STORE.get(AUG_TYPE, con_info.type)(**params)
        mask_prob = cfg.dataset.con.mask_prob if hasattr(cfg.dataset.con, "mask_prob") else 0.5
        patch_size = cfg.dataset.con.patch_size if hasattr(cfg.dataset.con, "patch_size") else 16
        collate_fn = (
            partial(collate_and_generate_masks, patch_size=patch_size, mask_prob=mask_prob)
            if hasattr(cfg.dataset.con, "do_mask") and cfg.dataset.con.do_mask
            else None
        )
    else:
        collate_fn = None

    if ds_name == "im10":
        train_dataset = ds(PROCESSED_DATA_DIR, split="train", transform=train_transform)
    elif ds_name in ["coco-un", "stl"]:
        if hasattr(cfg.dataset, "split"):
            split = cfg.dataset.split
        else:
            split = "unlabeled"
        train_dataset = ds(PROCESSED_DATA_DIR, split=split, transform=train_transform)
    elif ds_name == "in":
        train_dataset = ds(IN_TRAIN_DIR, train=True, transform=train_transform)
    else:
        train_dataset = ds(
            PROCESSED_DATA_DIR, train=True, download=True, transform=train_transform
        )

    handle_label_noise(cfg, ds_name, train_dataset)

    train_dataset = handle_subset(cfg, ds_name, train_dataset)
    logger.info(f"Train Dataset: {len(train_dataset)} samples")

    if ds_name == "im10":
        val_dataset = ds(PROCESSED_DATA_DIR, split="val", transform=val_transform)
    elif ds_name == "stl":
        val_dataset = ds(PROCESSED_DATA_DIR, split="test", transform=val_transform)
    elif ds_name == "coco-un":
        val_dataset = ds(PROCESSED_DATA_DIR, split="val", transform=val_transform)
    elif ds_name == "in":
        val_dataset = ds(IN_VAL_DIR, train=False, transform=val_transform)
    else:
        val_dataset = ds(PROCESSED_DATA_DIR, train=False, transform=val_transform)
    logger.info(f"Val Dataset: {len(val_dataset)} samples")

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=NUM_WORKERS,
        persistent_workers=True,
        drop_last=True,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=2,
        persistent_workers=True,
        collate_fn=None,
    )

    return (train_loader, val_loader, T.Normalize(mean=mean, std=std), val_transform, None, None)
