from functools import partial

import torchvision.transforms as T

from ml_collections import ConfigDict
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import DataLoader, Subset, default_collate

from lite_ssl.util import STORE, DATA_TYPE, AUG_TYPE
from lite_ssl.utils.masking import generate_masks, IBotMasker
from lite_ssl.config import logger, NUM_WORKERS, IN_TRAIN_DIR, IN_VAL_DIR


def collate_and_generate_masks(batch, patch_size, mask_prob):
    scales, y = default_collate(batch)

    gl = 0
    temp = scales
    while isinstance(temp, list) or isinstance(temp, tuple):
        gl = len(temp)
        temp = temp[0]

    resolution = temp.shape[-2:]
    mask_generator = IBotMasker(input_size=(res // patch_size for res in resolution))
    masks = generate_masks(mask_generator, len(batch) * gl, mask_prob=mask_prob)

    return scales, masks, y


def handle_subset(cfg, ds_name, train_dataset):
    if hasattr(cfg, "subset") and 0 < cfg.subset.pct < 100:
        splitter = StratifiedShuffleSplit(
            n_splits=1, test_size=cfg.subset.pct / 100, random_state=cfg.subset.subset_seed
        )
        indices = list(range(len(train_dataset)))

        labels = train_dataset.targets

        for _, test_indices in splitter.split(indices, labels):
            indices = test_indices
        train_dataset = Subset(train_dataset, indices)
    return train_dataset


def get_mean_std(cfg):
    return (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)


def get_standard_transforms(cfg):
    crop, resize = 128, 160

    val_transform = [
        T.Resize(resize),
        T.CenterCrop(crop),
    ]

    val_transform = T.Compose(val_transform + [T.ToTensor()])

    return val_transform


def make_loaders(cfg):
    ds_name = cfg.dataset.name

    if ds_name not in ["in", "in100"]:
        raise ValueError(f"Unknown dataset {ds_name}")

    ds = STORE.get(DATA_TYPE, ds_name)

    mean, std = get_mean_std(cfg)
    val_transform = get_standard_transforms(cfg)

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

    train_dataset = ds(IN_TRAIN_DIR, train=True, transform=train_transform)

    train_dataset = handle_subset(cfg, ds_name, train_dataset)
    logger.info(f"Train Dataset: {len(train_dataset)} samples")

    val_dataset = ds(IN_VAL_DIR, train=False, transform=val_transform)
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

    return (
        train_loader,
        val_loader,
        T.Normalize(mean=mean, std=std),
        val_transform,
    )
