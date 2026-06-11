from torchvision import transforms as T


class GaussianBlur(T.RandomApply):
    """
    Apply Gaussian Blur to the PIL image.
    """

    def __init__(self, *, p: float = 0.5, radius_min: float = 0.1, radius_max: float = 2.0):
        # NOTE: torchvision is applying 1 - probability to return the original image
        keep_p = 1 - p
        transform = T.GaussianBlur(kernel_size=9, sigma=(radius_min, radius_max))
        super().__init__(transforms=[transform], p=keep_p)


class DINOAugmentation(object):
    def __init__(
        self,
        scales,
        crops_per_scale,
        global_crops_scale=(0.32, 1.0),
        local_crops_scale=(0.05, 0.32),
    ):
        self.scales = scales
        self.num_per_scale = crops_per_scale

        self.global_crops_size, self.local_crops_size = scales

        self.global_crops_scale = global_crops_scale
        self.local_crops_scale = local_crops_scale

        # random resized crop and flip
        self.geometric_augmentation_global = T.Compose(
            [
                T.RandomResizedCrop(
                    self.global_crops_size,
                    scale=self.global_crops_scale,
                    interpolation=T.InterpolationMode.BICUBIC,
                ),
                T.RandomHorizontalFlip(p=0.5),
            ]
        )

        self.geometric_augmentation_local = T.Compose(
            [
                T.RandomResizedCrop(
                    self.local_crops_size,
                    scale=self.local_crops_scale,
                    interpolation=T.InterpolationMode.BICUBIC,
                ),
                T.RandomHorizontalFlip(p=0.5),
            ]
        )

        # color distorsions / blurring
        color_jittering = T.Compose(
            [
                T.RandomApply(
                    [T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)],
                    p=0.8,
                ),
                T.RandomGrayscale(p=0.2),
            ]
        )

        global_transfo1_extra = GaussianBlur(p=1.0)

        global_transfo2_extra = T.Compose(
            [
                GaussianBlur(p=0.1),
                T.RandomSolarize(threshold=128, p=0.2),
            ]
        )

        local_transfo_extra = GaussianBlur(p=0.5)

        self.auggers = {
            "0_0": T.Compose(
                [
                    self.geometric_augmentation_global,
                    color_jittering,
                    global_transfo1_extra,
                    T.ToTensor(),
                ]
            ),
            "0_1": T.Compose(
                [
                    self.geometric_augmentation_global,
                    color_jittering,
                    global_transfo2_extra,
                    T.ToTensor(),
                ]
            ),
            "1": T.Compose(
                [
                    self.geometric_augmentation_local,
                    color_jittering,
                    local_transfo_extra,
                    T.ToTensor(),
                ]
            ),
        }

    def __call__(self, image):
        return [
            [
                (
                    self.auggers[f"{scale_idx}_{crop_idx % 2}"](image)
                    if scale_idx == 0
                    else self.auggers[f"{scale_idx}"](image)
                )
                for crop_idx in range(self.num_per_scale[scale_idx])
            ]
            for scale_idx in range(len(self.scales))
        ]


if __name__ == "__main__":
    from PIL import Image

    import matplotlib.pyplot as plt

    from lite_ssl.config import SAMPLE_DATA_DIR

    to_pil = T.ToPILImage()

    _sample_images = [
        "n02134418_sloth_bear.JPEG",
        "n02120079_Arctic_fox.JPEG",
        "n01537544_indigo_bunting.JPEG",
        "n01592084_chickadee.JPEG",
    ]

    augger = DINOAugmentation(
        scales=[224, 96],
        crops_per_scale=[2, 4],
    )

    for fname in _sample_images:
        _orig_img = Image.open(SAMPLE_DATA_DIR / "imagenet-sample-images" / fname).convert("RGB")

        _ll_imgs = augger(_orig_img)

        _, axs = plt.subplots(
            len(_ll_imgs),
            max(len(scale_imgs) for scale_imgs in _ll_imgs),
            figsize=(max(len(scale_imgs) for scale_imgs in _ll_imgs) * 3, len(_ll_imgs) * 3),
        )
        if axs.ndim == 1:
            axs = axs[None, :]
        for i, scale_imgs in enumerate(_ll_imgs):
            for j, img in enumerate(scale_imgs):
                axs[i, j].imshow(to_pil(img))
                axs[i, j].axis("off")

        # turn off axis not being used
        for i in range(len(_ll_imgs)):
            for j in range(len(_ll_imgs[i]), max(len(scale_imgs) for scale_imgs in _ll_imgs)):
                axs[i, j].axis("off")

        plt.tight_layout()
        plt.show()
