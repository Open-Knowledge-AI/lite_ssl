import torchvision.transforms as T


class SimpleAug:

    def __init__(self):
        self.aug = T.Compose(
            [
                T.RandomApply([T.ColorJitter(0.4, 0.4, 0.2, 0.1)], p=0.8),
                T.RandomGrayscale(p=0.2),
                T.RandomApply([T.GaussianBlur(kernel_size=9, sigma=(0.1, 2.0))]),
                T.RandomSolarize(threshold=128, p=0.2),
            ]
        )

    def __call__(self, img):
        return self.aug(img)
