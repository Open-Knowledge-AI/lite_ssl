from pathlib import Path
from torchvision.datasets import ImageFolder


class ImageNet(ImageFolder):
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    num_classes = 1000
    image_size = 224

    train_filename = "imagenet-train.tar.gz"
    test_filename = "imagenet-val.tar.gz"

    base_folder = "ILSVRC2012"
    train_folder = "train"
    test_folder = "val"

    def __init__(self, root, train=True, transform=None, target_transform=None, download=False):
        root = Path(root)

        # extract the dataset if it does not exist yet from the zip file in the root sub_dir
        if train:
            sub_dir = self.train_folder
        else:
            sub_dir = self.test_folder

        root = root / self.base_folder / sub_dir

        super().__init__(root, transform, target_transform)


class ImageNet100(ImageNet):
    num_classes = 100

    def __init__(self, root, train=True, transform=None, target_transform=None, download=False):
        super().__init__(root, train, transform, target_transform, download)

        _targets = [i for i in range(0, 1000, 10)]
        _indices = [i for i, label in enumerate(self.targets) if label % 10 == 0]

        # remap the labels
        self.targets = [_targets.index(label) for label in self.targets if label % 10 == 0]
        self.samples = [self.samples[i] for i in _indices]

        # remap the class_to_idx
        self.class_to_idx = {label: i for i, label in enumerate(_targets)}
        self.classes = [self.classes[i] for i in _targets]

        # remap the samples
        self.samples = [(path, self.class_to_idx[label]) for path, label in self.samples]

        # remap the targets
        self.targets = [label for _, label in self.samples]
