import os
import os.path

import numpy as np
import torch.utils.data as data

from PIL import Image
from loguru import logger
from torchvision.datasets.utils import download_url, extract_archive, check_integrity
from torchvision.datasets import CIFAR10, CIFAR100, VisionDataset, STL10 as STL10Base, Imagenette


class C10(CIFAR10):
    mean = (0.4915, 0.4823, 0.4468)
    std = (0.2470, 0.2435, 0.2616)
    num_classes = 10
    image_size = 32


def load_new_test_data(root, version="default"):
    data_path = root
    filename = "cifar10.1"
    if version == "default":
        filename += "-v6"
    elif version == "v0":
        filename += "-v0"
    else:
        raise ValueError('Unknown dataset version "{}".'.format(version))
    label_filename = filename + "-labels.npy"
    imagedata_filename = filename + "-data.npy"
    label_filepath = os.path.join(data_path, label_filename)
    imagedata_filepath = os.path.join(data_path, imagedata_filename)
    labels = np.load(label_filepath).astype(np.int64)
    imagedata = np.load(imagedata_filepath)
    assert len(labels.shape) == 1
    assert len(imagedata.shape) == 4
    assert labels.shape[0] == imagedata.shape[0]
    assert imagedata.shape[1] == 32
    assert imagedata.shape[2] == 32
    assert imagedata.shape[3] == 3
    if version == "default":
        assert labels.shape[0] == 2000
    elif version == "v0":
        assert labels.shape[0] == 2021
    return imagedata, labels


class C10G(data.Dataset):
    mean = (0.4915, 0.4823, 0.4468)
    std = (0.2470, 0.2435, 0.2616)
    num_classes = 10
    image_size = 32

    images_url = "https://github.com/modestyachts/CIFAR-10.1/blob/master/datasets/cifar10.1_v6_data.npy?raw=true"
    images_filename = "cifar10.1-v6-data.npy"

    labels_url = "https://github.com/modestyachts/CIFAR-10.1/blob/master/datasets/cifar10.1_v6_labels.npy?raw=true"
    labels_filename = "cifar10.1-v6-labels.npy"

    classes = [
        "airplane",
        "automobile",
        "bird",
        "cat",
        "deer",
        "dog",
        "frog",
        "horse",
        "ship",
        "truck",
    ]

    @property
    def targets(self):
        return self.labels

    def __init__(
        self, root, *args, transform=None, target_transform=None, download=True, **kwargs
    ):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform

        # delete the args and kwargs that are not used
        if args:
            logger.warning(f"Arguments {args} are not used in {self.__class__.__name__}.")
        if kwargs:
            logger.warning(
                f"Keyword arguments {kwargs} are not used in {self.__class__.__name__}."
            )

        # check if the dataset is already downloaded
        if not check_integrity(
            os.path.join(self.root, self.images_filename)
        ) or not check_integrity(os.path.join(self.root, self.labels_filename)):
            logger.info(f"{self.__class__.__name__} not found in {self.root}, downloading...")

            if download:
                self.download()
            else:
                raise RuntimeError(
                    f"Dataset not found in {self.root}. "
                    "You can use `download=True` to download it."
                )
        else:
            logger.info(f"{self.__class__.__name__} already exists in {self.root}")

        images, labels = load_new_test_data(root)

        self.data = images
        self.labels = labels

        self.class_to_idx = {_class: i for i, _class in enumerate(self.classes)}

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.labels[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.data)

    def download(self):
        root = self.root
        download_url(self.images_url, root, self.images_filename)
        download_url(self.labels_url, root, self.labels_filename)

    def __repr__(self):
        fmt_str = "Dataset " + self.__class__.__name__ + "\n"
        fmt_str += "    Number of datapoints: {}\n".format(self.__len__())
        fmt_str += "    Root Location: {}\n".format(self.root)
        tmp = "    Transforms (if any): "
        fmt_str += "{0}{1}\n".format(
            tmp, self.transform.__repr__().replace("\n", "\n" + " " * len(tmp))
        )
        tmp = "    Target Transforms (if any): "
        fmt_str += "{0}{1}".format(
            tmp, self.target_transform.__repr__().replace("\n", "\n" + " " * len(tmp))
        )
        return fmt_str


class C100(CIFAR100):
    mean = (0.5071, 0.4865, 0.4409)
    std = (0.2673, 0.2564, 0.2762)
    num_classes = 100
    image_size = 32


class C10C(VisionDataset):
    mean = C10.mean
    std = C10.std
    num_classes = C10.num_classes
    image_size = C10.image_size

    url = "https://zenodo.org/record/2535967/files/CIFAR-10-C.tar?download=1"
    filename = "CIFAR-10-C.tar"
    base_folder = "CIFAR-10-C"
    md5 = "56bf5dcef84df0e2308c6dcbcbbd8499"
    per_severity = 10000

    severities = [1, 2, 3, 4, 5]
    corruptions = [
        "gaussian_noise",
        "shot_noise",
        "impulse_noise",
        "speckle_noise",
        "defocus_blur",
        "glass_blur",
        "motion_blur",
        "zoom_blur",
        "gaussian_blur",
        "snow",
        "frost",
        "fog",
        "spatter",
        "brightness",
        "contrast",
        "saturate",
        "elastic_transform",
        "pixelate",
        "jpeg_compression",
    ]

    def __init__(
        self,
        root,
        download=False,
        severity=1,
        corruption="gaussian_noise",
        transform=None,
        target_transform=None,
    ):
        assert severity in self.severities
        assert corruption in self.corruptions

        super().__init__(root, transform=transform, target_transform=target_transform)
        self.slice = slice((severity - 1) * self.per_severity, severity * self.per_severity)

        if download:
            if (root / self.filename).exists():
                logger.info(f"{self.__class__.__name__} already downloaded")
            else:
                download_url(self.url, root, self.filename, self.md5)

        if not os.path.exists(os.path.join(root, self.base_folder)):
            logger.info(f"Extracting {self.__class__.__name__}")
            extract_archive(root / self.filename, root)

        # now load the picked numpy arrays
        images_file_path = os.path.join(self.root, self.base_folder, f"{corruption}.npy")
        self.data = np.load(images_file_path)[self.slice]
        labels_file_path = os.path.join(self.root, self.base_folder, f"labels.npy")
        self.targets = np.load(labels_file_path)[self.slice]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index: int):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target


class C100C(C10C):
    mean = C100.mean
    std = C100.std
    num_classes = C100.num_classes
    image_size = C100.image_size

    url = "https://zenodo.org/record/3555552/files/CIFAR-100-C.tar?download=1"
    filename = "CIFAR-100-C.tar"
    base_folder = "CIFAR-100-C"
    md5 = "11f0ed0f1191edbf9fa23466ae6021d3"
    per_severity = 10000

    severities = [1, 2, 3, 4, 5]
    corruptions = [
        "gaussian_noise",
        "shot_noise",
        "impulse_noise",
        "speckle_noise",
        "defocus_blur",
        "glass_blur",
        "motion_blur",
        "zoom_blur",
        "gaussian_blur",
        "snow",
        "frost",
        "fog",
        "spatter",
        "brightness",
        "contrast",
        "saturate",
        "elastic_transform",
        "pixelate",
        "jpeg_compression",
    ]


class C10CBar(VisionDataset):
    mean = C10.mean
    std = C10.std
    num_classes = C10.num_classes
    image_size = C10.image_size

    filename = "CIFAR-10-C-Bar.zip"
    base_folder = "CIFAR10-c-bar"
    per_severity = 10000

    severities = [1, 2, 3, 4, 5]
    corruptions = [
        "blue_noise",
        "brownish_noise",
        "checkerboard_cutout",
        "inverse_sparkles",
        "pinch_and_twirl",
        "ripple",
        "circular_motion_blur",
        "lines",
        "sparkles",
        "transverse_chromatic_abberation",
    ]

    def __init__(
        self,
        root,
        download=False,
        severity=1,
        corruption="lines",
        transform=None,
        target_transform=None,
    ):
        assert severity in self.severities
        assert corruption in self.corruptions

        super().__init__(root, transform=transform, target_transform=target_transform)
        self.slice = slice((severity - 1) * self.per_severity, severity * self.per_severity)

        if download:
            raise NotImplementedError("CIFAR C-Bar dataset(s) cannot be downloaded automatically")

        if not os.path.exists(os.path.join(root, self.base_folder)):
            logger.info(f"Extracting {self.__class__.__name__}")
            extract_archive(os.path.join(root, self.filename), root)

        # now load the picked numpy arrays
        images_file_path = os.path.join(self.root, self.base_folder, f"{corruption}.npy")
        self.data = np.load(images_file_path)[self.slice]
        labels_file_path = os.path.join(self.root, self.base_folder, f"labels.npy")
        self.targets = np.load(labels_file_path)[self.slice]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target


class C100CBar(C10CBar):
    mean = C100.mean
    std = C100.std
    num_classes = C100.num_classes
    image_size = C100.image_size

    filename = "CIFAR-100-C-Bar.zip"
    base_folder = "CIFAR100-c-bar"


class STL10(STL10Base):
    mean = C10.mean
    std = C10.std

    num_classes = 10
    image_size = 96

    def _check_integrity(self) -> bool:
        # if the folder exists, then is gucci
        if not (self.root / self.base_folder).exists():
            # check if file name exists
            if not (self.root / self.filename).exists():
                return False
            else:
                extract_archive(self.root / self.filename, self.root)
                return True
        else:
            return True


class IM10(Imagenette):
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    num_classes = 10
    image_size = 160

    def __init__(
        self,
        root,
        split: str = "train",
        size: str = "160px",
        download=False,
        transform=None,
        target_transform=None,
    ):
        super().__init__(root, split, size, download, transform, target_transform)


if __name__ == "__main__":
    from lite_ssl.config import PROCESSED_DATA_DIR

    ds = IM10(PROCESSED_DATA_DIR, size="160px", download=True)
    print(len(ds._samples))

    # ds = C10(PROCESSED_DATA_DIR, download=True)
    # print(ds.data.shape)
    #
    # ds = C100(PROCESSED_DATA_DIR, download=True)
    # print(ds.data.shape)
    #
    # _ = C10C(PROCESSED_DATA_DIR, download=True, severity=1, corruption="gaussian_noise")
    # _ = C100C(PROCESSED_DATA_DIR, download=True, severity=1, corruption="gaussian_noise")
    # # _ = C10CBar(PROCESSED_DATA_DIR, download=True, extract_only=False, severity=1, corruption='blue_noise_sample')
    # # _ = C100CBar(PROCESSED_DATA_DIR, download=True, extract_only=False, severity=1, corruption='blue_noise_sample')
    #
    # ds = STL10(PROCESSED_DATA_DIR, download=True)
    # print(ds.data.shape)
    #
    # ds = IM10(PROCESSED_DATA_DIR, download=True)
    # print(len(ds._samples))
