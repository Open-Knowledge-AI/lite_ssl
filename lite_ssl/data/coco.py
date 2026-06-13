from pathlib import Path

from torch.utils.data import Dataset
from torchvision.datasets.folder import default_loader


class CocoUnlabelled(Dataset):
    base_folder = "COCO"

    train_dir = "train2017"
    val_dir = "val2017"
    unlabelled_dir = "unlabeled2017"
    test_dir = "test2017"
    annotations_dir = "annotations"

    def __init__(
        self,
        root,
        split="unlabeled",
        transform=None,
        target_transform=None,
        download=False,
        **kwargs,
    ):
        self.root = Path(root)

        self.split = split
        self.transform = transform
        self.target_transform = target_transform

        # Map split to directory name
        split_to_dir = {
            "train": self.train_dir,
            "val": self.val_dir,
            "unlabeled": self.unlabelled_dir,
            "test": self.test_dir,
        }

        if split not in split_to_dir:
            raise ValueError(
                f"Split '{split}' not supported. Choose from {list(split_to_dir.keys())}"
            )

        self.data_dir = self.root / self.base_folder / split_to_dir[split]

        # Check if directory exists
        if not self.data_dir.exists():
            raise RuntimeError(f"Dataset directory not found: {self.data_dir}")

        # Get all image files from the directory
        self.image_files = []
        valid_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}

        for filename in self.data_dir.iterdir():
            file_ext = filename.suffix
            if file_ext in valid_extensions:
                self.image_files.append(filename)

        # Sort for consistent ordering
        self.image_files.sort()

        # Loader function - default_loader loads as RGB by default
        self.loader = default_loader

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is the image file name
        """
        img_name = self.image_files[index]
        img_path = self.data_dir / img_name

        # Load image (default_loader loads as RGB)
        image = self.loader(img_path)

        # For unlabelled data, we return -1 as target for compatibility
        target = -1

        if self.transform is not None:
            image = self.transform(image)

        return image, target

    def __len__(self):
        return len(self.image_files)

    def __repr__(self):
        head = "Dataset " + self.__class__.__name__
        body = [f"Number of datapoints: {len(self)}"]
        if self.root is not None:
            body.append(f"Root location: {self.root}")
        body.append(f"Split: {self.split}")
        if self.transform is not None:
            body.append(f"Transform: {self.transform}")
        lines = [head] + [" " * 4 + line for line in body]
        return "\n".join(lines)
