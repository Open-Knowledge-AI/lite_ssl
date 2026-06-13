from pathlib import Path

from torch.utils.data import Dataset
from torchvision.datasets.folder import default_loader


class NICO(Dataset):
    base_folder = "NICO_DG"

    contexts = ["grass", "rock", "water"]

    def __init__(
        self,
        root,
        split=None,
        transform=None,
        target_transform=None,
        download=False,
        **kwargs,
    ):
        self.root = Path(root)

        self.context, self.object = split

        assert (
            self.context in self.contexts
        ), f"Context '{self.context}' not supported. Choose from {self.contexts}"

        context_dir = self.root / self.base_folder / self.context

        if not context_dir.exists():
            raise RuntimeError(f"Context directory not found: {context_dir}")

        if not self.object in (objects := list(map(lambda _p: _p.name, context_dir.iterdir()))):
            raise RuntimeError(
                f"Object '{self.object}' not found in context '{self.context}': {objects}"
            )

        self.transform = transform
        self.target_transform = target_transform

        self.data_dir = self.root / self.base_folder / self.context / self.object

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
        body.append(f"Split: {self.context, self.object}")
        if self.transform is not None:
            body.append(f"Transform: {self.transform}")
        lines = [head] + [" " * 4 + line for line in body]
        return "\n".join(lines)


if __name__ == "__main__":
    from lite_ssl.config import PROCESSED_DATA_DIR

    dataset = NICO(root=PROCESSED_DATA_DIR, split=("grass", "wolf"))
    print(dataset)
