"""Dataset and DataModule for the FMOW dataset."""

# Imports Python builtins.
import os.path as osp

# Imports Python packages.
import numpy as np
import wilds

# Imports PyTorch packages.
from pl_bolts.transforms.dataset_normalizations import imagenet_normalization
from torchvision.transforms import (
    CenterCrop,
    Compose,
    RandomHorizontalFlip,
    RandomResizedCrop,
    Resize,
    ToTensor,
)

# Imports milkshake packages.
from milkshake.datamodules.dataset import Dataset
from milkshake.datamodules.datamodule import DataModule
from milkshake.utils import to_np


class FMOWDataset(Dataset):
    """Dataset for the FMOW dataset."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def download(self):
        pass

    def load_data(self):
        dataset = wilds.get_dataset(
            dataset="fmow",
            download=True,
            root_dir=self.root,
        )

        column_names = dataset.metadata_fields
        spurious_cols = column_names.index("region")
        spurious = to_np(dataset._metadata_array[:, spurious_cols])

        prefix = osp.join(self.root, "fmow_v1.1", "images")
        self.data = np.asarray([osp.join(prefix, f"rgb_img_{idx}.png")
                                for idx in dataset.full_idxs])
        self.targets = dataset.y_array

        # Spurious 5 represents "other" locations (unused).
        self.groups = [
            np.argwhere(spurious == 0).squeeze(),
            np.argwhere(spurious == 1).squeeze(),
            np.argwhere(spurious == 2).squeeze(),
            np.argwhere(spurious == 3).squeeze(),
            np.argwhere(spurious == 4).squeeze(),
        ]

        # Splits 1 and 2 are in-distribution val and test (unused).
        split = dataset._split_array
        self.train_indices = np.argwhere(split == 0).flatten()
        self.val_indices = np.argwhere(split == 3).flatten()
        self.test_indices = np.argwhere(split == 4).flatten()

        # Adds group indices into targets for metrics.
        targets = []
        for j, t in enumerate(self.targets):
            g = [k for k, group in enumerate(self.groups) if j in group][0]
            targets.append([t, g])
        self.targets = np.asarray(targets)

class FMOW(DataModule):
    """DataModule for the FMOW dataset."""

    def __init__(self, args, **kwargs):
        super().__init__(args, FMOWDataset, 62, 5, **kwargs)

    def augmented_transforms(self):
        transform = Compose([
            RandomResizedCrop(self.image_size, scale=(0.7, 1.0)),
            RandomHorizontalFlip(),
            ToTensor(),
            imagenet_normalization(),
        ])

        return transform

    def default_transforms(self):
        resize_ratio = 256 / 224

        transform = Compose([
            Resize(int(resize_ratio * self.image_size)),
            CenterCrop(self.image_size),
            ToTensor(),
            imagenet_normalization(),
        ])

        return transform
