"""Dataset and DataModule for the CelebA dataset."""

# Imports Python builtins.
import os.path as osp

# Imports Python packages.
import numpy as np
import pandas as pd
import wget

# Imports PyTorch packages.
from pl_bolts.transforms.dataset_normalizations import imagenet_normalization
from torchvision.datasets.utils import (
    download_file_from_google_drive,
    extract_archive,
)
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


class CelebADataset(Dataset):
    """Dataset for the CelebA dataset."""

    def __init__(self, *xargs, **kwargs):
        super().__init__(*xargs, **kwargs)

    def download(self):
        celeba_dir = osp.join(self.root, "celeba")
        if not osp.isdir(celeba_dir):
            # This function can fail if the Google Drive link has received many
            # recent requests. One may have to download and unzip it manually.
            download_file_from_google_drive(
                "0B7EVK8r0v71pZjFTYXZWM3FlRnM",
                celeba_dir,
                filename="celeba.zip",
            )
            extract_archive(osp.join(celeba_dir, "celeba.zip"))

            url = (
                "https://raw.githubusercontent.com/PolinaKirichenko/"
                "deep_feature_reweighting/main/celeba_metadata.csv"
            )
            wget.download(url, out=celeba_dir)

    def load_data(self):
        celeba_dir = osp.join(self.root, "celeba")
        metadata_df = pd.read_csv(osp.join(celeba_dir, "celeba_metadata.csv"))
        self.data = np.asarray(metadata_df["img_filename"].values)
        self.data = np.asarray([osp.join(celeba_dir, d) for d in self.data])

        self.targets = np.asarray(metadata_df["y"].values)
        gender = np.asarray(metadata_df["place"].values)
        nonblond = np.argwhere(self.targets == 0).flatten()
        blond = np.argwhere(self.targets == 1).flatten()
        women = np.argwhere(gender == 0).flatten()
        men = np.argwhere(gender == 1).flatten()

        self.groups = [
            np.intersect1d(nonblond, women),
            np.intersect1d(nonblond, men),
            np.intersect1d(blond, women),
            np.intersect1d(blond, men),
        ]

        split = np.asarray(metadata_df["split"].values)
        self.train_indices = np.argwhere(split == 0).flatten()
        self.val_indices = np.argwhere(split == 1).flatten()
        self.test_indices = np.argwhere(split == 2).flatten()

        # Adds group indices into targets for metrics.
        targets = []
        for j, t in enumerate(self.targets):
            g = [k for k, group in enumerate(self.groups) if j in group][0]
            targets.append([t, g])
        self.targets = np.asarray(targets)

class CelebA(DataModule):
    """DataModule for the CelebA dataset."""

    def __init__(self, args, **kwargs):
        super().__init__(args, CelebADataset, 2, 4, **kwargs)

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
