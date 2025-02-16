"""DataModule for the SVHN dataset."""

# Imports Python builtins.
import os
import os.path as osp

# Imports Python packages.
import numpy as np
import scipy.io as sio

# Imports PyTorch packages.
import torch
from torchvision.datasets import SVHN as TorchVisionSVHN
from torchvision.datasets.utils import download_url
from torchvision.transforms import Compose, Normalize, RandomCrop, Resize

# Imports milkshake packages.
from milkshake.datamodules.datamodule import DataModule
from milkshake.datamodules.dataset import Dataset

# Normalization values for SVHN
mean = [0.4377, 0.4438, 0.4728]
std = [0.1980, 0.2010, 0.1970]


class SVHNDataset(Dataset, TorchVisionSVHN):
    """Dataset for the SVHN dataset."""

    def __init__(self, *xargs, **kwargs):
        Dataset.__init__(self, *xargs, **kwargs)

    def download(self):
        os.makedirs(osp.join(self.root, "svhn"), exist_ok=True)

        download_url(
            self.split_list["train"][0],
            osp.join(self.root, "svhn"),
            self.split_list["train"][1],
            self.split_list["train"][2],
        )
        download_url(
            self.split_list["test"][0],
            osp.join(self.root, "svhn"),
            self.split_list["test"][1],
            self.split_list["test"][2],
        )

    def load_data(self):
        fname = self.split_list["train"][1] if self.train else self.split_list["test"][1]
        loaded_mat = sio.loadmat(os.path.join(self.root, "svhn", fname))
        self.data = torch.tensor(loaded_mat["X"]).float() / 255
        self.targets = torch.tensor(loaded_mat["y"]).long().squeeze()

        # The SVHN dataset assigns the class label "10" to the digit 0.
        # This makes it inconsistent with several loss functions
        # which expect the class labels to be in the range [0, C-1].
        self.targets[self.targets == 10] = 0
        self.data = torch.permute(self.data, (3, 2, 0, 1)) # CHW format

class SVHN(DataModule):
    """DataModule for the MNIST dataset."""

    def __init__(self, args, **kwargs):
        super().__init__(args, SVHNDataset, 10, 0, **kwargs)

        if args.image_size != 32:
            print(
                "Warning: image size is not the standard 32x32 for SVHN."
                " Please adjust args.image_size if this was not your intention."
            )

    def augmented_transforms(self):
        return Compose([
            RandomCrop(self.image_size, padding=4),
            Normalize(mean, std),
        ])

    def default_transforms(self):
        return Compose([
            Resize(self.image_size),
            Normalize(mean, std)
        ])
