"""DataModule for the BinaryMNIST dataset."""

# Imports PyTorch packages.
import torch
from torchvision.transforms import Compose, Normalize, Resize

# Imports milkshake packages.
from milkshake.datamodules.datamodule import DataModule
from milkshake.datamodules.mnist import MNISTDataset


class BinaryMNIST(DataModule):
    """DataModule for the BinaryMNIST dataset.

    The BinaryMNIST dataset uses the same data as the MNIST dataset, but
    turns it into a binary classification task between odd and even digits.
    This is different from the usual BinaryMNIST, which instead changes all
    pixel values to either 0 or 1 (we keep the original pixel values).
    """

    def __init__(self, args, **kwargs):
        super().__init__(args, MNISTDataset, 2, 0, **kwargs)
        
        if args.image_size != 28:
            print(
                "Warning: image size is not the standard 28x28 for MNIST."
                " Please adjust args.image_size if this was not your intention."
            )

    def augmented_transforms(self):
        return self.default_transforms()

    def default_transforms(self):
        return Compose([
            Resize(self.image_size),
            Normalize(mean=(0.5,), std=(0.5,)),
        ])

    def train_preprocess(self, dataset_train):
        dataset_train.targets = torch.tensor([target % 2 for target in dataset_train.targets])
        dataset_train = super().train_preprocess(dataset_train)
        return dataset_train

    def val_preprocess(self, dataset_val):
        dataset_val.targets = torch.tensor([target % 2 for target in dataset_val.targets])
        dataset_val = super().val_preprocess(dataset_val)
        return dataset_val

    def test_preprocess(self, dataset_test):
        dataset_test.targets = torch.tensor([target % 2 for target in dataset_test.targets])
        dataset_test = super().test_preprocess(dataset_test)
        return dataset_test
