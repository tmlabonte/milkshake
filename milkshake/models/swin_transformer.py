"""ResNet model implementation."""

# Imports PyTorch packages.
import torch
from torch import nn
import torchvision.models as models

# Imports milkshake packages.
from milkshake.models.model import Model


class SwinTransformer(Model):
    """Swin Transformer model implementation."""

    def __init__(self, args):
        """Initializes a Swin Transformer model.

        Args:
            args: The configuration dictionary.
        """

        super().__init__(args)

        swin_transformers = {
            "tiny": models.swin_t,
            "small": models.swin_s,
            "base": models.swin_b,
        }

        weights = None
        if args.swin_transformer_pretrained:
            weights = "IMAGENET1K_V1"

        # TODO: Add more Swin Transformer options.
        self.model = swin_transformers[args.swin_transformer_version](
            dropout=args.dropout_prob,
            weights=weights,
        )

        self.model.head = nn.Linear(self.model.head.in_features, args.num_classes)

        # Freezes all parameters except those in the last layer.
        if args.train_fc_only:
            for p in self.model.parameters():
                p.requires_grad = False
            for p in self.model.head.parameters():
                p.requires_grad = True

    def load_msg(self):
        display_version = self.hparams.swin_transformer_version.capitalize()
        if self.hparams.resnet_pretrained:
            return f"Loading ImageNet1K-pretrained Swin Transformer {display_version}."
        else:
            return f"Loading Swin Transformer {self.hparams.display_version} with no pretraining."

