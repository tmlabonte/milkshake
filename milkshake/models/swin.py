"""Swin Transformer model implementation."""

# Imports Python packages.
from transformers import SwinForImageClassification, SwinModel

# Imports milkshake packages.
from milkshake.models.model import Model


class Swin(Model):
    """Swin Transformer model implementation."""

    def __init__(self, args):
        """Initializes a Swin Transformer model.

        Args:
            args: The configuration dictionary.
        """

        super().__init__(args)

        if (args.swin_pretrained == "imagenet21k"
                and args.swin_version in ("tiny", "small")):
            raise ValueError((
                "ImageNet21K weights are only available"
                " for Swin Transformer Base and Large."
            ))

        swin_types = {
            "classifier": SwinForImageClassification,
            "feature_extractor": SwinModel,
        }

        weights = f"microsoft/swin-{args.swin_version}-patch4-window7-224"
        if args.swin_pretrained == "imagenet21k":
            weights += "-in22k"

        self.model = swin_types[args.swin_type].from_pretrained(
            weights,
            ignore_mismatched_sizes=True,
            num_labels=args.num_classes,
        )

        # Freezes all parameters except those in the last layer.
        if args.train_fc_only:
            for p in self.model.parameters():
                p.requires_grad = False
            for p in self.model.classifier.parameters():
                p.requires_grad = True

    def load_msg(self):
        if self.hparams.swin_pretrained == "imagenet21k":
            display_pretrained = "ImageNet21K"
        else:
            display_pretrained = "ImageNet1K"
        display_type = self.hparams.swin_type.replace("_", " ")
        display_version = self.hparams.swin_version.capitalize()

        return (
            f"Loading {display_pretrained}-pretrained Swin"
            f" Transformer {display_version} {display_type}."
        )

    def forward(self, inputs):
        """Predicts using the model.

        Args:
            inputs: A torch.Tensor of model inputs.

        Returns:
            The model prediction as a torch.Tensor.
        """

        outputs = self.model(inputs)

        if self.hparams.swin_type == "classifier":
            return outputs.logits
        else:
            return outputs.last_hidden_state
