"""Vision Transformer model implementation."""

# Imports Python packages.
from transformers import ViTForImageClassification, ViTModel

# Imports milkshake packages.
from milkshake.models.model import Model


class ViT(Model):
    """Vision Transformer model implementation."""

    def __init__(self, args):
        """Initializes a Vision Transformer model.

        Args:
            args: The configuration dictionary.
        """

        super().__init__(args)

        vit_types = {
            "classifier": ViTForImageClassification,
            "feature_extractor": ViTModel,
        }

        weights = f"google/vit-{args.vit_version}-patch16-224"
        if args.vit_pretrained == "imagenet21k":
            weights += "-in21k"

        self.model = vit_types[args.vit_type].from_pretrained(
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
        if self.hparams.vit_pretrained == "imagenet21k":
            display_pretrained = "ImageNet21K"
        else:
            display_pretrained = "ImageNet1K"
        display_type = self.hparams.vit_type.replace("_", " ")
        display_version = self.hparams.vit_version.capitalize()

        return (
            f"Loading {display_pretrained}-pretrained Vision"
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

        if self.hparams.vit_type == "classifier":
            return outputs.logits
        else:
            return outputs.last_hidden_state
