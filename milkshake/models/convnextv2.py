"""ConvNeXtV2 model implementation."""

# Imports Python builtins.
import types

# Imports Python packages.
from transformers import ConvNextV2Config, ConvNextV2ForImageClassification

# Imports milkshake packages.
from milkshake.models.model import Model


class ConvNeXtV2(Model):
    """ConvNeXtV2 model implementation."""

    def __init__(self, args):
        """Initializes a ConvNeXtV2 model.

        Args:
            args: The configuration dictionary.
        """

        super().__init__(args)

        if args.convnextv2_pretrained == "none":
            hidden_sizes = [args.convnextv2_initial_width * (2 ** j)
                            for j in range(0, 4)]
            depths = [
                args.convnextv2_stage124_depth,
                args.convnextv2_stage124_depth,
                args.convnextv2_stage3_depth,
                args.convnextv2_stage124_depth,
            ]
            config = ConvNextV2Config(
                hidden_sizes=hidden_sizes,
                depths=depths,
                num_labels=args.num_classes,
            )
            self.model = ConvNextV2ForImageClassification(config)
        else:
            if args.convnextv2_pretrained == "imagenet1k":
                suffix = "-1k-224"
            else:
                if args.convnextv2_version not in ["nano", "tiny", "base", "large"]:
                    raise ValueError((
                        "ImageNet22K pretraining is only available for"
                        " ConvNeXtV2 Nano, Tiny, Base, and Large."
                    ))
                suffix = "-22k-224"

            prefix = "facebook/convnextv2-"
            versions = ["atto", "femto", "pico", "nano",
                        "tiny", "base", "large", "huge"]
            models = {v: prefix + v + suffix for v in versions}

            self.model = ConvNextV2ForImageClassification.from_pretrained(
                models[args.convnextv2_version],
                num_labels=args.num_classes,
                ignore_mismatched_sizes=True,
            )

        # Modifies forward to return only logits.
        self.model.base_forward = self.model.forward
        def forward(self, x):
            return self.base_forward(x).logits
        self.model.forward = types.MethodType(forward, self.model)

        # Freezes all parameters except those in the last layer.
        if args.train_fc_only:
            for p in self.model.parameters():
                p.requires_grad = False
            for p in self.model.classifier.parameters():
                p.requires_grad = True

    def load_msg(self):
        if self.hparams.convnextv2_pretrained == "none":
            return (f"Loading ConvNeXtV2 with initial width"
                    f" {self.hparams.convnextv2_initial_width}, stage 124 depth"
                    f" {self.hparams.convnextv2_stage124_depth}, and stage 3"
                    f" depth {self.hparams.convnextv2_stage3_depth}.")
        elif self.hparams.convnextv2_pretrained == "imagenet1k":
            return (f"Loading ConvNeXtV2"
                    f" {self.hparams.convnextv2_version.capitalize()}"
                    f" pretrained on ImageNet1K.")
        else:
            return (f"Loading ConvNeXtV2"
                    f" {self.hparams.convnextv2_version.capitalize()}"
                    f" pretrained on ImageNet22K.")

