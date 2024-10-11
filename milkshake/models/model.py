"""Parent class and training logic for a classification model."""

# Imports Python builtins.
from abc import abstractmethod

# Imports PyTorch packages.
from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch.nn.parameter import is_lazy
from torch.optim import Adam, AdamW, SGD
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, MultiStepLR

# Imports milkshake packages.
from milkshake.models.logger import Logger  
from milkshake.utils import compute_accuracy

# TODO: Implement after_n_steps logging similarly to after_epoch.


class Model(pl.LightningModule):
    """Parent class and training logic for a classification model.

    Attributes:
        self.hparams: The configuration dictionary.
        self.milkshake_logger: A milkshake.Logger.
        self.model: A torch.nn.Module.
        self.optimizer: A torch.optim optimizer.
    """

    def __init__(self, args):
        """Initializes a Model.

        Args:
            args: The configuration dictionary.
        """

        super().__init__()

        # Saves args into self.hparams.
        self.save_hyperparameters(args)
        print(self.load_msg())

        self.model = None

        optimizers = {"adam": Adam, "adamw": AdamW, "sgd": SGD}
        self.optimizer = optimizers[args.optimizer]

    def log_metrics(self, result, stage, dataloader_idx):
        """Logs metrics using the step results."""
        self.milkshake_logger.log_metrics(result, stage, dataloader_idx)

    def collate_metrics(self, step_results, stage):
        """Collates and logs metrics by class and group."""
        self.milkshake_logger.collate_metrics(step_results, stage)

    @abstractmethod
    def load_msg(self):
        """Returns a descriptive message about the Model configuration."""

    def has_uninitialized_params(self):
        """Returns whether the model has uninitialized parameters."""

        for param in self.model.parameters():
            if is_lazy(param):
                return True
        return False

    def setup(self, stage):
        """Initializes model parameters if necessary.

        Args:
            stage: "train", "val", or "test".
        """

        if self.has_uninitialized_params():
            if stage == "fit":
                dataloader = self.trainer.datamodule.train_dataloader()
            elif stage == "validate":
                dataloader = self.trainer.datamodule.val_dataloader()
            elif stage == "test":
                dataloader = self.trainer.datamodule.test_dataloader()
            else:
                dataloader = self.trainer.datamodule.predict_dataloader()
            dummy_batch = next(iter(dataloader))
            self.forward(dummy_batch[0])

        self.milkshake_logger = Logger(self.hparams, self.trainer, self.log)

    def forward(self, inputs):
        """Predicts using the model.

        Args:
            inputs: A torch.Tensor of model inputs.

        Returns:
            The model prediction as a torch.Tensor.
        """

        return torch.squeeze(self.model(inputs), dim=-1)

    def configure_optimizers(self):
        """Returns the optimizer and learning rate scheduler."""

        optimizer = self.optimizer(
            self.model.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )

        if isinstance(optimizer, SGD):
            optimizer.momentum = self.hparams.momentum

        if self.hparams.lr_scheduler == "cosine":
            scheduler = CosineAnnealingLR(
                optimizer,
                T_max=self.hparams.max_epochs,
            )
        elif self.hparams.lr_scheduler == "cosine_warmup":
            scheduler = LinearWarmupCosineAnnealingLR(
                optimizer,
                self.hparams.lr_warmup_epochs,
                self.hparams.max_epochs,
            )
        elif self.hparams.lr_scheduler == "linear":
            scheduler = LinearLR(
                optimizer,
                start_factor=1,
                end_factor=self.hparams.lr_drop,
                total_iters=self.hparams.max_epochs,
            )
        elif self.hparams.lr_scheduler == "step":
            scheduler = MultiStepLR(
                optimizer,
                self.hparams.lr_steps,
                gamma=self.hparams.lr_drop,
            )

        return [optimizer], [scheduler]

    def step(self, batch, idx):
        """Performs a single step of prediction and loss calculation.

        Args:
            batch: A tuple containing the inputs and targets as torch.Tensor.
            idx: The index of the given batch.

        Returns:
            A dictionary containing the loss, prediction probabilities, and targets.

        Raises:
            ValueError: Class weights are specified with MSE loss, or MSE loss
            is specified for a multiclass classification task.
        """

        inputs, orig_targets = batch

        # Removes extra targets (e.g., group index used for metrics).
        if orig_targets[0].ndim > 0:
            targets = orig_targets[:, 0]
        else:
            targets = orig_targets

        logits = self(inputs)

        # Ensures logits is a torch.Tensor.
        if isinstance(logits, (tuple, list)):
            logits = torch.squeeze(logits[0], dim=-1)

        # Initializes class weights if desired.
        weights = torch.ones(self.hparams.num_classes, device=logits.device)
        if self.hparams.class_weights:
            if self.hparams.loss == "mse":
                raise ValueError("Cannot use class weights with MSE.")
            weights = torch.tensor(self.hparams.class_weights, device=logits.device)

        # Computes loss and prediction probabilities.
        if self.hparams.loss == "cross_entropy":
            if self.hparams.num_classes == 1:
                loss = F.binary_cross_entropy_with_logits(logits, targets, weight=weights)
                probs = torch.sigmoid(logits)
            else:
                loss = F.cross_entropy(logits, targets, weight=weights,
                        label_smoothing=self.hparams.label_smoothing)
                probs = F.softmax(logits, dim=1)
        elif self.hparams.loss == "mse":
            if self.hparams.num_classes == 1:
                loss = F.mse_loss(logits, targets.float())
                probs = torch.sigmoid(logits)
            elif self.hparams.num_classes == 2:
                loss = F.mse_loss(logits[:, 0], targets.float())
                probs = F.softmax(logits, dim=1)
            else:
                raise ValueError("MSE is only an option for binary classification.")

        return {"loss": loss, "probs": probs, "targets": orig_targets}

    def step_and_log_metrics(self, batch, idx, dataloader_idx, stage):
        """Performs a step, then computes and logs metrics.

        Args:
            batch: A tuple containing the inputs and targets as torch.Tensor.
            idx: The index of the given batch.
            dataloader_idx: The index of the current dataloader.
            stage: "train", "val", or "test".

        Returns:
            A dictionary containing the loss, prediction probabilities, targets, and metrics.
        """

        result = self.step(batch, idx)

        accs = compute_accuracy(
            result["probs"],
            result["targets"],
            self.hparams.num_classes,
            self.hparams.num_groups,
        )

        self.milkshake_logger.add_metrics_to_result(result, accs, dataloader_idx)

        self.milkshake_logger.log_metrics(result, stage, dataloader_idx)

        return result

    def training_step(self, batch, idx, dataloader_idx=0):
        """Performs a single training step.

        Args:
            batch: A tuple containing the inputs and targets as torch.Tensor.
            idx: The index of the given batch.
            dataloader_idx: The index of the current dataloader.

        Returns:
            A dictionary containing the loss, prediction probabilities, targets, and metrics.
        """

        return self.step_and_log_metrics(batch, idx, dataloader_idx, "train")

    def training_epoch_end(self, training_step_outputs):
        """Collates metrics upon completion of the training epoch.

        Args:
            training_step_outputs: List of dictionary outputs of self.training_step.
        """

        self.collate_metrics(training_step_outputs, "train")

    def validation_step(self, batch, idx, dataloader_idx=0):
        """Performs a single validation step.

        Args:
            batch: A tuple containing the inputs and targets as torch.Tensor.
            idx: The index of the given batch.
            dataloader_idx: The index of the current dataloader.

        Returns:
            A dictionary containing the loss, prediction probabilities, targets, and metrics.
        """

        return self.step_and_log_metrics(batch, idx, dataloader_idx, "val")

    def validation_epoch_end(self, validation_step_outputs):
        """Collates metrics upon completion of the validation epoch.

        Args:
            validation_step_outputs: List of dictionary outputs of self.validation_step.
        """

        self.collate_metrics(validation_step_outputs, "val")

    def test_step(self, batch, idx, dataloader_idx=0):
        """Performs a single test step.

        Args:
            batch: A tuple containing the inputs and targets as torch.Tensor.
            idx: The index of the given batch.
            dataloader_idx: The index of the current dataloader.

        Returns:
            A dictionary containing the loss, prediction probabilities, targets, and metrics.
        """

        return self.step_and_log_metrics(batch, idx, dataloader_idx, "test")

    def test_epoch_end(self, test_step_outputs):
        """Collates metrics upon completion of the test epoch.

        Args:
            test_step_outputs: List of dictionary outputs of self.test_step.
        """

        self.collate_metrics(test_step_outputs, "test")

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        """Performs a single prediction step.

        Args:
            batch: A tuple containing the inputs and targets as torch.Tensor.
            idx: The index of the given batch.
            dataloader_idx: The index of the current dataloader.
        """

        logits = self(batch)

        if self.hparams.num_classes == 1:
            probs = torch.sigmoid(logits)
            preds = (probs >= 0.5).int()
        else:
            probs = F.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)

        return preds


