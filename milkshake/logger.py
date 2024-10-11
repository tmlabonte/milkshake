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
from milkshake.utils import compute_accuracy

class Logger:
    """Handles logging for model training, validation, and testing."""

    def __init__(self, trainer):
        """Initializes the logger with a reference to the trainer."""
        self.trainer = trainer

    def log_helper(self, names, values, add_dataloader_idx=False):
        """Compresses calls to self.trainer.log.

        Args:
            names: A list of metric names to log.
            values: A list of metric values to log.
            add_dataloader_idx: Whether to include the dataloader index in the name.
        """
        for name, value in zip(names, values):
            self.trainer.log(
                name,
                value,
                on_step=(name in ("loss", "train_loss")),
                on_epoch=(name not in ("loss", "train_loss")),
                prog_bar=(name in ("train_acc", "val_loss", "val_acc")),
                sync_dist=True,
                add_dataloader_idx=add_dataloader_idx,
            )

    def log_helper2(self, names, values, dataloader_idx):
        """Calls log_helper as necessary for each DataLoader.

        Args:
            names: A list of metric names to log.
            values: A list of metric values to log.
            dataloader_idx: The index of the current DataLoader.
        """
        if dataloader_idx == 0:
            self.log_helper(names, values)

        try:
            self.log_helper(names, values, add_dataloader_idx=True)
        except Exception as e:
            pass

    def log_metrics(self, result, stage, dataloader_idx):
        """Logs metrics using the step results.

        Args:
            result: The output of self.step.
            stage: "train", "val", or "test".
            dataloader_idx: The index of the current dataloader.
        """
        names = []
        values = []
        for name in self.trainer.hparams.metrics:
            if name in result and "by_class" not in name and "by_group" not in name:
                names.append(f"{stage}_{name}")
                values.append(result[name])

        names.extend(["epoch", "step"])
        values.extend([float(self.trainer.current_epoch), float(self.trainer.global_step)])

        self.log_helper2(names, values, dataloader_idx)

    def collate_metrics(self, step_results, stage):
        """Collates and logs metrics by class and group.

        Args:
            step_results: List of dictionary results of self.validation_step or self.test_step.
            stage: "val" or "test".
        """
        if type(step_results[0]) == dict:
            step_results = [step_results]

        for step_result in step_results:
            dataloader_idx = step_result[0]["dataloader_idx"]

            def collate_and_sum(name):
                stacked = torch.stack([result[name] for result in step_result])
                return torch.sum(stacked, 0)

            if any(m in self.trainer.hparams.metrics for m in [
                "acc_by_class", "acc5_by_class", "acc_by_group", "acc5_by_group"]):
                names = []
                values = []
                total_by_class = collate_and_sum("total_by_class")
                total_by_group = collate_and_sum("total_by_group")

                if "acc_by_class" in self.trainer.hparams.metrics:
                    acc_by_class = collate_and_sum("correct_by_class") / total_by_class
                    names.extend([f"{stage}_acc_class{j:02d}" for j in range(len(acc_by_class))])
                    values.extend(list(acc_by_class))

                if "acc5_by_class" in self.trainer.hparams.metrics:
                    acc5_by_class = collate_and_sum("correct5_by_class") / total_by_class
                    names.extend([f"{stage}_acc5_class{j:02d}" for j in range(len(acc5_by_class))])
                    values.extend(list(acc5_by_class))

                if "acc_by_group" in self.trainer.hparams.metrics:
                    acc_by_group = collate_and_sum("correct_by_group") / total_by_group
                    names.extend([f"{stage}_acc_group{j:02d}" for j in range(len(acc_by_group))])
                    values.extend(list(acc_by_group))

                if "acc5_by_group" in self.trainer.hparams.metrics:
                    acc5_by_group = collate_and_sum("correct5_by_group") / total_by_group
                    names.extend([f"{stage}_acc5_group{j:02d}" for j in range(len(acc5_by_group))])
                    values.extend(list(acc5_by_group))

                self.log_helper2(names, values, dataloader_idx)

        def add_metrics_to_result(self, result, accs, dataloader_idx):
            """Adds dataloader_idx and metrics from compute_accuracy to result dict.

            Args:
                result: A dictionary containing the loss, prediction probabilities, and targets.
                accs: The output of compute_accuracy.
                dataloader_idx: The index of the current DataLoader.
            """

        result["dataloader_idx"] = dataloader_idx
        result["acc"] = accs["acc"]
        result["acc5"] = accs["acc5"]

        for k in ["class", "group"]:
            result[f"acc_by_{k}"] = accs[f"acc_by_{k}"]
            result[f"acc5_by_{k}"] = accs[f"acc5_by_{k}"]
            result[f"correct_by_{k}"] = accs[f"correct_by_{k}"]
            result[f"correct5_by_{k}"] = accs[f"correct5_by_{k}"]
            result[f"total_by_{k}"] = accs[f"total_by_{k}"]

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

        self.add_metrics_to_result(result, accs, dataloader_idx)

        self.log_metrics(result, stage, dataloader_idx)

        return result

"""
Any method that deals with logging can now simply call the respective Logger methods:

def training_step(self, batch, idx, dataloader_idx=0):
    result = self.step(batch, idx)
    accs = compute_accuracy(result["probs"], result["targets"], self.hparams.num_classes, self.hparams.num_groups)
    self.add_metrics_to_result(result, accs, dataloader_idx)
    self.log_metrics(result, "train", dataloader_idx)  # Delegate to logger
    return result
"""
