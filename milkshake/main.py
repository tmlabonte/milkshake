"""Main script for training, validation, and testing."""

# Imports Python builtins.
from inspect import isclass
import os
import os.path as osp
import resource

# Imports Python packages.
import wandb
from PIL import ImageFile

# Imports PyTorch packages.
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, TQDMProgressBar
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.utilities.seed import seed_everything
import torch

# Imports milkshake packages.
from milkshake.args import parse_args
from milkshake.imports import valid_models_and_datamodules

# Prevents PIL from throwing invalid error on large image files.
ImageFile.LOAD_TRUNCATED_IMAGES = True

# Prevents DataLoader memory error.
rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (8192, rlimit[1]))

# Silences wandb printouts.
os.environ["WANDB_SILENT"]="true"


def load_weights(args, model):
    """Loads model weights or resumes training checkpoint.

    Args:
        args: The configuration dictionary.
        model: A model which inherits from milkshake.models.Model.

    Returns:
        The input model, possibly with the state dict loaded.
    """

    args.ckpt_path = None
    if args.weights:
        if args.resume_training:
            # Resumes training state (weights, optimizer, epoch, etc.) from args.weights.
            args.ckpt_path = args.weights
            print(f"Resuming training state from {args.weights}.")
        else:
            # Loads just the weights from args.weights.
            checkpoint = torch.load(args.weights, map_location="cpu")
            state_dict = checkpoint["state_dict"]
            model.load_state_dict(state_dict, strict=False)
            print(f"Weights loaded from {args.weights}.")     

    return model

def load_trainer(args, addtl_callbacks=None):
    """Loads PL Trainer for training and validation.

    Args:
        args: The configuration dictionary.
        addtl_callbacks: Desired callbacks besides ModelCheckpoint and TQDMProgressBar.

    Returns:
        An instance of pytorch_lightning.Trainer parameterized by args.
        
    Raises:
        ValueError: addtl_callbacks is not None and not a list.
    """

    # Checkpoints model at the specified number of epochs.
    checkpointer1 = ModelCheckpoint(
        filename="{epoch:02d}-{val_loss:.3f}-{val_acc:.3f}",
        save_top_k=-1,
        every_n_epochs=args.ckpt_every_n_epoch,
    )

    # Checkpoints model with respect to validation loss.
    checkpointer2 = ModelCheckpoint(
        filename="best-{epoch:02d}-{val_loss:.3f}-{val_acc:.3f}",
        monitor="val_loss",
        every_n_epochs=args.ckpt_every_n_epoch,
    )

    progress_bar = TQDMProgressBar(refresh_rate=args.refresh_rate)

    # Sets DDP strategy for multi-GPU training.
    args.devices = int(args.devices)
    args.strategy = "ddp" if args.devices > 1 else None

    callbacks = [checkpointer1, checkpointer2, progress_bar]
    if addtl_callbacks is not None:
        if not isinstance(addtl_callbacks, list):
            raise ValueError("addtl_callbacks should be None or a list.")
        callbacks.extend(addtl_callbacks)

    logger = True # Activates TensorBoardLogger by default.
    if args.wandb:
        os.makedirs(args.wandb_dir, exist_ok=True)
        logger = WandbLogger(save_dir=args.wandb_dir, log_model="all")
        
    trainer = Trainer.from_argparse_args(
        args,
        callbacks=callbacks,
        logger=logger,
    )

    return trainer

def main(
    args,
    model_or_model_class,
    datamodule_or_datamodule_class,
    callbacks=None,
    model_hooks=None,
    verbose=True,
):
    """Main method for training and validation.

    Args:
        args: The configuration dictionary.
        model_or_model_class: A model or class which inherits from milkshake.models.Model.
        datamodule_or_datamodule_class: A datamodule or class which inherits from milkshake.datamodules.DataModule.
        callbacks: Any desired callbacks besides ModelCheckpoint and TQDMProgressBar.
        model_hooks: Any desired functions to run on the model before training.
        verbose: Whether to print the validation and test metrics.

    Returns:
        The trained model with its validation and test metrics.
    """

    os.makedirs(args.out_dir, exist_ok=True)

    # Sets global seed for reproducibility. Due to CUDA operations which can't
    # be made deterministic, the results may not be perfectly reproducible.
    seed_everything(seed=args.seed, workers=True)
    
    if isclass(datamodule_or_datamodule_class):
        datamodule = datamodule_or_datamodule_class(args)
    else:
        datamodule = datamodule_or_datamodule_class

    args.num_classes = datamodule.num_classes
    args.num_groups = datamodule.num_groups

    if isclass(model_or_model_class):
        model = model_or_model_class(args)
    else:
        model = model_or_model_class
        
    model = load_weights(args, model)

    if model_hooks:
        for hook in model_hooks:
            hook(model)

    trainer = load_trainer(args, addtl_callbacks=callbacks)
    if not args.eval_only:
        trainer.fit(model, datamodule=datamodule, ckpt_path=args.ckpt_path)

    val_metrics = trainer.validate(model, datamodule=datamodule, verbose=verbose)
    test_metrics = trainer.test(model, datamodule=datamodule, verbose=verbose)

    # Closes wanbd instance. Important for experiments which run main() many times.
    if args.wandb:
        wandb.finish()
    
    return model, val_metrics, test_metrics


if __name__ == "__main__":
    args = parse_args()
    
    models, datamodules = valid_models_and_datamodules()

    main(args, models[args.model], datamodules[args.datamodule])

