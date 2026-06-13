from typing import Union, Sequence, Any

import torch
import lightning as pl
import torchmetrics

from loguru import logger
from lightning import Callback
from ml_collections import ConfigDict

from lite_ssl.loss import get_loss
from lite_ssl.metric import get_metric
from lite_ssl.config import MODELS_DIR
from lite_ssl.util import STORE, MODEL_TYPE
from lite_ssl.optim import init_optims_from_config
from lite_ssl.scheduling import Schedule, Scheduler


def apply_lr_multiplier(loc, step, sched):
    return loc.get("lr_multiplier", 1.0) * sched(step)


def apply_wd_multiplier(loc, step, sched):
    return loc.get("wd_multiplier", 1.0) * sched(step)


class OnlyImageAugmentationWrapper(torch.nn.Module):
    """
    Wrapper for the augmentation module to only apply the augmentation to the image.
    """

    def __init__(self, aug_module):
        super().__init__()
        self.aug_module = aug_module

    def forward(self, *batch):
        x, *y = batch
        x = self.aug_module(x)
        return x, *y


class MyIdentity(torch.nn.Module):
    """
    A module that does nothing.
    """

    def forward(self, *args, **kwargs):
        if len(args) == 0:
            return args[0]
        else:
            return args


class MySequential(torch.nn.Sequential):
    """
    A sequential module that allows for the use of a custom forward method.
    """

    def forward(self, *args, **kwargs):
        for module in self:
            args = module(*args, **kwargs)
        return args


class BaseTrainer(pl.LightningModule):
    def __init__(
        self, config: ConfigDict, normalisation: torch.nn.Module, probe_valid_dl, probe_train_dl
    ):
        super().__init__()

        # initialise the config
        self.config = config

        if hasattr(self.config, "num_classes"):
            self.num_classes: int = self.config.num_classes
        else:
            self.num_classes = 10

        # initialise the module
        self.normalisation = normalisation

        self.criterion = self.make_criterion()

        metrics = self.make_metrics()
        metrics = torchmetrics.MetricCollection(metrics)

        if (
            self.config.dataset.name in ["in100", "in", "in21k"]
            or "imagenet" in self.config.dataset.name
        ):
            # don't use the metrics for training for ImageNet
            # since that makes it slower x2 as 140gb of training data metrics will be saved
            self.train_metrics = torchmetrics.MetricCollection({}).clone(prefix="train/")
        else:
            self.train_metrics = metrics.clone(prefix="train/")

        self.val_metrics = metrics.clone(prefix="val/")
        self.test_metrics = metrics.clone(prefix="test/")

        self.probe_val_dl: torch.utils.data.DataLoader = probe_valid_dl
        self.probe_train_dl: torch.utils.data.DataLoader = probe_train_dl

    def make_model(self):
        """
        Create the model.
        :return:
        """

        model = STORE.get(MODEL_TYPE, self.config.model.type)(**self.config.model.params)

        # check if finetune is set and find the model
        if hasattr(self.config, "finetune") and self.config.finetune.enable:
            # load the model from the checkpoint
            checkpoint = torch.load(MODELS_DIR / self.config.finetune.state_dict_path)
            logger.info(f"Loading model from {self.config.finetune.state_dict_path}")
            if "state_dict" in checkpoint:
                checkpoint = checkpoint["state_dict"]
            logger.success("Loaded model from checkpoint")

            # replace key with fc to classifier
            keys_to_del = []
            keys_to_replace = []

            for key in checkpoint.keys():
                if key.startswith("fc"):
                    keys_to_replace.append(key)

            for key in keys_to_replace:
                value = checkpoint[key]
                if (
                    self.config.model.params.num_classes
                    and value.shape[0] != self.config.model.params.num_classes
                ):
                    checkpoint[key.replace("fc", "throwaway")] = value
                    keys_to_del.append(key)

            logger.warning(f"Replacing keys: {keys_to_replace}")
            logger.warning(f"Deleting keys: {keys_to_del}")

            for key in keys_to_del:
                del checkpoint[key]

            missing_keys, unexpected_keys = model.load_state_dict(checkpoint, strict=False)
            # assert that missing keys all have "fc" in their name if any missing keys
            assert (
                all("fc" in key for key in missing_keys) or not missing_keys
            ), "Missing keys should be related to the classifier, " "but found: {}".format(
                missing_keys
            )

            # assert that all unexpected keys are "throwaway"
            assert (
                all("throwaway" in key for key in unexpected_keys) or not unexpected_keys
            ), "Unexpected keys should be related to the classifier, " "but found: {}".format(
                unexpected_keys
            )

            if len(missing_keys) > 0 or len(unexpected_keys) > 0:
                logger.warning(f"Missing keys: {missing_keys}")
                logger.warning(f"Unexpected keys: {unexpected_keys}")

            if hasattr(self.config.finetune, "frozen") and self.config.finetune.frozen:
                for name, param in model.named_parameters():
                    if "fc" not in name:
                        param.requires_grad = False
                    else:
                        param.requires_grad = True

        return model

    def make_opt_sched(
        self, config, trainable_modules
    ) -> tuple[list[torch.optim.Optimizer], Scheduler]:
        """
        Create the optimisers.
        :return:
        """
        opt = init_optims_from_config(config, trainable_modules)

        scheduler = Scheduler()
        for key, sched in config.scheduler:

            if key == "lr":
                for group_num in range(len(opt.param_groups)):
                    scheduler.add(
                        opt.param_groups[group_num],
                        key,
                        Schedule.parse(sched),
                        apply_lr_multiplier,
                    )

            if key == "weight_decay":
                for group_num in range(len(opt.param_groups)):
                    scheduler.add(
                        opt.param_groups[group_num],
                        key,
                        Schedule.parse(sched),
                        apply_wd_multiplier,
                    )

        return [opt], scheduler

    def make_criterion(self):
        """
        Create the loss function.
        :return:
        """
        return get_loss(self.config.loss.type)(**self.config.loss.params)

    def make_metrics(self):
        """
        Create the metrics.
        :return:
        """
        try:
            metrics = {
                short_name: get_metric(metric_dict.type)(**metric_dict.params)
                for short_name, metric_dict in self.config.metrics.items()
            }
        except Exception:
            metrics = {}

        return metrics

    def batch_to_loss(self, batch, train=False):
        x, y = batch

        y_hat = self.model(self.normalisation(x))["logits"]  # type: ignore
        loss = self.criterion(y_hat, y)
        return loss, y, y_hat

    def log_loss(self, loss, prefix, prog_bar, on_epoch, on_step):
        # loss can be a dict
        if isinstance(loss, dict):
            for key, value in loss.items():
                self.log(
                    f"{prefix}/{key}",
                    value.detach(),
                    prog_bar=prog_bar,
                    on_epoch=on_epoch,
                    on_step=on_step,
                    sync_dist=on_epoch,
                )
            return loss["loss"]
        else:
            self.log(
                f"{prefix}/loss",
                loss,
                prog_bar=prog_bar,
                on_epoch=on_epoch,
                on_step=on_step,
                sync_dist=on_epoch,
            )
            return loss

    def log_metrics(self, batch_metrics):
        metric_dict = {
            k: torch.mean(v) if len(v.shape) > 0 else v
            for k, v in batch_metrics.items()
            if not torch.all(torch.isnan(v))
        }
        self.log_dict(metric_dict, prog_bar=True, on_epoch=False, on_step=True)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        loss, y, y_hat = self.batch_to_loss(batch, train=True)

        loss = self.log_loss(loss, prefix="train", prog_bar=True, on_epoch=False, on_step=True)

        with torch.no_grad():
            try:
                batch_metrics = self.train_metrics(y_hat, y)
                self.log_metrics(batch_metrics)
            except Exception as e:
                logger.error(f"Error computing training metrics {y_hat.shape=}, {y.shape=}: {e}")

        if isinstance(loss, dict):
            return loss["loss"]  # we only want to backpropagate the main loss
        else:
            return loss

    def on_train_epoch_end(self):
        self.train_metrics.reset()

    def validation_step(self, batch, batch_idx):
        loss, y, y_hat = self.batch_to_loss(batch, train=False)

        loss = self.log_loss(loss, prefix="val", prog_bar=True, on_epoch=True, on_step=False)
        try:
            self.val_metrics.update(y_hat, y)
        except Exception as e:
            logger.error(f"Error computing validation metrics: {e}")

        return loss

    def on_validation_epoch_end(self):
        self.log_dict(
            self.val_metrics.compute(), prog_bar=True, on_epoch=True, on_step=False, sync_dist=True
        )
        self.val_metrics.reset()

    def test_step(self, batch, batch_idx):
        loss, y, y_hat = self.batch_to_loss(batch, train=False)

        loss = self.log_loss(loss, prefix="test", prog_bar=True, on_epoch=True, on_step=False)

        try:
            self.test_metrics.update(y_hat, y)
        except Exception as e:
            logger.error(f"Error computing test metrics: {e}")

        return loss

    def on_test_epoch_end(self) -> None:
        self.log_dict(
            self.test_metrics.compute(),
            prog_bar=True,
            on_epoch=True,
            on_step=False,
            sync_dist=True,
        )
        self.test_metrics.reset()

    def configure_optimizers(self):
        return self.optims, []

    def configure_callbacks(self) -> Union[Sequence[Callback], Callback]:
        """
        Override this method to configure callbacks for the trainer.
        """

        callbacks: list[Any] = [self.scheduler]

        return callbacks
