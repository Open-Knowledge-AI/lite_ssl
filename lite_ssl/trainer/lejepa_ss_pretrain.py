from pathlib import Path
from typing import Union, Sequence

import torch
import torch.nn.functional as F

from lightning import Callback
from ml_collections import ConfigDict

from lite_ssl.loss.custom import SIGReg
from lite_ssl.trainer.base import BaseTrainer
from lite_ssl.optim import init_optims_from_config
from lite_ssl.scheduling import Schedule, Scheduler
from lite_ssl.config import logger, MODELS_DIR, WANDB_PROJECT
from lite_ssl.trainer.util import load_and_interpolate_pos_embed, filter_state_dict
from lite_ssl.model.image.vitv2 import (
    proj_vitv2_tiny,
    proj_vitv2_small,
    proj_vitv2_base,
    proj_vitv2_large,
)
from lite_ssl.model.image.vitv3 import (
    proj_vitv3_tiny,
    proj_vitv3_small,
    proj_vitv3_base,
    proj_vitv3_large,
)
from lite_ssl.model.image.resnet.extended import proj_resnet_18, proj_resnet_34, proj_resnet_50


class OnlineProbe(torch.nn.Module):

    def __init__(self, embed_dim, num_classes, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.online_probe = torch.nn.Sequential(
            torch.nn.LayerNorm(embed_dim), torch.nn.Linear(embed_dim, num_classes)
        )

    def forward(self, emb):
        return self.online_probe(emb)


def apply_lr_multiplier(loc, step, sched):
    return loc.get("lr_multiplier", 1.0) * sched(step)


def apply_wd_multiplier(loc, step, sched):
    return loc.get("wd_multiplier", 1.0) * sched(step)


class LeJepaSSLoss(torch.nn.Module):

    def __init__(
        self,
        *ignored_args,
        **ignored_kwargs,
    ):
        """
        Initializes the LossWrapper.

        Args:
            ignored_kwargs: consistency loss specific kwargs
        """
        super().__init__()
        logger.warning(f"Ignored Kwargs: {ignored_kwargs}")

        self.sig_reg = SIGReg(num_slices=4096)
        self.probe_loss = torch.nn.CrossEntropyLoss()

        self.sigreg_lambda_ = 0.05

    def forward(
        self, global_step, batch_size, crop_cls, latent, probe_logits, y_gt, *args, **kwargs
    ):
        normalised_mcws = self.compute_mcws(latent)

        # [batch size, dimensions]
        centers = crop_cls.view(-1, batch_size, crop_cls.shape[-1]).mean(0)

        # [-1, batch size, dimensions]
        chunked_cls = crop_cls.view(-1, batch_size, crop_cls.shape[-1])
        sim_cls = (centers - chunked_cls).square().mean()
        sig_reg_loss_cls = torch.mean(
            torch.stack(
                [self.sig_reg(chunk, global_step) for chunk in chunked_cls],
                dim=0,
            )
        )

        acc, ce = self.compute_online_probe_loss(probe_logits, y_gt)

        return (
            {
                "loss": (
                    (1 - self.sigreg_lambda_) * sim_cls
                    + self.sigreg_lambda_ * sig_reg_loss_cls
                    + ce
                ),
                "inv_cls": sim_cls,
                "sigreg": sig_reg_loss_cls,
                "probe/ce": ce,
                "probe/acc": acc,
                "mcws": normalised_mcws,
            },
            y_gt,
            probe_logits,
        )

    def compute_online_probe_loss(self, probe_logits, y_gt):
        ce = self.probe_loss(probe_logits, y_gt)
        with torch.no_grad():
            acc = probe_logits.argmax(dim=-1).eq(y_gt).sum().float().div_(y_gt.shape[0])
        return acc, ce

    @classmethod
    def compute_mcws(cls, latent):
        with torch.no_grad():
            mean_channel_wise_std = torch.std(F.normalize(latent, dim=-1), dim=0).mean()
            normalised_mcws = mean_channel_wise_std / (1 / latent.shape[1] ** 0.5)
        return normalised_mcws


class LeJepaSSTrainer(BaseTrainer):

    def __init__(
        self, config: ConfigDict, normalisation: torch.nn.Module, probe_valid_dl, probe_train_dl
    ):
        super().__init__(config, normalisation, probe_valid_dl, probe_train_dl)

        self.model, self.online_probe = self.make_model()
        self.criterion = self.make_criterion()

        self.optims, self.scheduler = self.make_opt_sched(
            self.config, [self.model, self.online_probe, self.criterion]
        )

        if "sigreg_sched" in self.config.cva:
            self.scheduler.add(
                self.criterion,
                "sigreg_lambda_",
                Schedule.parse(self.config.cva.sigreg_sched),
            )

        self.save_hyperparameters(self.config.to_dict())

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
                    if (
                        "is_cva_module_proj" in opt.param_groups[group_num]
                        and opt.param_groups[group_num]["is_cva_module_proj"]
                    ):
                        if "cva_module_proj_sched" in self.config.cva:
                            cva_proj_sched = Schedule.parse(self.config.cva.cva_module_proj_sched)
                        else:
                            cva_proj_sched = Schedule.parse(sched)

                        scheduler.add(
                            opt.param_groups[group_num],
                            key,
                            cva_proj_sched,
                            apply_lr_multiplier,
                            "proj_sched",
                        )
                    elif (
                        "is_cva_module_cls" in opt.param_groups[group_num]
                        and opt.param_groups[group_num]["is_cva_module_cls"]
                    ):
                        if "cva_module_cls_sched" in self.config.cva:
                            cva_cls_sched = Schedule.parse(self.config.cva.cva_module_cls_sched)
                        else:
                            cva_cls_sched = Schedule.parse(sched)

                        scheduler.add(
                            opt.param_groups[group_num],
                            key,
                            cva_cls_sched,
                            apply_lr_multiplier,
                            "cls_sched",
                        )
                    elif (
                        "is_online_probe" in opt.param_groups[group_num]
                        and opt.param_groups[group_num]["is_online_probe"]
                    ):
                        op_sched = Schedule.parse(
                            "CatSched(LinSched(1e-5, 1e-3), CosSched(1e-3, 1e-6), 10)"
                        )

                        scheduler.add(
                            opt.param_groups[group_num],
                            key,
                            op_sched,
                            None,
                            "op_sched_lr",
                        )
                    else:
                        scheduler.add(
                            opt.param_groups[group_num],
                            key,
                            Schedule.parse(sched),
                            apply_lr_multiplier,
                        )

            if key == "weight_decay":
                for group_num in range(len(opt.param_groups)):
                    if (
                        "is_online_probe" in opt.param_groups[group_num]
                        and opt.param_groups[group_num]["is_online_probe"]
                    ):
                        op_sched = Schedule.parse("ConstSched(1e-7)")

                        scheduler.add(
                            opt.param_groups[group_num],
                            key,
                            op_sched,
                            apply_wd_multiplier,
                            "op_sched_wd",
                        )
                    else:
                        scheduler.add(
                            opt.param_groups[group_num],
                            key,
                            Schedule.parse(sched),
                            apply_wd_multiplier,
                        )

        return [opt], scheduler

    def get_model_cls(self, model_type):
        if model_type == "vitv3_t":
            model_cls = proj_vitv3_tiny
        elif model_type == "vitv3_s":
            model_cls = proj_vitv3_small
        elif model_type == "vitv3_b":
            model_cls = proj_vitv3_base
        elif model_type == "vitv3_l":
            model_cls = proj_vitv3_large
        elif model_type == "vitv2_t":
            model_cls = proj_vitv2_tiny
        elif model_type == "vitv2_s":
            model_cls = proj_vitv2_small
        elif model_type == "vitv2_b":
            model_cls = proj_vitv2_base
        elif model_type == "vitv2_l":
            model_cls = proj_vitv2_large
        elif model_type == "rn18":
            model_cls = proj_resnet_18
        elif model_type == "rn34":
            model_cls = proj_resnet_34
        elif model_type == "rn50":
            model_cls = proj_resnet_50
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
        return model_cls

    def make_model(self):
        model_type = self.config.model.type
        model_params = self.config.model.get("params", {})

        model_cls = self.get_model_cls(model_type)

        model = model_cls(**model_params)
        online_probe = OnlineProbe(model.embed_dim, self.config.num_classes)

        if hasattr(self.config.model, "pt"):
            pretrained_weights_path = (
                MODELS_DIR / WANDB_PROJECT / Path(self.config.model.pt.path) / "last.ckpt"
            )

            m_pre = (
                self.config.model.pt.s_pre if hasattr(self.config.model.pt, "s_pre") else "model"
            )

            ckpt_sd = torch.load(pretrained_weights_path, map_location=self.device)["state_dict"]
            s_sd = {
                k.replace(f"{m_pre}.", "", 1): v
                for k, v in ckpt_sd.items()
                if k.startswith(f"{m_pre}.")
            }

            online_probe_sd = {
                k.replace("online_probe.", "", 1): v
                for k, v in ckpt_sd.items()
                if k.startswith("online_probe.")
            }

            # we might need to interpolate the keys with *pos_embed* to the new image_size
            if s_sd:
                print("Loading and interpolating student model weights...")
                s_sd = load_and_interpolate_pos_embed(s_sd, model)
                s_sd = filter_state_dict(s_sd, model)

                # Load student model weights
                missing_keys, unexpected_keys = model.load_state_dict(s_sd, strict=False)
                if missing_keys:
                    logger.warning(f"Model missing keys: {missing_keys}")
                if unexpected_keys:
                    logger.warning(f"Model unexpected keys: {unexpected_keys}")

            if online_probe_sd:
                missing_keys, unexpected_keys = online_probe.load_state_dict(
                    online_probe_sd, strict=False
                )
                if missing_keys:
                    logger.warning(f"Model missing keys: {missing_keys}")
                if unexpected_keys:
                    logger.warning(f"Model unexpected keys: {unexpected_keys}")

        return model, online_probe

    def make_criterion(self):
        return LeJepaSSLoss()

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def batch_to_loss(self, batch, train=False):
        """
        Converts a batch of data into a loss value.
        Args:
            batch: A batch of data containing source and target features, and labels.
            train: If True, the model is in training mode.

        Returns:
            torch.Tensor: The computed loss value.
        """

        xs, y_gt = batch

        batch_size = xs[0][0].shape[0]
        x = self.normalisation(torch.cat(xs[0], dim=0))  # this class only expects global views

        outs = self(x, is_training=True)

        raw_latent = outs["raw_latent"]
        cls_seed = self.model.cva_module_proj(outs["latent"])  # [B, D]

        y_hat = self.online_probe(raw_latent.detach())

        loss, labels, logits = self.criterion(
            self.global_step,
            batch_size,
            cls_seed,
            raw_latent,
            y_hat,
            y_gt.repeat(y_hat.size(0) // y_gt.size(0)),
        )

        return loss, labels, logits

    def validation_step(self, batch, batch_idx):
        """
        Validation step for the trainer.

        Args:
            batch: A batch of data containing source and target features, and labels.
            batch_idx: Index of the batch.
        Returns:
            torch.Tensor: The computed loss value.
        """

        x, y = batch
        out = self.model(self.normalisation(x), is_training=True)
        raw_latent = out["raw_latent"]
        mcws = self.criterion.compute_mcws(raw_latent)

        y_hat = self.online_probe(raw_latent.detach())
        ce = F.cross_entropy(y_hat, y)
        acc = y_hat.argmax(dim=-1).eq(y).sum().float().div_(y.shape[0])

        loss = {"loss": ce, "probe/ce": ce, "probe/acc": acc, "mcws": mcws}
        loss = self.log_loss(loss, prefix="val", prog_bar=True, on_epoch=True, on_step=False)

        return loss

    def configure_callbacks(self) -> Union[Sequence[Callback], Callback]:
        return [self.scheduler]
