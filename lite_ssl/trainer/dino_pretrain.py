from copy import deepcopy
from pathlib import Path
from typing import Union, Sequence

import torch
import torch.nn.functional as F

from lightning import Callback
from ml_collections import ConfigDict

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
from lite_ssl.loss.custom import (
    DINOLoss,
    KoLeoLoss,
)


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


class LastLayerLRMult:

    def __init__(self, turn_over_point=1250):
        self.turn_over_point = turn_over_point

    def __call__(self, loc, step, sched):
        if step < self.turn_over_point:
            return 0.0
        else:
            return apply_lr_multiplier(loc, step, sched)


def apply_wd_multiplier(loc, step, sched):
    return loc.get("wd_multiplier", 1.0) * sched(step)


class DiNoLoss(torch.nn.Module):

    def __init__(
        self,
        **ignored_kwargs,
    ):
        """
        Initializes the LossWrapper.

        Args:
            similarity: The type of similarity measure to use ("cosine" or "euclidean" or "gram").
            ignored_kwargs: consistency loss specific kwargs
        """
        super().__init__()
        logger.info(f"{ignored_kwargs=}")

        self.dino_cls_loss = DINOLoss()
        self.koleo = KoLeoLoss()

        self.probe_loss = torch.nn.CrossEntropyLoss()

        self.dino_lambda_ = 1.0
        self.koleo_lambda_ = 0.1

    def forward(
        self,
        g_cls_s,
        g_cls_t,
        normed_latent,
        latent,
        logits,
        y_gt,
        *args,
        **kwargs,
    ):
        """
        Forward pass for regularised loss computation.

        Args:
            g_cls_s: (torch.Tensor): The global latents from the student model (both projected and predicted)
            g_cls_t: (torch.Tensor): The global latents from the teacher model (only projected)
            normed_latent: (torch.Tensor): The global latent from the student but normed
            latent: (torch.Tensor): the raw pre-norm pre-head latent
            y_gt: (torch.Tensor): the ground truth pred
            logits: (torch.Tensor): the predicted online probe logits

        Returns:
            torch.Tensor: The computed loss value.
        """
        with torch.no_grad():
            mean_channel_wise_std = torch.std(F.normalize(latent, dim=-1), dim=0).mean()
            normalised_mcws = mean_channel_wise_std / (1 / latent.shape[1] ** 0.5)

        dino_loss = self.dino_cls_loss(g_cls_s, g_cls_t)

        normed_latents = torch.chunk(normed_latent, chunks=2, dim=0)
        koleo = 0.5 * (self.koleo(normed_latents[0]) + self.koleo(normed_latents[1]))

        ce = self.probe_loss(logits, y_gt)
        with torch.no_grad():
            acc = logits.argmax(dim=-1).eq(y_gt).sum().float().div_(y_gt.shape[0])

        return (
            {
                "loss": (self.dino_lambda_ * dino_loss + self.koleo_lambda_ * koleo + ce),
                "dino": dino_loss,
                "koleo": koleo,
                "mcws": normalised_mcws,
                "probe/ce": ce,
                "probe/acc": acc,
            },
            None,
            None,
        )


class DiNoTrainer(BaseTrainer):
    """
    Trainer for models with Consistent View Alignment (CVA).
    """

    def __init__(
        self, config: ConfigDict, normalisation: torch.nn.Module, probe_valid_dl, probe_train_dl
    ):
        super().__init__(config, normalisation, probe_valid_dl, probe_train_dl)

        self.model, self.teacher_model, self.online_probe = self.make_model()
        self._freeze_teacher()
        self.teacher_momentum = self.config.cva.get("teacher_momentum", 0.994)

        self.optims, self.scheduler = self.make_opt_sched(
            self.config, [self.model, self.online_probe]
        )

        self.img_size, self.patch_size = self.model.img_size, self.model.patch_size
        self.num_patch = self.img_size // self.patch_size

        if "teacher_momentum_sched" in self.config.cva:
            self.scheduler.add(
                self,
                "teacher_momentum",
                Schedule.parse(self.config.cva.teacher_momentum_sched),
            )

        if "teacher_temp_sched" in self.config.cva:
            if hasattr(self.criterion, "dino_cls_loss"):
                self.scheduler.add(
                    self.criterion.dino_cls_loss,
                    "teacher_temp",
                    Schedule.parse(self.config.cva.teacher_temp_sched),
                )

            if hasattr(self.criterion, "ibot_patch_loss"):
                self.scheduler.add(
                    self.criterion.ibot_patch_loss,
                    "teacher_temp",
                    Schedule.parse(self.config.cva.teacher_temp_sched),
                )

        self.ema_params_lists = self.make_param_lists()

        self.save_hyperparameters(self.config.to_dict())

    @classmethod
    def make_opt_sched(
        cls, config, trainable_modules
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
                        "is_last_layer" in opt.param_groups[group_num]
                        and opt.param_groups[group_num]["is_last_layer"]
                    ):
                        scheduler.add(
                            opt.param_groups[group_num],
                            key,
                            Schedule.parse(sched),
                            LastLayerLRMult(
                                config.cva.turn_over_point
                                if hasattr(config.cva, "turn_over_point")
                                else 1250
                            ),
                            "last_layer_lr",
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

    def make_param_lists(self):
        student_param_list = []
        teacher_param_list = []
        for ms, mt in zip(self.model.parameters(), self.teacher_model.parameters()):
            student_param_list += [ms]
            teacher_param_list += [mt]
        return student_param_list, teacher_param_list

    def _freeze_teacher(self):
        for p in self.teacher_model.parameters():
            p.requires_grad = False
        self.teacher_model.eval()

    def make_model(self):
        model_type = self.config.model.type
        model_params = self.config.model.get("params", {})

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

        model = model_cls(**model_params)
        teacher_model = deepcopy(model)
        online_probe = OnlineProbe(model.embed_dim, self.config.num_classes)

        if hasattr(self.config.model, "pt"):
            pretrained_weights_path = (
                MODELS_DIR / WANDB_PROJECT / Path(self.config.model.pt.path) / "last.ckpt"
            )

            m_pre = (
                self.config.model.pt.s_pre if hasattr(self.config.model.pt, "s_pre") else "model"
            )
            t_pre = (
                self.config.model.pt.t_pre
                if hasattr(self.config.model.pt, "t_pre")
                else "teacher_model"
            )

            ckpt_sd = torch.load(pretrained_weights_path, map_location=self.device)["state_dict"]

            s_sd = {
                k.replace(f"{m_pre}.", "", 1): v
                for k, v in ckpt_sd.items()
                if k.startswith(f"{m_pre}.")
            }
            t_sd = {
                k.replace(f"{t_pre}.", "", 1): v
                for k, v in ckpt_sd.items()
                if k.startswith(f"{t_pre}.")
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
                    print(f"Student model missing keys: {missing_keys}")
                if unexpected_keys:
                    print(f"Student model unexpected keys: {unexpected_keys}")

            # Process teacher model state dict
            if t_sd:
                print("Loading and interpolating teacher model weights...")
                t_sd = load_and_interpolate_pos_embed(t_sd, teacher_model)
                t_sd = filter_state_dict(t_sd, teacher_model)

                # Load teacher model weights
                missing_keys, unexpected_keys = teacher_model.load_state_dict(t_sd, strict=False)
                if missing_keys:
                    print(f"Teacher model missing keys: {missing_keys}")
                if unexpected_keys:
                    print(f"Teacher model unexpected keys: {unexpected_keys}")

            if online_probe_sd:
                missing_keys, unexpected_keys = online_probe.load_state_dict(
                    online_probe_sd, strict=False
                )
                if missing_keys:
                    logger.warning(f"Model missing keys: {missing_keys}")
                if unexpected_keys:
                    logger.warning(f"Model unexpected keys: {unexpected_keys}")

        return model, teacher_model, online_probe

    def make_criterion(self):
        """
        Creates the loss function for regularisation

        Returns:
            DiNoLoss: A wrapper around the classification and regularisation loss
        """
        return DiNoLoss()

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
        bs = xs[0][0].shape[0]

        x = [self.normalisation(torch.cat(_x, dim=0)) for _x in xs]  # [g * bs, *]

        with torch.no_grad():
            self.teacher_model.eval()
            teacher_outs = self.teacher_model(
                x[0],  # only global views
                is_training=True,
            )
            raw_latent = teacher_outs["raw_latent"]
            t_l = self.teacher_model.cva_module_proj(teacher_outs["latent"]).unflatten(
                0, (-1, bs)
            )  # [g, bs, k]

        student_outs = self(
            x,  # all views
            is_training=True,
        )
        s_l = torch.cat(
            [
                self.model.cva_module_proj(s_out["latent"]).unflatten(0, (-1, bs))
                for s_out in student_outs
            ],
            dim=0,
        )  # [(g + l), bs, k]

        y_hat = self.online_probe(raw_latent.detach())

        loss, labels, logits = self.criterion(
            s_l,
            t_l,
            student_outs[0]["latent"],
            raw_latent,
            y_hat,
            y_gt.repeat(y_hat.size(0) // y_gt.size(0)),
        )

        return loss, labels, logits

    def on_train_start(self):
        # make sure teacher is on same device as model
        self.teacher_model.to(self.device)
        self._freeze_teacher()

    @torch.no_grad()
    def _update_teacher(self):
        """Exponential Moving Average (EMA) update for the teacher model.
        Teacher ← m * Teacher + (1 - m) * Student
        Handles both parameters and buffers safely across dtypes/devices.
        """
        m = float(self.teacher_momentum)  # make sure it's a Python float, not tensor

        student_param_list, teacher_param_list = self.ema_params_lists

        with torch.autocast("cuda", enabled=False):
            with torch.no_grad():
                torch._foreach_mul_(teacher_param_list, m)
                torch._foreach_add_(teacher_param_list, student_param_list, alpha=1 - m)

    def optimizer_step(self, *args, **kwargs):
        # Let Lightning do the normal optimizer step
        super().optimizer_step(*args, **kwargs)

        with torch.no_grad():
            # Then do the teacher update
            self._update_teacher()

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
        out = self.teacher_model(self.normalisation(x), is_training=True)
        raw_latent = out["raw_latent"]

        with torch.no_grad():
            mean_channel_wise_std = torch.std(F.normalize(raw_latent, dim=-1), dim=0).mean()
            mcws = mean_channel_wise_std / (1 / raw_latent.shape[1] ** 0.5)

        y_hat = self.online_probe(raw_latent.detach())
        ce = F.cross_entropy(y_hat, y)
        acc = y_hat.argmax(dim=-1).eq(y).sum().float().div_(y.shape[0])

        loss = {"loss": ce, "probe/ce": ce, "probe/acc": acc, "mcws": mcws}
        loss = self.log_loss(loss, prefix="val", prog_bar=True, on_epoch=True, on_step=False)

        return loss

    def configure_callbacks(self) -> Union[Sequence[Callback], Callback]:
        return [self.scheduler]
