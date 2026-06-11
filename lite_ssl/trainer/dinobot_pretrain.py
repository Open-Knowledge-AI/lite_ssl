import torch
import torch.nn.functional as F

from lite_ssl.trainer.dino_pretrain import DiNoLoss, DiNoTrainer

from lite_ssl.loss.custom import (
    IBotPatchLoss,
)


class DiNoBOTLoss(DiNoLoss):

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
        super().__init__(**ignored_kwargs)

        self.ibot_patch_loss = IBotPatchLoss()
        self.ibot_lambda_ = 1.0

    def forward(
        self,
        g_cls_s,
        g_cls_t,
        masks,
        l_p_s,
        l_p_t,
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
            masks: (torch.Tensor): the mae masks
            l_p_s: (torch.Tensor): The patch latents from the student model
            l_p_t: (torch.Tensor): The patch latents from the teacher model
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

        masks_per_sample = masks.sum(dim=-1)
        weights_per_sample = 1 / masks_per_sample.clamp(min=1.0)
        ibot_vec = self.ibot_patch_loss(l_p_s, l_p_t)
        weights_per_patch = torch.repeat_interleave(weights_per_sample, masks_per_sample)
        ibot_loss = torch.sum(ibot_vec * weights_per_patch) / torch.count_nonzero(masks_per_sample)

        normed_latents = torch.chunk(normed_latent, chunks=2, dim=0)
        koleo = 0.5 * (self.koleo(normed_latents[0]) + self.koleo(normed_latents[1]))

        ce = self.probe_loss(logits, y_gt)
        with torch.no_grad():
            acc = logits.argmax(dim=-1).eq(y_gt).sum().float().div_(y_gt.shape[0])

        return (
            {
                "loss": (
                    self.dino_lambda_ * dino_loss
                    + self.ibot_lambda_ * ibot_loss
                    + self.koleo_lambda_ * koleo
                    + ce
                ),
                "dino": dino_loss,
                "ibot": ibot_loss,
                "mcws": normalised_mcws,
                "koleo": koleo,
                "probe/ce": ce,
                "probe/acc": acc,
            },
            None,
            None,
        )


class DiNoBOTTrainer(DiNoTrainer):

    def make_criterion(self):
        """
        Creates the loss function for regularisation

        Returns:
            DiNoLoss: A wrapper around the classification and regularisation loss
        """
        return DiNoBOTLoss()

    def batch_to_loss(self, batch, train=False):
        """
        Converts a batch of data into a loss value.
        Args:
            batch: A batch of data containing source and target features, and labels.
            train: If True, the model is in training mode.

        Returns:
            torch.Tensor: The computed loss value.
        """

        xs, masks, y_gt = batch
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
            t_p_l = teacher_outs["patch_latent"]
            t_p_l = self.teacher_model.cva_module_proj(
                t_p_l[masks.unsqueeze(-1).expand_as(t_p_l)].reshape(-1, t_p_l.size(-1))
            )

        masks_list = [
            masks,
            *[None for _ in range(len(xs) - 1)],
        ]  # masks only exist for the global views
        student_outs = self(
            x,  # all views
            masks_list,
            is_training=True,
        )
        s_l = torch.cat(
            [
                self.model.cva_module_proj(s_out["latent"]).unflatten(0, (-1, bs))
                for s_out in student_outs
            ],
            dim=0,
        )  # [(g + l), bs, k]

        s_p_l = student_outs[0]["patch_latent"]
        s_p_l = self.model.cva_module_proj(
            s_p_l[masks.unsqueeze(-1).expand_as(s_p_l)].reshape(-1, s_p_l.size(-1))
        )

        y_hat = self.online_probe(raw_latent.detach())

        loss, labels, logits = self.criterion(
            s_l,
            t_l,
            masks,
            s_p_l,
            t_p_l,
            student_outs[0]["latent"],
            raw_latent,
            y_hat,
            y_gt.repeat(y_hat.size(0) // y_gt.size(0)),
        )

        return loss, labels, logits
