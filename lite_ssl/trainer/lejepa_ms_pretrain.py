import torch

from lite_ssl.trainer.lejepa_ss_pretrain import LeJepaSSLoss, LeJepaSSTrainer


class LeJepaMSLoss(LeJepaSSLoss):

    def compute_sigreg_inv_cls(self, cls_projs, batch_size, global_step):
        sim_cls = 0.0
        total = 0

        centers = cls_projs[0].reshape(-1, batch_size, cls_projs[0].shape[-1]).mean(0)
        start = 0 if len(cls_projs) == 1 else 1
        for scale_cls in cls_projs[start:]:
            sim_cls += (
                (centers - scale_cls.reshape(-1, batch_size, cls_projs[0].shape[-1]))
                .square()
                .mean()
            )
            total += 1
        sim_cls = sim_cls / total

        sig_reg_loss_cls = torch.stack(
            [
                self.sig_reg(seedlet_cls_chunk, global_step)
                for scale_cls in cls_projs
                for seedlet_cls_chunk in scale_cls.reshape(-1, batch_size, cls_projs[0].shape[-1])
            ],
            dim=0,
        ).mean()

        return sig_reg_loss_cls, sim_cls

    def forward(
        self, global_step, batch_size, seed_clss, latent, probe_logits, y_gt, *args, **kwargs
    ):
        """
        Forward pass for regularised loss computation.

        Args:
            global_step: int
            batch_size: int
            seed_clss: (torch.Tensor):
            probe_logits: (torch.Tensor):
            y_gt: (torch.Tensor):
            latent: (torch.Tensor):

        Returns:
            torch.Tensor: The computed loss value.
        """
        normalised_mcws = self.compute_mcws(latent)

        sig_reg_loss_cls, sim_cls = self.compute_sigreg_inv_cls(seed_clss, batch_size, global_step)

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


class LeJepaMSTrainer(LeJepaSSTrainer):

    def make_criterion(self):
        """
        Creates the loss function for regularisation

        Returns:
            LeJepaMSLoss: A wrapper around the classification and regularisation loss
        """
        return LeJepaMSLoss()

    def batch_to_loss(self, batch, train=False):
        """
        Converts a batch of data into a loss value.
        Args:
            batch: A batch of data containing source and target features, and labels.
            train: If True, the model is in training mode.

        Returns:
            torch.Tensor: The computed loss value.
        """

        scales, y_gt = batch

        batch_size = scales[0][0].shape[0]

        scales = [self.normalisation(torch.cat(scale, dim=0)) for scale in scales]

        scale_outs = self(scales, is_training=True)
        raw_latent = scale_outs[0]["raw_latent"]

        scale_projs = [
            self.model.cva_module_proj(scale_out["latent"]) for scale_out in scale_outs
        ]  # [B, D]

        y_hat = self.online_probe(raw_latent.detach())

        loss, labels, logits = self.criterion(
            self.global_step,
            batch_size,
            scale_projs,
            raw_latent,
            y_hat,
            y_gt.repeat(y_hat.shape[0] // y_gt.shape[0]),
        )

        return loss, labels, logits
