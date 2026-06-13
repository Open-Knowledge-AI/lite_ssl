import torch
import torch.nn.functional as F

from torch import nn

from lite_ssl.loss.custom.util import SinkhornKnopp


class DINOLoss(nn.Module):
    def __init__(
        self,
        teacher_temp=0.04,
        student_temp=0.1,
    ):
        super().__init__()
        self.student_temp = student_temp
        self.teacher_temp = teacher_temp
        self.sk_teacher = SinkhornKnopp()

    def forward(self, student_logits, teacher_logits, ignore_diagonal=True):
        """
        Cross-entropy between softmax outputs of the teacher and student networks.
        student_logits: [student crops * batch, prototypes]
        teacher_logits:  [teacher crops * batch, prototypes] must sum to 1 over the last dim

        loss = 0
        count = 0
        for each sample `b` in the batch:
            for each student crop `s` of this sample:
                for each teacher crop `t` of this sample:
                    if ignore_diagonal and s == t:
                        continue
                    loss += cross_entropy(softmax(student_logits[s, b] / student_temp), teacher_probs[t, b])
                    count += 1
        return loss / count
        """
        gv, bs, k = teacher_logits.shape

        teacher_probs = self.sk_teacher(teacher_logits.flatten(0, 1), self.teacher_temp).unflatten(
            0, (gv, bs)
        )

        # 2) Stack into [num_crops, batch, k]
        if isinstance(student_logits, (list, tuple)):
            return torch.stack(
                [
                    self.compute_loss(bs, student_logit, teacher_probs, ignore_diagonal)
                    for student_logit in student_logits
                ],
                dim=0,
            ).mean()
        else:
            return self.compute_loss(bs, student_logits, teacher_probs, ignore_diagonal)

    def compute_loss(self, bs, student_logits, teacher_probs, ignore_diagonal):
        student_crops = student_logits.size(0)
        teacher_crops, _, _ = teacher_probs.shape

        # 3) Student log-softmax for loss
        log_student = F.log_softmax(student_logits.float() / self.student_temp, dim=-1)

        # -------------------------
        # Loss computation
        # -------------------------
        if not ignore_diagonal:
            loss = -torch.einsum("s b k, t b k -> ", log_student, teacher_probs)
            loss = loss / (bs * student_crops * teacher_crops)
        else:
            loss = -torch.einsum("s b k, t b k -> s t", log_student, teacher_probs)
            min_st = min(student_crops, teacher_crops)
            loss = torch.diagonal_scatter(loss, loss.new_zeros(min_st))
            loss = loss.sum() / (bs * student_crops * teacher_crops - bs * min_st)

        return loss


class SymplecticDINOLoss(DINOLoss):

    def forward(self, student_logits, teacher_logits, ignore_diagonal=True):
        """
        Cross-entropy between softmax outputs of the teacher and student networks.
        student_logits: [student crops * batch, prototypes]
        teacher_logits:  [teacher crops * batch, prototypes] must sum to 1 over the last dim

        loss = 0
        count = 0
        for each sample `b` in the batch:
            for each student crop `s` of this sample:
                for each teacher crop `t` of this sample:
                    if ignore_diagonal and s == t:
                        continue
                    loss += cross_entropy(softmax(student_logits[s, b] / student_temp), teacher_probs[t, b])
                    count += 1
        return loss / count
        """
        gv2, bs, k = teacher_logits.shape

        teacher_probs = self.sk_teacher(teacher_logits.flatten(0, 1), self.teacher_temp).unflatten(
            0, (gv2, bs)
        )
        teacher_probs_now, teacher_probs_next = torch.chunk(teacher_probs, chunks=2, dim=0)

        return torch.stack(
            [
                self.compute_loss(bs, student_logit, teacher_prob, ignore)
                for student_logit, teacher_prob, ignore in zip(
                    student_logits,
                    [teacher_probs_next, teacher_probs_now],
                    [ignore_diagonal, False],
                )
            ],
            dim=0,
        ).mean()


class EvidenceDINOLoss(DINOLoss):

    def forward(self, student_logits, teacher_logits, **kwargs):
        """
        Cross-entropy between softmax outputs of the teacher and student networks.
        student_logits: [student crops * batch, prototypes]
        teacher_logits:  [teacher crops * batch, prototypes] must sum to 1 over the last dim

        loss = 0
        count = 0
        for each sample `b` in the batch:
            for each student crop `s` of this sample:
                for each teacher crop `t` of this sample:
                    if ignore_diagonal and s == t:
                        continue
                    loss += cross_entropy(softmax(student_logits[s, b] / student_temp), teacher_probs[t, b])
                    count += 1
        return loss / count
        """
        gv, bs, k = teacher_logits.shape

        teacher_probs = self.sk_teacher(teacher_logits.flatten(0, 1), self.teacher_temp).unflatten(
            0, (gv, bs)
        )

        sg_logits, sgl_logits = student_logits
        # sizes = [sg_logits.size(0), sl_logits.size(0)]
        # tg_probs, tl_probs = teacher_probs.split_with_sizes(sizes, dim=0)

        return torch.stack(
            [
                self.compute_loss(bs, sg_logits, teacher_probs, True, only_diagonal=False),
                self.compute_loss(bs, sgl_logits, teacher_probs, True, only_diagonal=False),
                # self.compute_loss(bs, sl_logits, tl_probs, False, only_diagonal=True),
            ],
            dim=0,
        ).mean()

    def compute_loss(
        self, bs, student_logits, teacher_probs, ignore_diagonal=True, only_diagonal=False
    ):
        student_crops = student_logits.size(0)
        teacher_crops, _, _ = teacher_probs.shape

        # 3) Student log-softmax for loss
        log_student = F.log_softmax(student_logits.float() / self.student_temp, dim=-1)

        # -------------------------
        # Loss computation
        # -------------------------
        if only_diagonal:
            loss = -(log_student * teacher_probs).sum(dim=-1).mean()
        elif not ignore_diagonal:
            loss = -torch.einsum("s b k, t b k -> ", log_student, teacher_probs)
            loss = loss / (bs * student_crops * teacher_crops)
        else:
            loss = -torch.einsum("s b k, t b k -> s t", log_student, teacher_probs)
            min_st = min(student_crops, teacher_crops)
            loss = torch.diagonal_scatter(loss, loss.new_zeros(min_st))
            loss = loss.sum() / (bs * student_crops * teacher_crops - bs * min_st)

        return loss


class SlottedDINOLoss(nn.Module):
    def __init__(
        self,
        g_slots,
        teacher_temp=0.04,
        student_temp=0.1,
    ):
        super().__init__()
        self.g_slots = g_slots
        self.student_temp = student_temp
        self.teacher_temp = teacher_temp
        self.sk_teacher = SinkhornKnopp()

    def forward(self, student_logits, teacher_logits, matching_lg, *args, **kwargs):
        """
        Cross-entropy between the matched slots on the softmax outputs of the teacher and student networks.
        """
        bs = teacher_logits.size(0)

        teacher_probs = self.sk_teacher(teacher_logits.flatten(0, 1), self.teacher_temp)
        loss_gg = torch.nn.functional.cross_entropy(
            student_logits[0]
            .unflatten(1, (-1, self.g_slots))
            .roll(1, 1)
            .flatten(1, 2)
            .flatten(0, 1)
            / self.student_temp,
            teacher_probs,
        )

        idx = matching_lg.unsqueeze(-1).expand(-1, -1, teacher_probs.size(-1))
        global_matched_probs = torch.gather(
            teacher_probs.unflatten(0, (bs, -1)), dim=1, index=idx
        ).flatten(0, 1)
        loss_lg = torch.nn.functional.cross_entropy(
            student_logits[1].flatten(0, 1) / self.student_temp,
            global_matched_probs,
        )

        return loss_gg, loss_lg


if __name__ == "__main__":
    # ---------------------------------------------------
    # Test code
    # ---------------------------------------------------
    torch.manual_seed(0)

    # hyperparameters for test
    _bs = 32
    _s_crops = 6
    _t_crops = 2  # because DINOLoss splits into 2 with chunk()
    _k = 1024  # prototype dimension

    # Instantiate loss
    _criterion = SymplecticDINOLoss()

    _sl_l = torch.randn(_s_crops, _bs, _k)
    _tg_l = torch.randn(2 * _t_crops, _bs, _k)
    _sg_l = torch.randn(_t_crops, _bs, _k)

    # Forward pass
    _loss = _criterion([_sg_l, _sl_l], _tg_l, ignore_diagonal=True)

    print("Loss:", _loss.item())
