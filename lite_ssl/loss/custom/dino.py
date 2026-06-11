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
