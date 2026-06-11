import torch
import torch.nn.functional as F

from torch import nn

from lite_ssl.loss.custom.util import SinkhornKnopp


def lossfunc(t, s, temp):  # noqa: F811
    return torch.sum(t.float() * F.log_softmax(s.float() / temp, dim=-1), dim=-1)


class IBotPatchLoss(nn.Module):
    def __init__(
        self,
        teacher_temp=0.04,
        student_temp=0.1,
    ):
        super().__init__()
        self.student_temp = student_temp
        self.teacher_temp = teacher_temp
        self.sk_teacher = SinkhornKnopp()

    def forward(self, s, t):
        t = self.sk_teacher(t, self.teacher_temp)

        loss = lossfunc(t, s, self.student_temp)

        return -loss
