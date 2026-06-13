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
        # self.sk_teacher.compile()

    def forward(self, s, t):
        t = self.sk_teacher(t, self.teacher_temp)

        loss = lossfunc(t, s, self.student_temp)

        return -loss


if __name__ == "__main__":
    # ---------------------------------------------------
    # Test code
    # ---------------------------------------------------
    torch.manual_seed(0)

    # hyperparameters for test
    batch = 512
    ibot_patches = 20  # because DINOLoss splits into 2 with chunk()
    out_dim = 256  # prototype dimension

    # Instantiate loss
    criterion = IBotPatchLoss()

    # Create dummy student logits: shape [2*crops, batch, K]
    student_logits = torch.randn(batch * ibot_patches, out_dim)

    # Create dummy teacher probs: shape [2*crops, batch, K]
    teacher_probs = torch.randn(batch * ibot_patches, out_dim)

    # Forward pass
    loss = criterion(student_logits, teacher_probs)

    print("Loss:", loss.shape)
