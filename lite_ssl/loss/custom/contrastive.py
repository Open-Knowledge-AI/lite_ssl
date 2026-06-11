import torch
import torch.nn.functional as F

from functools import lru_cache


@lru_cache(maxsize=5)
def _get_correlated_mask(b, device, using_teacher, verbose=False):
    eye = torch.eye(2 * b, device=device, dtype=torch.uint8)
    shifted = eye.roll(-b, dims=1)
    mask = eye + shifted

    if using_teacher:
        l_pos = eye.roll(-b // 2, dims=1)
        r_pos = eye.roll(b // 2, dims=1)
        mask = mask + l_pos + r_pos

    mask = mask.bool().logical_not()

    if verbose:
        import matplotlib.pyplot as plt

        plt.imshow(mask.detach().cpu().numpy(), interpolation="nearest")
        plt.title("Correlated Mask")
        plt.colorbar()
        plt.tight_layout()
        plt.show()

    return mask


class NTXentLoss:
    def __init__(self, temperature=0.5):
        self.temperature = temperature

    def __call__(self, z_i, z_j):
        batch_size = z_i.size(0)

        z = torch.cat([z_i, z_j], dim=0)
        sim = F.cosine_similarity(z.unsqueeze(1), z.unsqueeze(0), dim=2) / self.temperature
        sim_i_j = torch.diag(sim, batch_size)
        sim_j_i = torch.diag(sim, -batch_size)

        positives = torch.cat([sim_i_j, sim_j_i], dim=0)

        # Build the identity mask
        negatives_mask = _get_correlated_mask(batch_size, z.device, using_teacher=True)

        negatives = sim[negatives_mask].view(2 * batch_size, -1)

        logits = torch.cat([positives.unsqueeze(1), negatives], dim=1)
        labels = torch.zeros(2 * batch_size, dtype=torch.long, device=z.device)

        return F.cross_entropy(logits, labels), logits, labels


class GlobalCosineRegression:

    def __init__(self, reduction="mean"):
        self.reduction = reduction

    def __call__(self, z_i, z_j):
        eps = torch.finfo(z_i.dtype).eps
        nz_i, nz_j = F.normalize(z_i, dim=-1, eps=eps), F.normalize(z_j, dim=-1, eps=eps)

        point_wise = 2 - 2 * (nz_i * nz_j).sum(-1)
        if self.reduction == "mean":
            return point_wise.mean()
        elif self.reduction == "sum":
            return point_wise.sum()
        else:
            return point_wise


if __name__ == "__main__":
    _loss_fn = GlobalCosineRegression(reduction="none")
    _z_i = torch.randn(8, 128)
    _z_j = torch.randn(8, 128)
    _loss = _loss_fn(_z_i, _z_j)
    print(f"Global Cosine Regression Loss: {_loss.shape}")
