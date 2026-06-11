import torch
import torch.nn as nn
import torch.nn.functional as F


def cosine_regression(x, y):
    return 2 - 2 * (x * y).sum(dim=-1)


class CosineRegressionLoss(nn.Module):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, vec_1, vec_2):
        eps = torch.finfo(vec_1.dtype).eps

        # normalise
        vec_1 = F.normalize(vec_1, dim=-1, p=2, eps=eps)
        vec_2 = F.normalize(vec_2, dim=-1, p=2, eps=eps)

        return cosine_regression(vec_1, vec_2)


class EuclideanRegressionLoss(nn.Module):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, vec_1, vec_2):
        return (vec_1 - vec_2).square().sum(dim=-1)


class GramMatrixConsistencyLoss(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mse_loss = torch.nn.MSELoss(reduction="none")

    def forward(self, feature_map_1, feature_map_2):
        eps = torch.finfo(feature_map_1.dtype).eps

        # apply normalisation
        flattened_feature_map_1 = F.normalize(feature_map_1, p=2, dim=-1, eps=eps)
        flattened_feature_map_2 = F.normalize(feature_map_2, p=2, dim=-1, eps=eps)

        mm = torch.bmm if feature_map_1.ndim == 3 else torch.mm

        gram_1 = mm(flattened_feature_map_1, flattened_feature_map_1.transpose(-1, -2))
        gram_2 = mm(flattened_feature_map_2, flattened_feature_map_2.transpose(-1, -2))

        return self.mse_loss(gram_1, gram_2).mean()


if __name__ == "__main__":
    _b = 32
    _r = 12
    _d = 384

    # Test Gram Matrix Consistency Loss
    feature_map_1 = torch.randn(_b, _r * _r, _d, requires_grad=False, device="cuda")
    feature_map_2 = torch.randn(_b, _r * _r, _d, requires_grad=True, device="cuda")

    cosine_loss_fn = CosineRegressionLoss()
    cosine_loss = cosine_loss_fn(feature_map_1, feature_map_2)
    print(f"Cosine Regression Loss: {cosine_loss.shape}")

    euclidean_loss_fn = EuclideanRegressionLoss()
    euclidean_loss = euclidean_loss_fn(feature_map_1, feature_map_2)
    print(f"Euclidean Regression Loss: {euclidean_loss.shape}")

    gram_loss_fn = GramMatrixConsistencyLoss()
    gram_loss = gram_loss_fn(feature_map_1, feature_map_2)
    print(f"Gram Matrix Consistency Loss: {gram_loss.item()}")
