from functools import partial

import torch
import torch.nn as nn

from einops import rearrange
from torch.nn.init import trunc_normal_


def _make_lna_block(input_dim, output_dim, bias, norm_op, activation):
    layers = [nn.Linear(input_dim, output_dim, bias=bias)]
    if norm_op is not None:
        layers.append(norm_op(output_dim))
    if activation is not None:
        layers.append(activation())
    return nn.Sequential(*layers)


def _build_projector(n_layers, in_dim, out_dim, hidden_dim, activation=nn.GELU):
    norm_op = partial(nn.BatchNorm1d, track_running_stats=False)
    if n_layers > 1:
        layers = _make_lna_block(in_dim, hidden_dim, True, norm_op, activation)
        for _ in range(n_layers - 2):
            layers += _make_lna_block(hidden_dim, hidden_dim, True, norm_op, activation)
        layers += nn.Sequential(*[nn.Linear(hidden_dim, out_dim, bias=False), norm_op(out_dim)])
        return nn.Sequential(*layers)
    else:
        layers = [nn.Linear(in_dim, out_dim, bias=False), norm_op(out_dim)]
        return nn.Sequential(*layers)


def _build_predictor(n_layers, in_out_dim, bottleneck_dim, activation=nn.GELU):
    norm_op = partial(nn.BatchNorm1d, track_running_stats=False)
    layers = [_make_lna_block(in_out_dim, bottleneck_dim, True, norm_op, activation)]

    for _ in range(n_layers - 1):
        layers += _make_lna_block(bottleneck_dim, bottleneck_dim, True, norm_op, activation)

    layers += _make_lna_block(bottleneck_dim, in_out_dim, False, None, None)
    return nn.Sequential(*layers)


class CVAHead(nn.Module):
    def __init__(
        self,
        in_dim,
        out_dim=1024,
        projector_layers=3,
        predictor_layers=1,
        hidden_dim=2048,
        bottleneck_dim=256,
        act_op=nn.GELU,
        use_predictor=True,
    ):
        super().__init__()
        projector_layers = max(projector_layers, 1)

        self.projector = _build_projector(
            projector_layers,
            in_dim,
            out_dim,
            hidden_dim=hidden_dim,
            activation=act_op,
        )

        if use_predictor:
            self.predictor = _build_predictor(
                predictor_layers,
                out_dim,
                bottleneck_dim,
                activation=act_op,
            )

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def project(self, latent):
        if latent.ndim == 2:
            return self.projector(latent)

        if latent.ndim == 4:
            # spatial_latent: (B, C, H, W)
            b, _, h, w = latent.shape
            flattened_latent = rearrange(latent, "b c h w -> (b h w) c").contiguous()

            proj = self.projector(flattened_latent)

            # make it spatial again
            return rearrange(proj, "(b h w) c -> b c h w", b=b, h=h, w=w).contiguous()

        if latent.ndim == 3:
            # (B, N, C)
            b, n, _ = latent.shape

            return self.projector(latent.flatten(0, 1)).unflatten(0, (b, n))

        raise ValueError(f"{latent.ndim=}D latent input is not supported")

    def predict(self, latent):
        if latent.ndim == 2:
            return self.predictor(self.projector(latent))

        if latent.ndim == 4:
            # spatial_latent: (B, C, H, W)
            b, _, h, w = latent.shape
            flattened_latent = rearrange(latent, "b c h w -> (b h w) c").contiguous()

            projection = self.projector(flattened_latent)
            pred = self.predictor(projection)

            # make it spatial again
            return rearrange(pred, "(b h w) c -> b c h w", b=b, h=h, w=w).contiguous()

        if latent.ndim == 3:
            # (B, N, C)
            b, n, _ = latent.shape
            return self.predictor(self.projector(latent.flatten(0, 1))).unflatten(0, (b, n))

        raise ValueError(f"{latent.ndim=}D latent input is not supported")

    def project_predict(self, latent):
        projected = self.project(latent)
        predicted = self.predictor(projected)
        return projected, predicted

    def forward(self, latent, project_only=False):
        if project_only:
            return self.project(latent)

        return self.predict(latent)


class IdentityHead(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def project(self, x):
        return x

    def predict(self, x):
        return x

    def project_predict(self, x):
        return x, x

    def forward(self, x, **kwargs):
        return x


class CVAHeadList(torch.nn.Module):
    def __init__(self, num_scales=2, **params):
        super().__init__()
        self.heads = torch.nn.ModuleList([CVAHead(**params) for _ in range(num_scales)])

    def forward(self, x, scale_idx, project_only=False):
        return self.heads[scale_idx](x, project_only=project_only)


if __name__ == "__main__":
    model = CVAHead(
        768,
        512,
        hidden_dim=2048,
        bottleneck_dim=256,
        act_op=nn.GELU,
    )
    print(model)
    x = torch.randn(2, 36, 768)
    out = model(x, project_only=True)

    print("Output shape:", out.shape)  # Expected: (2, 2048, 6, 6)
    out2 = model(x, project_only=False)
    print("Output shape after prediction:", out2.shape)  # Expected: (2, 2048, 6, 6)
