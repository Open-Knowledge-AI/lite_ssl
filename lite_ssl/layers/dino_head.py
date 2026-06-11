import torch
import torch.nn as nn

from torch.nn.init import trunc_normal_
from torch.nn.utils.parametrizations import weight_norm


class DINOHead(nn.Module):
    def __init__(
        self,
        in_dim,
        out_dim=2**16,
        use_bn=False,
        nlayers=3,
        hidden_dim=2048,
        bottleneck_dim=256,
        mlp_bias=True,
        use_last_layer=True,
    ):
        super().__init__()
        nlayers = max(nlayers, 1)

        self.use_last_layer = use_last_layer

        self.mlp = _build_mlp(
            nlayers,
            in_dim,
            bottleneck_dim,
            hidden_dim=hidden_dim,
            use_bn=use_bn,
            bias=mlp_bias,
        )

        if use_last_layer:
            self.last_layer = weight_norm(nn.Linear(bottleneck_dim, out_dim, bias=False))
            self.last_layer.parametrizations.weight.original0.data.fill_(1)

    def init_weights(self) -> None:
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x, **kwargs):
        x = self.mlp(x)

        if self.use_last_layer:
            eps = torch.finfo(x.dtype).eps
            x = nn.functional.normalize(x, dim=-1, p=2, eps=eps)
            return self.last_layer(x)
        else:
            return x


def _build_mlp(nlayers, in_dim, bottleneck_dim, hidden_dim=None, use_bn=False, bias=True):
    if nlayers == 1:
        return nn.Linear(in_dim, bottleneck_dim, bias=not use_bn)
    else:
        layers = [nn.Linear(in_dim, hidden_dim, bias=bias)]
        if use_bn:
            layers.append(nn.BatchNorm1d(hidden_dim, track_running_stats=False))
        layers.append(nn.GELU())
        for _ in range(nlayers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim, bias=bias))
            if use_bn:
                layers.append(nn.BatchNorm1d(hidden_dim, track_running_stats=False))
            layers.append(nn.GELU())
        layers.append(nn.Linear(hidden_dim, bottleneck_dim, bias=not use_bn))
        return nn.Sequential(*layers)
