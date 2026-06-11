import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from lite_ssl.layers.dino_head import DINOHead

try:
    from xformers.ops import memory_efficient_attention

    XFORMERS_AVAILABLE = True
except ImportError:
    XFORMERS_AVAILABLE = False


class SingleHeadedAttention(nn.Module):

    def __init__(self, d_model: int, d_head: int = None):
        """
        Single-head cross-attention using xformers if available.

        Args:
            d_model: input/output dim
            d_head: internal attention dim (defaults to d_model)
        """
        super().__init__()

        self.d_model = d_model
        self.d_head = d_head if d_head is not None else d_model

        self.q_proj = nn.Linear(d_model, self.d_head, bias=True)
        self.k_proj = nn.Linear(d_model, self.d_head, bias=True)
        self.v_proj = nn.Linear(d_model, self.d_head, bias=True)

        self.out_proj = nn.Linear(self.d_head, d_model, bias=True)

        self.scale = 1.0 / math.sqrt(self.d_head)

    def forward(self, q_in, kv_in):
        """
        Q_in: (bs, n_g, d_model)
        KV_in: (bs, n_l, d_model)
        mask: optional (not all xformers kernels support arbitrary masks)

        returns: (bs, n_g, d_model)
        """
        q = self.q_proj(q_in)  # (bs, n_g, d_head)
        k = self.k_proj(kv_in)  # (bs, n_l, d_head)
        v = self.v_proj(kv_in)  # (bs, n_l, d_head)

        # Add single head dimension → (bs, n, 1, d_head)
        q = q.unsqueeze(2)
        k = k.unsqueeze(2)
        v = v.unsqueeze(2)

        if XFORMERS_AVAILABLE:
            out = memory_efficient_attention(q, k, v, scale=self.scale)  # (bs, n_g, 1, d_head)
        else:
            # fallback to standard attention
            attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # (bs, n_g, 1, n_l)

            attn_weights = torch.softmax(attn_scores, dim=-1)
            out = torch.matmul(attn_weights, v)  # (bs, n_g, 1, d_head)

        out = out.squeeze(2)  # (bs, n_g, d_head)
        out = self.out_proj(out)  # (bs, n_g, d_model)

        return out


class SoftOMPCrossAttention(nn.Module):

    def __init__(
        self,
        *args,
        temperature=0.1,
        **kwargs,
    ):
        super().__init__()

        self.temperature = temperature

    def forward(self, q_in, kv_in):
        """
        q_in: (bs, n_g, d_model)   → targets (v_j)
        kv_in: (bs, n_l, d_model)  → dictionary (v_i)

        returns:
            recon: (bs, n_g, d_model)
        """
        bs, n_g, d_model = q_in.shape
        n_l = kv_in.shape[1]

        # initialize
        k = F.normalize(kv_in, dim=-1, eps=1e-8)

        recon = torch.zeros(bs, n_g, d_model, device=q_in.device, dtype=q_in.dtype)
        for _ in range(n_l):
            r = q_in - recon  # (bs, n_g, d)

            # ---- soft atom selection ----
            dir_scores = torch.einsum("bgd,bld->bgl", F.normalize(r, dim=-1), k)  # (bs, n_g, n_l)
            dir_weights = torch.einsum("bgd,bld->bgl", r, k)  # (bs, n_g, n_l)
            weights = F.softmax(dir_scores / 0.1, dim=-1) * dir_weights.clamp(min=0.0)
            # (bs, n_g, n_l)

            direction = torch.einsum("bgl,bld->bgd", weights, k)  # (bs, n_g, d)

            # ---- update reconstruction ----
            recon = recon + direction  # (bs, n_g, d)

        return recon


class SoftNonNegativeRecon(nn.Module):

    def __init__(self, temperature: float = 0.1):
        super().__init__()
        self.temperature = temperature

    def forward(self, q_in, kv_in):
        """
        q_in:  (bs, n_g, d_model) → targets
        kv_in: (bs, n_l, d_model) → dictionary

        returns:
            recon: (bs, n_g, d_model)
        """
        q = F.normalize(q_in, dim=-1, eps=1e-8)
        k = F.normalize(kv_in, dim=-1, eps=1e-8)

        scores = torch.einsum("bgd,bld->bgl", q, k)  # (bs, n_g, n_l)
        alpha = F.softplus(scores / self.temperature, beta=10.0)  # (bs, n_g, n_l), non-negative
        recon = torch.einsum("bgl,bld->bgd", alpha, k)  # (bs, n_g, d_model)

        return recon


class LGHead(nn.Module):

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
        only_last_layer=False,
    ):
        super().__init__()
        self.dino_head = DINOHead(
            in_dim=in_dim,
            out_dim=out_dim,
            use_bn=use_bn,
            nlayers=nlayers,
            hidden_dim=hidden_dim,
            bottleneck_dim=bottleneck_dim,
            mlp_bias=mlp_bias,
            use_last_layer=use_last_layer,
            only_last_layer=only_last_layer,
        )

        self.att_head = SoftOMPCrossAttention()

    def forward(self, query, kv):
        """

        Args:
            query: [vg, bs, d_proj]
            kv: [vl, bs, d_model]

        Returns:
            Tuple [proj, recombined_local_proj, logits]
        """
        _n, _b, _ = kv.size()

        projection = self.dino_head.project(kv.flatten(0, 1)).unflatten(
            0, (_n, _b)
        )  # (bs, n_l, bottleneck_dim)
        att_out = self.att_head(query.permute(1, 0, 2), projection.permute(1, 0, 2))
        att_out = att_out.permute(1, 0, 2)  # (bs, n_g, bottleneck_dim)
        logits = self.dino_head.predict(att_out)

        return projection, att_out, logits

    def reconstruct(self, query, kv):
        return self.att_head(query.permute(1, 0, 2), kv.permute(1, 0, 2)).permute(1, 0, 2)

    def project(self, x):
        return self.dino_head.project(x)

    def project_and_predict(self, x):
        proj = self.dino_head.project(x)
        logits = self.dino_head.predict(proj)
        return proj, logits

    def predict_from_projection(self, x):
        return self.dino_head.predict(x)

    def predict(self, x):
        return self.dino_head(x)


if __name__ == "__main__":
    _device = "cuda" if torch.cuda.is_available() else "cpu"

    # Reproducibility
    _torch_seed = 42
    torch.manual_seed(_torch_seed)

    # Dummy dimensions
    _bs = 32
    _n_g = 2
    _n_l = 6

    _d_model = 768
    _d_hidden = 2048
    _d_proj = 64

    # Instantiate model
    _model = LGHead(
        in_dim=_d_model,
        out_dim=2**10,
        use_bn=True,
        bottleneck_dim=_d_proj,
        hidden_dim=_d_hidden,
        use_last_layer=True,
    ).to(_device)

    # Dummy inputs
    _q_in = torch.randn(_n_g, _bs, _d_proj, device=_device)
    _kv_in = torch.randn(_n_l, _bs, _d_model, device=_device)

    # Forward pass
    _target = _model.predict_from_projection(_q_in.flatten(0, 1)).unflatten(0, (_n_g, _bs))
    _proj, _recombined, _logits = _model(_q_in, _kv_in)

    print("Projection shape:", _proj.shape)
    print("Recombined shape:", _recombined.shape)
    print("Logits shape:", _logits.shape)

    # Simple loss to test backward
    _loss = (_target.mean(0) - _logits).square().mean() + (_q_in - _recombined).square().mean()

    print("Loss:", _loss.item())

    # Backward pass
    _loss.backward()

    # Check gradients exist
    _grad_norm = 0.0
    for _name, _param in _model.named_parameters():
        if _param.grad is not None:
            _grad_norm += _param.grad.norm().item()

    print("Total grad norm:", _grad_norm)
