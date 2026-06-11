import torch
import torch.nn as nn
import torch.distributed as dist


class SIGReg(nn.Module):
    """
    Distributed, cached implementation of SIGReg.

    Enforces Gaussianization of embeddings using sliced
    characteristic function matching.
    """

    def __init__(
        self,
        num_slices=4096,
        t_min=-3.0,
        t_max=3.0,
        num_t=5,
        eps=1e-6,
    ):
        super().__init__()

        self.num_slices = num_slices
        self.num_t = num_t
        self.eps = eps

        # ---- cached buffers (initialized lazily) ----
        self.register_buffer("_A", None, persistent=False)
        self.register_buffer("_t", None, persistent=False)
        self.register_buffer("_gauss_cf", None, persistent=False)
        self._cached_step = None

        self.t_min = t_min
        self.t_max = t_max

    # -------------------------------------------------
    # utilities
    # -------------------------------------------------

    def _build_cache(self, num_channels, device, global_step):
        """
        Build and cache slice directions and CF grids.
        Must be called on all ranks with same global_step.
        """

        if self._cached_step == global_step:
            return

        # ---- deterministic slice sampling ----
        g = torch.Generator(device=device)
        g.manual_seed(global_step)

        A = torch.randn(
            num_channels,
            self.num_slices,
            generator=g,
            device=device,
        )
        A = A / (A.norm(dim=0, keepdim=True) + self.eps)

        # ---- characteristic function grid ----
        t = torch.linspace(
            self.t_min,
            self.t_max,
            self.num_t,
            device=device,
        )

        # CF of N(0,1)
        gauss_cf = torch.exp(-0.5 * t**2)

        # ---- cache ----
        self._A = A
        self._t = t
        self._gauss_cf = gauss_cf
        self._cached_step = global_step

    @classmethod
    def _all_reduce_mean(cls, x):
        if not dist.is_initialized():
            return x
        dist.all_reduce(x, op=dist.ReduceOp.SUM)
        return x / dist.get_world_size()

    # -------------------------------------------------
    # forward
    # -------------------------------------------------
    def forward(self, x, global_step, *ignored_args, **ignored_kwargs):
        """
        Args:
            x: Tensor of shape (num_samples, num_channels)
            global_step: int, must be synchronized across ranks

        Returns:
            scalar SIGReg loss
        """
        *_, num_samples, dim = x.shape

        device = x.device

        # ---- build cache if needed ----
        self._build_cache(x.shape[-1], device, global_step)

        rand = self._A  # (dim, M)
        t = self._t  # (T,)
        gauss_cf = self._gauss_cf  # (T,)

        # -------------------------------------------------
        # 1. random projections (slices)
        # -------------------------------------------------
        proj = x @ rand  # (num_samples, M)

        # -------------------------------------------------
        # 2. empirical characteristic function
        # -------------------------------------------------
        # shape: (num_samples, M, T)
        x_t = proj.unsqueeze(-1) * t

        ecf = torch.exp(1j * x_t).mean(dim=-3)  # (M, T)

        # -------------------------------------------------
        # 3. distributed aggregation
        # -------------------------------------------------
        ecf = self._all_reduce_mean(ecf)

        # -------------------------------------------------
        # 4. Epps–Pulley type statistic
        # -------------------------------------------------
        diff = (ecf - gauss_cf).abs().square()
        weighted = diff * gauss_cf

        # integrate over t
        stat_per_slice = torch.trapz(weighted, t, dim=-1)

        # scale by global num_samples
        if dist.is_initialized():
            num_samples_global = num_samples * dist.get_world_size()
        else:
            num_samples_global = num_samples

        loss = stat_per_slice.mean() * num_samples_global

        return loss
