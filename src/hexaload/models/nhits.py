"""N-HiTS in pure PyTorch (Oreshkin et al., 2022 — simplified identity basis).

Stacks of blocks, each block pools the input at its own rate (multi-resolution),
predicts a backcast (subtracted from the input for the next block — double
residual) and a forecast head; the final forecast is the sum over blocks.
Univariate: consumes only the ``__y__`` column of the history window. The
forecast head emits ``h * Q`` values -> quantile outputs ``(B,H,Q)``.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from .torch_base import TorchForecaster


class _NHITSBlock(nn.Module):
    def __init__(
        self,
        input_size: int,
        h: int,
        n_quantiles: int,
        pool: int,
        hidden: int,
        n_layers: int,
    ) -> None:
        super().__init__()
        self.pool = pool
        self.input_size = input_size
        t_pooled = max(1, input_size // pool)
        layers: list[nn.Module] = [nn.Linear(t_pooled, hidden), nn.ReLU()]
        for _ in range(n_layers - 1):
            layers += [nn.Linear(hidden, hidden), nn.ReLU()]
        self.mlp = nn.Sequential(*layers)
        self.backcast_head = nn.Linear(hidden, t_pooled)
        self.forecast_head = nn.Linear(hidden, h * n_quantiles)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # x: (B,T) -> pooled (B,T//pool)
        xp = F.avg_pool1d(x.unsqueeze(1), self.pool).squeeze(1)
        z = self.mlp(xp)
        back = self.backcast_head(z)  # (B, T//pool)
        fore = self.forecast_head(z)  # (B, h*Q)
        back_up = F.interpolate(
            back.unsqueeze(1), scale_factor=self.pool, mode="nearest"
        ).squeeze(1)[..., : self.input_size]
        return x - back_up, fore


class _NHITSModule(nn.Module):
    def __init__(
        self,
        n_features: int,
        input_size: int,
        h: int,
        n_quantiles: int,
        n_stacks: int = 3,
        pool_rates: tuple[int, ...] = (1, 2, 4, 8),
        hidden: int = 128,
        n_layers: int = 2,
    ) -> None:
        super().__init__()
        self.h = h
        self.n_quantiles = n_quantiles
        self.blocks = nn.ModuleList(
            _NHITSBlock(input_size, h, n_quantiles, p, hidden, n_layers)
            for _s in range(n_stacks)
            for p in pool_rates
        )

    def forward(self, x_hist: torch.Tensor, x_fut: torch.Tensor) -> torch.Tensor:
        b = x_hist.shape[0]
        x = x_hist[..., -1]  # univariate: the __y__ column only
        out = torch.zeros(b, self.h * self.n_quantiles, device=x.device)
        for blk in self.blocks:
            x, fore = blk(x)
            out = out + fore
        return out.view(b, self.h, self.n_quantiles)


class NHITS(TorchForecaster):
    """N-HiTS on the target series (univariate; ignores the exogenous block)."""

    def __init__(
        self,
        *,
        n_stacks: int = 3,
        pool_rates: tuple[int, ...] = (1, 2, 4, 8),
        hidden: int = 128,
        n_layers: int = 2,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self._n_stacks = n_stacks
        self._pool_rates = pool_rates
        self._hidden = hidden
        self._n_layers = n_layers

    def _new_module(self, n_features: int) -> nn.Module:
        return _NHITSModule(
            n_features=n_features,
            input_size=self._input_size,
            h=self._h,
            n_quantiles=len(self._quantiles),
            n_stacks=self._n_stacks,
            pool_rates=self._pool_rates,
            hidden=self._hidden,
            n_layers=self._n_layers,
        )
