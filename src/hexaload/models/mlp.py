"""MLP forecaster on sliding windows (pure PyTorch).

Input: flattened history window + flattened future exogenous block ->
multi-quantile head (pinball). The only from-scratch model that consumes the
future exogenous block directly, so it is the deep counterpart of the GBMs.
"""

from __future__ import annotations

import torch
from torch import nn

from .torch_base import TorchForecaster


class _MLPModule(nn.Module):
    def __init__(
        self,
        n_features: int,
        input_size: int,
        h: int,
        n_quantiles: int,
        hidden: tuple[int, ...] = (256, 128),
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.h = h
        self.n_quantiles = n_quantiles
        in_dim = (input_size + h) * n_features
        layers: list[nn.Module] = []
        prev = in_dim
        for width in hidden:
            layers += [nn.Linear(prev, width), nn.ReLU(), nn.Dropout(dropout)]
            prev = width
        self.body = nn.Sequential(*layers)
        self.head = nn.Linear(prev, h * n_quantiles)

    def forward(self, x_hist: torch.Tensor, x_fut: torch.Tensor) -> torch.Tensor:
        # (B,T,F), (B,H,F) -> (B,H,Q)
        b = x_hist.shape[0]
        z = torch.cat([x_hist.flatten(1), x_fut.flatten(1)], dim=1)
        return self.head(self.body(z)).view(b, self.h, self.n_quantiles)


class MLPFeatures(TorchForecaster):
    """Windowed MLP on the feature table (history + future exog block)."""

    def __init__(
        self,
        *,
        hidden: tuple[int, ...] = (256, 128),
        dropout: float = 0.1,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self._hidden = hidden
        self._dropout = dropout

    def _new_module(self, n_features: int) -> nn.Module:
        return _MLPModule(
            n_features=n_features,
            input_size=self._input_size,
            h=self._h,
            n_quantiles=len(self._quantiles),
            hidden=self._hidden,
            dropout=self._dropout,
        )
