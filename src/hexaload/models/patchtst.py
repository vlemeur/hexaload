"""PatchTST in pure PyTorch (Nie et al., 2023 — vanilla, channel-independent).

The history window is split into overlapping patches, linearly embedded +
positional encoding, processed by a Transformer encoder; the flattened
patches are concatenated with the future exogenous block and fed to a linear
quantile head -> ``(B,H,Q)``.
"""

from __future__ import annotations

import torch
from torch import nn

from .torch_base import TorchForecaster


class _PatchTSTModule(nn.Module):
    def __init__(
        self,
        n_features: int,
        input_size: int,
        h: int,
        n_quantiles: int,
        patch_len: int = 16,
        stride: int = 8,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
        ff: int = 128,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.h = h
        self.n_quantiles = n_quantiles
        self.patch_len = patch_len
        self.stride = stride
        n_patches = (input_size - patch_len) // stride + 1
        self.n_patches = n_patches
        self.patch_embed = nn.Linear(patch_len * n_features, d_model)
        self.pos = nn.Parameter(torch.zeros(1, n_patches, d_model))
        nn.init.trunc_normal_(self.pos, std=0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=ff,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.head = nn.Linear(n_patches * d_model + h * n_features, h * n_quantiles)

    def forward(self, x_hist: torch.Tensor, x_fut: torch.Tensor) -> torch.Tensor:
        b = x_hist.shape[0]
        # (B,T,F) -> patches (B,N,F,patch_len) -> (B,N,patch_len*F)
        patches = x_hist.unfold(1, self.patch_len, self.stride)
        patches = patches.permute(0, 1, 3, 2).reshape(
            b, self.n_patches, self.patch_len * x_hist.shape[2]
        )
        z = self.patch_embed(patches) + self.pos
        z = self.encoder(z)  # (B,N,d_model)
        out = self.head(torch.cat([z.flatten(1), x_fut.flatten(1)], dim=1))
        return out.view(b, self.h, self.n_quantiles)


class PatchTSTFeatures(TorchForecaster):
    """PatchTST on the feature table (history patches + future exog block)."""

    def __init__(
        self,
        *,
        patch_len: int = 16,
        stride: int = 8,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
        ff: int = 128,
        dropout: float = 0.1,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self._patch = dict(
            patch_len=patch_len,
            stride=stride,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            ff=ff,
            dropout=dropout,
        )

    def _new_module(self, n_features: int) -> nn.Module:
        return _PatchTSTModule(
            n_features=n_features,
            input_size=self._input_size,
            h=self._h,
            n_quantiles=len(self._quantiles),
            **self._patch,
        )
