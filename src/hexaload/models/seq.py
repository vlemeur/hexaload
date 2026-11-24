"""Sequence models in pure PyTorch: LSTM/GRU encoder and TCN encoder.

Both encode the history window ``x_hist (B,T,F)`` into a vector, concatenate
the flattened future exogenous block ``x_fut (B,H,F)``, and emit quantile
forecasts ``(B,H,Q)`` through a shared linear head (pinball-trained).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from .torch_base import TorchForecaster


class _SeqHead(nn.Module):
    """concat(encoded history (B,D), flattened future exog (B,H*F)) -> (B,H,Q)."""

    def __init__(
        self, d: int, h: int, n_features: int, n_quantiles: int, hidden: int = 128
    ) -> None:
        super().__init__()
        self.h = h
        self.n_quantiles = n_quantiles
        self.net = nn.Sequential(
            nn.Linear(d + h * n_features, hidden),
            nn.ReLU(),
            nn.Linear(hidden, h * n_quantiles),
        )

    def forward(self, enc: torch.Tensor, x_fut: torch.Tensor) -> torch.Tensor:
        b = enc.shape[0]
        z = torch.cat([enc, x_fut.flatten(1)], dim=1)
        return self.net(z).view(b, self.h, self.n_quantiles)


class _RNNModule(nn.Module):
    """LSTM/GRU encoder over the history window + shared quantile head."""

    def __init__(
        self,
        n_features: int,
        input_size: int,
        h: int,
        n_quantiles: int,
        rnn: str = "LSTM",
        hidden: int = 128,
        num_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        rnn_cls = nn.LSTM if rnn == "LSTM" else nn.GRU
        self.rnn = rnn_cls(
            n_features,
            hidden,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = _SeqHead(hidden, h, n_features, n_quantiles)

    def forward(self, x_hist: torch.Tensor, x_fut: torch.Tensor) -> torch.Tensor:
        out, _ = self.rnn(x_hist)  # (B,T,hidden)
        return self.head(out[:, -1, :], x_fut)


class _CausalConv1d(nn.Module):
    """Causal conv: left-pad by (kernel-1)*dilation so no future leaks in."""

    def __init__(self, in_ch: int, out_ch: int, kernel: int, dilation: int) -> None:
        super().__init__()
        self.pad = (kernel - 1) * dilation
        self.conv = nn.Conv1d(in_ch, out_ch, kernel, dilation=dilation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B,F,T) -> (B,out,T)
        return self.conv(F.pad(x, (self.pad, 0)))


class _TCNModule(nn.Module):
    """Dilated causal conv stack over the history window + shared quantile head."""

    def __init__(
        self,
        n_features: int,
        input_size: int,
        h: int,
        n_quantiles: int,
        hidden: int = 64,
        n_blocks: int = 3,
        kernel: int = 3,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        for b in range(n_blocks):
            layers.append(
                _CausalConv1d(
                    n_features if b == 0 else hidden, hidden, kernel, dilation=2**b
                )
            )
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
        self.net = nn.Sequential(*layers)
        self.head = _SeqHead(hidden, h, n_features, n_quantiles)

    def forward(self, x_hist: torch.Tensor, x_fut: torch.Tensor) -> torch.Tensor:
        z = self.net(x_hist.transpose(1, 2))  # (B,hidden,T)
        return self.head(z[:, :, -1], x_fut)  # last timestep = encoding


class RNNFeatures(TorchForecaster):
    """LSTM/GRU encoder on the feature table (history + future exog block)."""

    def __init__(
        self,
        *,
        rnn: str = "LSTM",
        hidden: int = 128,
        num_layers: int = 2,
        dropout: float = 0.1,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self._rnn = rnn
        self._hidden = hidden
        self._num_layers = num_layers
        self._dropout = dropout

    def _new_module(self, n_features: int) -> nn.Module:
        return _RNNModule(
            n_features=n_features,
            input_size=self._input_size,
            h=self._h,
            n_quantiles=len(self._quantiles),
            rnn=self._rnn,
            hidden=self._hidden,
            num_layers=self._num_layers,
            dropout=self._dropout,
        )


class TCNFeatures(TorchForecaster):
    """TCN (dilated causal convolutions) on the feature table."""

    def __init__(
        self,
        *,
        hidden: int = 64,
        n_blocks: int = 3,
        kernel: int = 3,
        dropout: float = 0.1,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self._hidden = hidden
        self._n_blocks = n_blocks
        self._kernel = kernel
        self._dropout = dropout

    def _new_module(self, n_features: int) -> nn.Module:
        return _TCNModule(
            n_features=n_features,
            input_size=self._input_size,
            h=self._h,
            n_quantiles=len(self._quantiles),
            hidden=self._hidden,
            n_blocks=self._n_blocks,
            kernel=self._kernel,
            dropout=self._dropout,
        )
