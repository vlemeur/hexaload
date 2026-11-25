"""Model registry + Optuna search spaces (one entry per backtest model name).

``build_model(name, params)`` instantiates a model exactly as run_backtest.py
does, but with tuned params; ``suggest_params(name, trial)`` proposes a point in
the search space; ``default_params(name)`` is the fallback (constructor defaults).
"""

from __future__ import annotations

from typing import Any

from hexaload.models import (
    NHITS,
    ChronosBoltTSFM,
    HGBFeatures,
    MLPFeatures,
    PatchTSTFeatures,
    RidgeFeatures,
    RNNFeatures,
    SktimeAutoETS,
    SktimeTheta,
    TCNFeatures,
    TimestralTSFM,
    XGBFeatures,
)

# models with a meaningful search space (others run on constructor defaults)
TUNABLE = (
    "ridge_features",
    "hgb_features",
    "xgb_features",
    "mlp_features",
    "rnn_features",
    "tcn_features",
    "nhits",
    "patchtst_features",
    "timestral_209m",
    "chronos_bolt",
)

# model groups — how each family is fed in the backtest loop
FEATURE_MODELS = (
    "ridge_features",
    "hgb_features",
    "xgb_features",
    "mlp_features",
    "rnn_features",
    "tcn_features",
    "patchtst_features",
)
UNIVARIATE_MODELS = ("sktime_autoets", "sktime_theta", "nhits")
TSFM_MODELS = ("timestral_209m", "chronos_bolt")
ALL_MODEL_NAMES = FEATURE_MODELS + UNIVARIATE_MODELS + TSFM_MODELS
TORCH_MODELS = (
    "mlp_features",
    "rnn_features",
    "tcn_features",
    "nhits",
    "patchtst_features",
)


def default_params(model: str) -> dict[str, Any]:
    """Empty dict = constructor defaults (the honest baseline)."""
    return {}


def _normalize_params(params: dict[str, Any]) -> dict[str, Any]:
    """Decode tuple params stored as "a-b-c" strings in the best-params YAML
    (Optuna categoricals must be scalars, so tuples are string-encoded)."""
    p = dict(params)
    for key in ("hidden", "pool_rates"):
        v = p.get(key)
        if isinstance(v, str):
            p[key] = tuple(int(x) for x in v.split("-"))
    return p


def build_model(model: str, params: dict[str, Any] | None = None) -> Any:
    """Instantiate a backtest model by name, with (possibly tuned) params."""
    p = _normalize_params(params or {})
    if model == "ridge_features":
        return RidgeFeatures(**p)
    if model == "hgb_features":
        return HGBFeatures(**p)
    if model == "xgb_features":
        return XGBFeatures(**p)
    if model == "mlp_features":
        return MLPFeatures(**p)
    if model == "rnn_features":
        return RNNFeatures(**p)
    if model == "tcn_features":
        return TCNFeatures(**p)
    if model == "nhits":
        return NHITS(**p)
    if model == "patchtst_features":
        return PatchTSTFeatures(**p)
    if model == "sktime_autoets":
        return SktimeAutoETS(**p)
    if model == "sktime_theta":
        return SktimeTheta(**p)
    if model == "timestral_209m":
        return TimestralTSFM("mistralai/Timestral-209M", **p)
    if model == "chronos_bolt":
        return ChronosBoltTSFM("amazon/chronos-bolt-base", **p)
    raise ValueError(f"unknown model {model!r}")


def suggest_params(model: str, trial: Any) -> dict[str, Any]:
    """One Optuna point in the search space of ``model``."""
    if model == "ridge_features":
        return {"alpha": trial.suggest_float("alpha", 1e-3, 100.0, log=True)}
    if model == "hgb_features":
        return {
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
            "max_leaf_nodes": trial.suggest_int("max_leaf_nodes", 15, 63),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 10, 60),
            "l2_regularization": trial.suggest_float(
                "l2_regularization", 1e-3, 10.0, log=True
            ),
            # capped: OMP_NUM_THREADS=1 in this process (anti-segfault with
            # torch/libomp), so each fit is single-threaded — keep trees cheap,
            # early stopping does the rest
            "max_iter": trial.suggest_int("max_iter", 150, 600),
        }
    if model == "xgb_features":
        return {
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 8),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
            "n_estimators": trial.suggest_int("n_estimators", 200, 800),
            "early_stopping_rounds": trial.suggest_int("early_stopping_rounds", 20, 60),
        }
    if model == "mlp_features":
        hidden = tuple(
            int(v)
            for v in trial.suggest_categorical(
                "hidden", ["128-64", "256-128", "256-128-64"]
            ).split("-")
        )
        return {
            "hidden": hidden,
            "dropout": trial.suggest_float("dropout", 0.0, 0.3),
            "lr": trial.suggest_float("lr", 3e-4, 3e-3, log=True),
            "batch_size": trial.suggest_categorical("batch_size", [256, 512, 1024]),
            "input_size": trial.suggest_categorical(
                "input_size", [24 * 7, 24 * 14, 24 * 28]
            ),
        }
    if model == "rnn_features":
        return {
            "rnn": trial.suggest_categorical("rnn", ["LSTM", "GRU"]),
            "hidden": trial.suggest_categorical("hidden", [64, 128, 256]),
            "num_layers": trial.suggest_int("num_layers", 1, 3),
            "dropout": trial.suggest_float("dropout", 0.0, 0.3),
            "lr": trial.suggest_float("lr", 3e-4, 3e-3, log=True),
            "batch_size": trial.suggest_categorical("batch_size", [256, 512, 1024]),
            "input_size": trial.suggest_categorical(
                "input_size", [24 * 7, 24 * 14, 24 * 28]
            ),
        }
    if model == "tcn_features":
        return {
            "hidden": trial.suggest_categorical("hidden", [32, 64, 128]),
            "n_blocks": trial.suggest_int("n_blocks", 2, 5),
            "kernel": trial.suggest_categorical("kernel", [3, 5]),
            "dropout": trial.suggest_float("dropout", 0.0, 0.3),
            "lr": trial.suggest_float("lr", 3e-4, 3e-3, log=True),
            "batch_size": trial.suggest_categorical("batch_size", [256, 512, 1024]),
            "input_size": trial.suggest_categorical(
                "input_size", [24 * 7, 24 * 14, 24 * 28]
            ),
        }
    if model == "nhits":
        pool_rates = tuple(
            int(v)
            for v in trial.suggest_categorical(
                "pool_rates", ["1-2-4-8", "1-2-4", "1-2-4-8-16"]
            ).split("-")
        )
        return {
            "n_stacks": trial.suggest_int("n_stacks", 2, 4),
            "pool_rates": pool_rates,
            "hidden": trial.suggest_categorical("hidden", [64, 128, 256]),
            "n_layers": trial.suggest_int("n_layers", 1, 3),
            "lr": trial.suggest_float("lr", 3e-4, 3e-3, log=True),
            "batch_size": trial.suggest_categorical("batch_size", [256, 512, 1024]),
            "input_size": trial.suggest_categorical(
                "input_size", [24 * 7, 24 * 14, 24 * 28]
            ),
        }
    if model == "patchtst_features":
        return {
            "patch_len": trial.suggest_categorical("patch_len", [8, 16, 24]),
            "stride": trial.suggest_categorical("stride", [4, 8, 12]),
            "d_model": trial.suggest_categorical("d_model", [32, 64, 128]),
            "n_heads": trial.suggest_categorical("n_heads", [2, 4, 8]),
            "n_layers": trial.suggest_int("n_layers", 1, 3),
            "lr": trial.suggest_float("lr", 3e-4, 3e-3, log=True),
            "batch_size": trial.suggest_categorical("batch_size", [256, 512, 1024]),
            "input_size": trial.suggest_categorical(
                "input_size", [24 * 7, 24 * 14, 24 * 28]
            ),
        }
    if model in ("timestral_209m", "chronos_bolt"):
        return {
            "context_hours": trial.suggest_categorical(
                "context_hours", [24 * 30, 24 * 90, 24 * 180, 24 * 365]
            )
        }
    # not tunable (baselines, RTE, ETS/Theta): constructor defaults
    return {}
