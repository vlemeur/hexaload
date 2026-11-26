"""Model registry + Optuna search spaces (one entry per backtest model name).

``build_model(name, params)`` instantiates a model exactly as run_backtest.py
does, but with tuned params; ``suggest_params(name, trial)`` proposes a point in
the search space; ``default_params(name)`` is the fallback (constructor defaults).
"""

from __future__ import annotations

from typing import Any

from hexaload.models import (
    NHITS,
    BiasEWMA,
    ChronosBoltTSFM,
    EnsembleFeatures,
    HGBFeatures,
    MLPFeatures,
    MondrianConformal,
    PatchTSTFeatures,
    RidgeFeatures,
    RNNFeatures,
    SktimeAutoETS,
    SktimeTheta,
    TCNFeatures,
    TimestralLoRA,
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
TSFM_MODELS = ("timestral_209m", "timestral_lora_209m", "chronos_bolt")
# bonus bricks (v2): extra backtest rows built from the feature models above.
# They need the feature table; --drop-features applies to them too (norte run).
VARIANT_MODELS = (
    "ens_features",  # online convex ensemble hgb + xgb + ridge
    "hgb_features_ewma",  # EWMA bias correction on hgb
    "hgb_features_mondrian",  # per-local-hour conformal intervals on hgb
    "hgb_features_ewma_mondrian",  # both: Mondrian conformal around EWMA-corrected hgb
)


def build_variant(model: str, model_params: dict[str, dict] | None = None) -> Any:
    """Instantiate a bonus-brick backtest row, reusing tuned member params."""
    params = model_params or {}
    if model == "ens_features":
        members = ("hgb_features", "xgb_features", "ridge_features")
        return EnsembleFeatures([build_model(m, params.get(m)) for m in members])

    def _mk(base: str, suffix: str):
        bp = dict(params.get(base) or {})
        w = {k: bp.pop(k) for k in WRAPPER_KEYS if k in bp}
        inner = build_model(base, bp)
        vh = int(w.get("wrap_val_hours", 24 * 14))
        if suffix == "_ewma_mondrian":
            return MondrianConformal(
                BiasEWMA(inner, val_hours=vh, alpha=float(w.get("ewma_alpha", 0.1))),
                val_hours=vh,
                min_bucket=int(w.get("mondrian_min_bucket", 7)),
            )
        if suffix == "_ewma":
            return BiasEWMA(inner, val_hours=vh, alpha=float(w.get("ewma_alpha", 0.1)))
        return MondrianConformal(
            inner, val_hours=vh, min_bucket=int(w.get("mondrian_min_bucket", 7))
        )

    for suffix in ("_ewma_mondrian", "_ewma", "_mondrian"):
        if model.endswith(suffix):
            return _mk(model[: -len(suffix)], suffix)
    raise ValueError(f"unknown variant {model!r}")


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


# hyperparameters of the post-hoc wrappers (EWMA bias + Mondrian conformal),
# tuned alongside the base model; consumed by build_variant / the tuning
# objective, never passed to the base model constructor
WRAPPER_KEYS = ("ewma_alpha", "wrap_val_hours", "mondrian_min_bucket")


def _normalize_params(params: dict[str, Any]) -> dict[str, Any]:
    """Decode tuple params stored as "a-b-c" strings in the best-params YAML
    (Optuna categoricals must be scalars, so tuples are string-encoded)."""
    p = dict(params)
    for k in WRAPPER_KEYS:
        p.pop(k, None)
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
    if model == "timestral_lora_209m":
        # not in TUNABLE: constructor defaults (lora_r/lr/n_steps/...), like baselines
        return TimestralLoRA("mistralai/Timestral-209M", **p)
    if model == "chronos_bolt":
        return ChronosBoltTSFM("amazon/chronos-bolt-base", **p)
    raise ValueError(f"unknown model {model!r}")


def _wrapper_space(trial: Any) -> dict[str, Any]:
    """Search space of the post-hoc wrappers (EWMA bias + Mondrian conformal)."""
    return {
        "ewma_alpha": trial.suggest_float("ewma_alpha", 0.02, 0.5, log=True),
        "wrap_val_hours": trial.suggest_categorical(
            "wrap_val_hours", [24 * 7, 24 * 14, 24 * 30]
        ),
        "mondrian_min_bucket": trial.suggest_int("mondrian_min_bucket", 5, 14),
    }


def suggest_params(model: str, trial: Any) -> dict[str, Any]:
    """One Optuna point in the search space of ``model``."""
    if model == "ridge_features":
        return {
            "alpha": trial.suggest_float("alpha", 1e-3, 100.0, log=True),
            **_wrapper_space(trial),
        }
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
            **_wrapper_space(trial),
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
            **_wrapper_space(trial),
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
