"""Train the production Timestral LoRA adapter on all data up to a freeze.

Much bigger than the per-fold backtest row: multi-year training window,
month-stratified window sampling, validation tail with best-checkpoint restore.
Writes ``adapter.safetensors`` + ``adapter_config.json`` + a README model card
under ``--out``, optionally pushes them to a Hugging Face repo (``--push``),
and writes the git pointer ``docs/data/lora_adapter.json`` used by the serving
code to pin the adapter version.

    uv run python scripts/train_lora.py --n-steps 3000 --out data/lora/run1
    uv run python scripts/train_lora.py --n-steps 3000 --push \
        --hf-repo you/timestral-209m-lora-fr-load

On an Apple Silicon Mac the pipe runs on MPS (float64 scaler casts are
downcast to float32 by hexaload.models._mps_shim). Needs HF_TOKEN only with
--push. Never run this in the daily forecast Action: it is a rare, heavy job.

Memory: training activations scale with batch x (1 + n_exo_rows) x context
patches. On a ~20 GiB MPS pool the proven fit is batch=2 with
train-context-hours=720 (the backtest row's config); batch=4 or 90-day
contexts OOM. More steps over more years (stratified sampling) matter far
more than a longer context.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
from pathlib import Path

import pandas as pd

from hexaload.backtest.load import exogenous_columns, load_features, load_targets
from hexaload.models.tsfm_lora import (
    TimestralLoRA,
    finetune_lora,
    inject_lora,
    save_adapter,
)
from hexaload.progress import Progress, setup_logging

logger = logging.getLogger("hexaload.train_lora")

README_TEMPLATE = """---
license: other
license_name: modified-mit-derivative
tags: [time-series, forecasting, lora, timestral, hexaload, rte-eco2mix]
---

# {hf_repo} — Timestral-209M LoRA adapter (French national electricity load)

LoRA adapter (r={r}, alpha={alpha}) fine-tuned on the RTE eco2mix hourly
consommation series (hexaload benchmark data), base `mistralai/Timestral-209M`
@ `{base_rev}`.

- sha256(adapter.safetensors): `{sha}`
- trained on {n_hours} h up to {freeze} (seed {seed}), val tail {val_h} h,
  best step {best_step} (val pinball {best_val:.1f}), {n_steps} steps
- hyperparams: lr={lr}, batch={batch}, train_context_hours={tch}, horizon={h}

License: derivative of the Timestral weights — Modified MIT (see the base
repo), which caps commercial use for organisations above the revenue threshold.
"""


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo", default="mistralai/Timestral-209M")
    p.add_argument("--device", default=None, help="cpu|cuda|mps (default: auto)")
    p.add_argument("--n-steps", type=int, default=3000)
    p.add_argument(
        "--batch",
        type=int,
        default=2,
        help="2 fits a ~20 GiB MPS pool with 37 exo rows; 4 OOMs",
    )
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=float, default=32.0)
    p.add_argument(
        "--train-context-hours",
        type=int,
        default=24 * 30,
        help="720 h is the proven MPS fit; 90 d OOMs at batch 2+ on ~20 GiB",
    )
    p.add_argument("--horizon", type=int, default=24)
    p.add_argument("--val-tail-hours", type=int, default=24 * 30)
    p.add_argument("--ckpt-every", type=int, default=200)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--freeze", default=None, help="as-of UTC; default = data max")
    p.add_argument("--out", default="data/lora/timestral-209m-lora")
    p.add_argument("--push", action="store_true", help="upload to --hf-repo")
    p.add_argument("--hf-repo", default=None)
    p.add_argument("--hf-private", action="store_true")
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()
    setup_logging(verbose=args.verbose)

    y, _ref, _wx, _src = load_targets()
    X, manifest = load_features()
    exo = exogenous_columns(manifest)
    y = y.interpolate(method="time").bfill().ffill()
    freeze = pd.Timestamp(args.freeze, tz="UTC") if args.freeze else y.index.max()
    y = y.loc[:freeze]
    X_exo = (
        X[exo].reindex(y.index).interpolate(method="time").bfill().ffill().fillna(0.0)
    )
    logger.info(
        "train on %d h up to %s | exo %d cols | device=%s",
        len(y),
        freeze,
        len(exo),
        args.device or "auto",
    )

    wrapper = TimestralLoRA(args.repo, device=args.device)
    wrapper._load()
    pipe = wrapper._pipe
    n_inj = inject_lora(pipe.model, args.lora_r, args.lora_alpha)
    pipe.model.to(pipe.device)
    logger.info("injected %d LoRA layers (r=%d) on %s", n_inj, args.lora_r, pipe.device)

    with Progress(args.n_steps, desc="lora-train", unit="step") as prog:
        hist = finetune_lora(
            pipe,
            y,
            X_exo,
            lr=args.lr,
            n_steps=args.n_steps,
            batch=args.batch,
            train_context_hours=args.train_context_hours,
            horizon=args.horizon,
            seed=args.seed,
            val_tail_hours=args.val_tail_hours,
            ckpt_every=args.ckpt_every,
            stratify_by_month=True,
            progress=prog,
        )

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    adapter = out / "adapter.safetensors"
    save_adapter(pipe.model, adapter)
    sha = hashlib.sha256(adapter.read_bytes()).hexdigest()
    best_val = float("nan")
    for s, v in reversed(hist["val"]):
        if s == hist["best_step"]:
            best_val = v
            break
    cfg = {
        "base_repo": args.repo,
        "adapter_file": adapter.name,
        "sha256": sha,
        "hyperparams": {
            "lora_r": args.lora_r,
            "lora_alpha": args.lora_alpha,
            "lr": args.lr,
            "n_steps": args.n_steps,
            "batch": args.batch,
            "train_context_hours": args.train_context_hours,
            "horizon": args.horizon,
            "seed": args.seed,
            "stratify_by_month": True,
        },
        "train": {"n_hours": int(len(y)), "freeze": str(freeze)},
        "val": {
            "tail_hours": args.val_tail_hours,
            "best_step": hist["best_step"],
            "best_val": best_val,
        },
        "license": (
            "derivative of mistralai/Timestral-209M weights (Modified MIT) — "
            "see the base repo"
        ),
    }
    (out / "adapter_config.json").write_text(json.dumps(cfg, indent=2) + "\n")
    logger.info("wrote %s (sha256 %s…) + adapter_config.json", adapter, sha[:12])

    rev = None
    if args.push:
        if not args.hf_repo:
            p.error("--push needs --hf-repo")
        from huggingface_hub import HfApi

        api = HfApi()
        api.create_repo(args.hf_repo, exist_ok=True, private=args.hf_private)
        base_rev = api.model_info(args.repo).sha
        readme = out / "README.md"
        readme.write_text(
            README_TEMPLATE.format(
                hf_repo=args.hf_repo,
                r=args.lora_r,
                alpha=args.lora_alpha,
                base_rev=base_rev,
                sha=sha,
                n_hours=len(y),
                freeze=freeze,
                seed=args.seed,
                val_h=args.val_tail_hours,
                best_step=hist["best_step"],
                best_val=best_val,
                n_steps=args.n_steps,
                lr=args.lr,
                batch=args.batch,
                tch=args.train_context_hours,
                h=args.horizon,
            )
        )
        for f in (adapter, out / "adapter_config.json", readme):
            api.upload_file(
                path_or_fileobj=str(f), path_in_repo=f.name, repo_id=args.hf_repo
            )
        rev = api.repo_info(args.hf_repo).sha
        pointer = {
            "adapter_repo_id": args.hf_repo,
            "adapter_revision": rev,
            "base_repo": args.repo,
            "base_revision": base_rev,
            "sha256": sha,
        }
        Path("docs/data/lora_adapter.json").write_text(
            json.dumps(pointer, indent=2) + "\n"
        )
        logger.info(
            "pushed to hf.co/%s @ %s + git pointer docs/data/lora_adapter.json",
            args.hf_repo,
            rev,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
