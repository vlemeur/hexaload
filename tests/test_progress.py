"""Tests for the single-bar progress display (non-TTY path in pytest)."""

from __future__ import annotations

import logging

import pytest

from hexaload.progress import NullProgress, Progress


def test_null_progress_is_noop():
    with NullProgress() as p:
        p.context(model="x", fold="2025-01-01")
        p.advance(3, trial=2)
        p.close()


def test_progress_counts_even_when_disabled(capsys):
    # pytest stderr is not a TTY -> bar disabled, but the counter still advances
    with Progress(total=4, desc="t", unit="fold", log_every_s=0.0) as p:
        assert p.tty is False
        p.context(fold="2025-01-01", model="hgb_features")
        p.advance(1, model="hgb_features")
        p.advance(1, model="xgb_features")
        assert p.n == 2
        assert p.total == 4
    assert p.n == 2  # closed cleanly


def test_tqdm_log_handler_routes_through_tqdm_write(monkeypatch):
    from hexaload import progress as progress_mod

    calls: list[str] = []
    monkeypatch.setattr(
        progress_mod.tqdm, "write", lambda msg, file=None: calls.append(msg)
    )
    handler = progress_mod.TqdmLogHandler()
    handler.setFormatter(logging.Formatter("%(levelname)s %(name)s: %(message)s"))
    rec = logging.LogRecord("x", logging.INFO, __file__, 1, "hello bar", None, None)
    handler.emit(rec)
    assert calls == ["INFO x: hello bar"]


def test_setup_logging_installs_tqdm_handler_and_quiets_optuna():
    from hexaload.progress import TqdmLogHandler, setup_logging

    root = logging.getLogger()
    old_handlers, old_level = root.handlers, root.level
    try:
        setup_logging(verbose=False)
        assert any(isinstance(h, TqdmLogHandler) for h in root.handlers)
        assert root.level == logging.INFO
        optuna = pytest.importorskip("optuna")
        assert optuna.logging.get_verbosity() == optuna.logging.WARNING
        assert logging.getLogger("mlflow").level == logging.WARNING
    finally:
        root.handlers, root.level = old_handlers, old_level
