"""Single-bar progress display (à la onomatron's BenchProgress).

One tqdm bar for the whole run, with the current context (fold / model / trial /
epoch / best metric) in the bar's postfix — instead of stacking several bars
that fight over the terminal.

- TTY (real terminal): an in-place bar on stderr — one line that overwrites
  itself, no scrolling. Redraw is left to tqdm's ``mininterval`` (postfix
  updates use ``refresh=False``), so hooks can call ``context()`` freely.
- non-TTY (pipe, IDE panel, redirected log): no bar frames at all — a throttled
  one-line INFO log carries the same context, so logs stay readable.

The bar writes to stderr (same stream as logging) so ``> run.log 2>&1`` keeps
everything in one file and stdout stays clean for the final table.
"""

from __future__ import annotations

import logging
import sys
import time
from contextlib import suppress
from types import TracebackType
from typing import Any

from tqdm import tqdm

logger = logging.getLogger(__name__)


class TqdmLogHandler(logging.StreamHandler):
    """Route log records through ``tqdm.write``: messages are printed ABOVE the
    active progress bar (which tqdm redraws right after), instead of
    interleaving with bar frames and making the terminal scroll."""

    def emit(self, record: logging.LogRecord) -> None:
        tqdm.write(self.format(record), file=sys.stderr)


def setup_logging(verbose: bool = False, level: int = logging.INFO) -> None:
    """App-level logging setup (called by scripts only, never by libraries).

    - all logs go through :class:`TqdmLogHandler` so they never cut the bar;
    - Optuna's per-trial chatter is silenced (``optuna.logging`` verbosity);
    - unless ``verbose``, the noisy third-party / data-loading loggers are
      raised to WARNING — the ``hexaload.*`` milestone logs stay at INFO.
    """
    handler = TqdmLogHandler()
    handler.setFormatter(logging.Formatter("%(levelname)s %(name)s: %(message)s"))
    logging.root.handlers = [handler]
    logging.root.setLevel(level)
    with suppress(ImportError):
        import optuna

        optuna.logging.set_verbosity(optuna.logging.WARNING)
    if not verbose:
        for name in (
            "mlflow",
            "hexaload.data",
            "hexaload.backtest.load",
            "hexaload.tuning.tracking",
        ):
            logging.getLogger(name).setLevel(logging.WARNING)


class BaseProgress:
    """No-op progress sink (used when no display is wanted, e.g. in tests)."""

    def context(self, **kw: Any) -> None:
        pass

    def advance(self, n: int = 1, **ctx: Any) -> None:
        pass

    def close(self) -> None:
        pass

    def __enter__(self) -> BaseProgress:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()


class NullProgress(BaseProgress):
    """Explicit no-op."""


class Progress(BaseProgress):
    """One tqdm bar for the whole run (TTY) or throttled INFO lines (non-TTY)."""

    def __init__(
        self,
        total: int,
        desc: str = "hexaload",
        unit: str = "step",
        log_every_s: float = 5.0,
    ) -> None:
        self.tty = sys.stderr.isatty()
        self.total = total
        self.n = 0  # own counter: tqdm(disable=True).update() is a no-op
        self.log_every_s = log_every_s
        self._last_log = 0.0
        self._ctx: dict[str, str] = {}
        self.bar = tqdm(
            total=total,
            desc=desc,
            unit=unit,
            file=sys.stderr,
            disable=not self.tty,  # bars only make sense on a real terminal
            mininterval=0.5,
            position=0,  # stays on the first line
            leave=True,
        )

    def __enter__(self) -> Progress:
        return self

    def context(self, **kw: Any) -> None:
        """Update the context shown in the postfix / log line (cheap, no redraw)."""
        self._ctx.update({k: str(v) for k, v in kw.items()})
        if self.tty:
            # refresh=False: let tqdm's mininterval decide when to redraw
            self.bar.set_postfix_str(self._postfix(), refresh=False)
        else:
            self._maybe_log()

    def advance(self, n: int = 1, **ctx: Any) -> None:
        """Move the bar by ``n`` and update the context."""
        self.context(**ctx)
        self.n += n
        self.bar.update(n)
        self._maybe_log()

    def close(self) -> None:
        self.bar.close()

    def _postfix(self) -> str:
        return " ".join(f"{k}={v}" for k, v in self._ctx.items())

    def _maybe_log(self) -> None:
        """Non-TTY fallback: one throttled INFO line with the same context."""
        if self.tty:
            return
        now = time.monotonic()
        if now - self._last_log < self.log_every_s:
            return
        self._last_log = now
        logger.info("progress %d/%d %s", self.n, self.total, self._postfix())
