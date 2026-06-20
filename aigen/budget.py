"""Spend guard for AI-gen providers (P5 MVP).

A :class:`BudgetGuard` enforces a per-session and per-day USD cap on
paid AI requests. The whole point is that a fresh install **cannot
spend money by accident**: both caps default to ``0.0`` (read from the
``AIGEN_SESSION_BUDGET_USD`` / ``AIGEN_DAILY_BUDGET_USD`` env vars),
which means *every* request with ``cost > 0`` is rejected with
:class:`BudgetExceeded`. Only the free, local-upscale path (cost ``0``)
runs out of the box. The operator opts into paid spend by exporting a
non-zero budget.

Spend is tracked two ways:

  * **in-process** — a running session total, reset on process restart.
  * **on-disk ledger** — a small JSON file under a scratch/cache dir
    (``AIGEN_LEDGER_PATH`` env override, else ``cache/aigen/ledger.json``
    which is gitignored). The ledger persists *daily* spend across
    restarts so the daily cap survives a server bounce. It is keyed by
    UTC date; stale days are pruned on load.

Nothing here touches the network. ``check()`` is a pure pre-flight
predicate; ``record()`` is called by the caller only *after* a paid
request actually succeeded.
"""
from __future__ import annotations

import json
import logging
import os
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

log = logging.getLogger("psobb_editor.aigen.budget")

# Env var names — single source of truth, also used by tests.
ENV_SESSION_BUDGET = "AIGEN_SESSION_BUDGET_USD"
ENV_DAILY_BUDGET = "AIGEN_DAILY_BUDGET_USD"
ENV_LEDGER_PATH = "AIGEN_LEDGER_PATH"

# Tiny epsilon so float rounding doesn't make an exactly-on-budget
# request spuriously fail.
_EPS = 1e-9


class BudgetExceeded(Exception):
    """Raised by :meth:`BudgetGuard.check` when a request would exceed a cap.

    Carries enough structured context for the HTTP layer to build a
    clear 402/403 message: which cap was hit, the cost, and the
    remaining headroom.
    """

    def __init__(
        self,
        message: str,
        *,
        scope: str,
        cost: float,
        limit: float,
        spent: float,
    ) -> None:
        super().__init__(message)
        self.scope = scope  # "session" | "daily"
        self.cost = cost
        self.limit = limit
        self.spent = spent
        self.remaining = max(0.0, limit - spent)


def _env_float(name: str, default: float = 0.0) -> float:
    """Parse a non-negative float from an env var; default on missing/garbage."""
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        val = float(raw)
    except (TypeError, ValueError):
        log.warning("invalid %s=%r — treating as %.2f", name, raw, default)
        return default
    if val < 0:
        log.warning("%s=%r is negative — clamping to 0", name, raw)
        return 0.0
    return val


def _utc_today() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def _default_ledger_path() -> Path:
    """Resolve the ledger path: env override, else ``cache/aigen/ledger.json``.

    The default lives under ``cache/`` which is gitignored, so the ledger
    is never committed. Resolved relative to this package's repo root.
    """
    env = os.environ.get(ENV_LEDGER_PATH)
    if env:
        return Path(env).expanduser().resolve()
    repo_root = Path(__file__).resolve().parent.parent
    return repo_root / "cache" / "aigen" / "ledger.json"


class BudgetGuard:
    """Session + daily USD spend guard.

    Parameters mirror the env vars but can be passed explicitly (tests do
    this to avoid mutating the process environment). When a parameter is
    ``None`` the corresponding env var is read at construction time.
    """

    def __init__(
        self,
        *,
        session_budget_usd: Optional[float] = None,
        daily_budget_usd: Optional[float] = None,
        ledger_path: Optional[Path] = None,
    ) -> None:
        self.session_budget = (
            _env_float(ENV_SESSION_BUDGET)
            if session_budget_usd is None
            else max(0.0, float(session_budget_usd))
        )
        self.daily_budget = (
            _env_float(ENV_DAILY_BUDGET)
            if daily_budget_usd is None
            else max(0.0, float(daily_budget_usd))
        )
        self._ledger_path = (
            Path(ledger_path) if ledger_path is not None else _default_ledger_path()
        )
        self._lock = threading.Lock()
        self._session_spent = 0.0
        # Daily spend is lazily loaded from the ledger keyed by UTC date.
        self._daily_spent = self._load_daily_spend()

    # ------------------------------------------------------------------ ledger
    def _load_daily_spend(self) -> float:
        """Load today's spend from the JSON ledger; tolerate a missing/corrupt file."""
        today = _utc_today()
        try:
            with open(self._ledger_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError, OSError):
            return 0.0
        if not isinstance(data, dict):
            return 0.0
        days = data.get("days")
        if not isinstance(days, dict):
            return 0.0
        try:
            return float(days.get(today, 0.0))
        except (TypeError, ValueError):
            return 0.0

    def _persist_daily_spend(self) -> None:
        """Write today's running daily total to the ledger (atomic tmp+rename).

        Prunes any day that isn't today so the file can't grow unbounded.
        Best-effort: a write failure logs and is swallowed (the in-process
        totals remain authoritative for the running session).
        """
        today = _utc_today()
        payload = {"schema": 1, "days": {today: round(self._daily_spent, 6)}}
        try:
            self._ledger_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self._ledger_path.with_suffix(".json.tmp")
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2)
            os.replace(tmp, self._ledger_path)
        except OSError as e:
            log.warning("could not persist aigen ledger to %s: %s", self._ledger_path, e)

    # ------------------------------------------------------------------ API
    @property
    def session_spent(self) -> float:
        return self._session_spent

    @property
    def daily_spent(self) -> float:
        return self._daily_spent

    def remaining_session(self) -> float:
        return max(0.0, self.session_budget - self._session_spent)

    def remaining_daily(self) -> float:
        return max(0.0, self.daily_budget - self._daily_spent)

    def check(self, cost: float) -> None:
        """Pre-flight: raise :class:`BudgetExceeded` if ``cost`` won't fit.

        A ``cost <= 0`` request (the free local path) always passes — even
        when both budgets are ``0`` — so a fresh install can still do
        real local work. A ``cost > 0`` request must fit under BOTH the
        session and the daily remaining headroom.
        """
        if cost <= _EPS:
            return
        with self._lock:
            if self._session_spent + cost > self.session_budget + _EPS:
                raise BudgetExceeded(
                    f"request cost ${cost:.4f} exceeds the remaining session budget "
                    f"(${self.remaining_session():.4f} of ${self.session_budget:.2f}). "
                    f"Set {ENV_SESSION_BUDGET} to enable paid generation.",
                    scope="session",
                    cost=cost,
                    limit=self.session_budget,
                    spent=self._session_spent,
                )
            if self._daily_spent + cost > self.daily_budget + _EPS:
                raise BudgetExceeded(
                    f"request cost ${cost:.4f} exceeds the remaining daily budget "
                    f"(${self.remaining_daily():.4f} of ${self.daily_budget:.2f}). "
                    f"Set {ENV_DAILY_BUDGET} to raise the daily cap.",
                    scope="daily",
                    cost=cost,
                    limit=self.daily_budget,
                    spent=self._daily_spent,
                )

    def record(self, cost: float) -> None:
        """Record actual spend after a paid request SUCCEEDED.

        ``cost <= 0`` is a no-op (free path leaves no trace in the ledger).
        Updates the in-process session + daily totals and persists daily
        spend to the ledger.
        """
        if cost <= _EPS:
            return
        with self._lock:
            self._session_spent += cost
            self._daily_spent += cost
            self._persist_daily_spend()

    def snapshot(self) -> dict:
        """Return a JSON-friendly view of the current budget state."""
        return {
            "session_budget_usd": self.session_budget,
            "daily_budget_usd": self.daily_budget,
            "session_spent_usd": round(self._session_spent, 6),
            "daily_spent_usd": round(self._daily_spent, 6),
            "session_remaining_usd": round(self.remaining_session(), 6),
            "daily_remaining_usd": round(self.remaining_daily(), 6),
            "generation_enabled": self.session_budget > 0 and self.daily_budget > 0,
            "ledger_path": str(self._ledger_path),
        }


__all__ = [
    "BudgetGuard",
    "BudgetExceeded",
    "ENV_SESSION_BUDGET",
    "ENV_DAILY_BUDGET",
    "ENV_LEDGER_PATH",
]
