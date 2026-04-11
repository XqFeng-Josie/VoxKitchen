"""Deterministic UTC datetime helpers."""

from __future__ import annotations

from datetime import datetime, timezone


def now_utc() -> datetime:
    """Return the current UTC datetime with tzinfo attached.

    Wrapped in a function so tests can monkeypatch it when determinism matters.
    """
    return datetime.now(tz=timezone.utc)
