"""Shared pytest fixtures for voxkitchen tests."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest


@pytest.fixture
def fixed_datetime() -> datetime:
    """A deterministic UTC datetime for reproducible tests."""
    return datetime(2026, 4, 11, 10, 30, 0, tzinfo=timezone.utc)


@pytest.fixture
def tmp_jsonl_gz(tmp_path: Path) -> Path:
    """Return a path to a temporary .jsonl.gz file inside tmp_path."""
    return tmp_path / "cuts.jsonl.gz"
