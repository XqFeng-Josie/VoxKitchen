"""Generate short, sortable, deterministically-formatted pipeline run IDs."""

from __future__ import annotations

import secrets
from datetime import datetime, timezone


def generate_run_id() -> str:
    """Return a fresh run id like ``run-20260411T103000-a1b2``.

    Format:
    - ``run-`` prefix for greppability
    - Compact ISO-8601 UTC timestamp (no punctuation) for sortability
    - 4-hex-character random suffix to disambiguate runs within the same second
    """
    now = datetime.now(tz=timezone.utc)
    ts = now.strftime("%Y%m%dT%H%M%S")
    suffix = secrets.token_hex(2)
    return f"run-{ts}-{suffix}"
