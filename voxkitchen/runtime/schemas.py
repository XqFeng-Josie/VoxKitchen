"""Load and cache the merged op_schemas.json.

Read by ``vkit validate`` and ``vkit doctor`` so they can reason about
operators that live in another env and are not importable in the parent.

Lookup order:

1. ``$VKIT_OP_SCHEMAS`` — explicit override (tests, custom layouts)
2. ``/opt/voxkitchen/op_schemas.json`` — Docker default
3. ``None`` — dev mode, caller should fall back to the in-process registry
"""

from __future__ import annotations

import json
import logging
import os
from functools import lru_cache
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

DEFAULT_SCHEMAS_PATH = Path("/opt/voxkitchen/op_schemas.json")


@lru_cache(maxsize=1)
def load_op_schemas() -> dict[str, dict[str, Any]] | None:
    """Return ``{op_name: operator-info-dict}`` or ``None`` if no file is found.

    ``operator-info-dict`` has the shape emitted by
    :mod:`voxkitchen.runtime.dump_schemas`:
    ``{"config_schema": {...}, "required_extras": [...], "device": "...",
    "module": "...", "doc": "..."}``.
    """
    override = os.environ.get("VKIT_OP_SCHEMAS")
    candidates = [Path(override)] if override else []
    candidates.append(DEFAULT_SCHEMAS_PATH)

    for path in candidates:
        if path.is_file():
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except Exception as exc:  # noqa: BLE001 — advisory, fall back
                logger.warning("could not parse %s: %s", path, exc)
                continue
            if not isinstance(data, dict):
                logger.warning("%s is not a JSON object — ignoring", path)
                continue
            return data
    return None


def reset_cache() -> None:
    """Drop cached read. For tests."""
    load_op_schemas.cache_clear()
