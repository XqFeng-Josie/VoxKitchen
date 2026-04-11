"""Unit tests for voxkitchen.utils.run_id.generate_run_id."""

from __future__ import annotations

import re

from voxkitchen.utils.run_id import generate_run_id


def test_run_id_is_nonempty_string() -> None:
    rid = generate_run_id()
    assert isinstance(rid, str)
    assert len(rid) > 0


def test_run_id_has_prefix_and_sortable_timestamp() -> None:
    """Format: run-YYYYMMDDTHHMMSS-<4-hex-chars>"""
    rid = generate_run_id()
    assert re.fullmatch(r"run-\d{8}T\d{6}-[0-9a-f]{4}", rid) is not None


def test_run_ids_are_unique_across_calls() -> None:
    ids = {generate_run_id() for _ in range(50)}
    assert len(ids) == 50
