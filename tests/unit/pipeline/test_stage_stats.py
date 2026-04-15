"""Unit tests for stage execution statistics (_stats.json)."""

from __future__ import annotations

import json
from pathlib import Path


def test_stats_json_written_after_stage(tmp_path: Path) -> None:
    """Verify that _write_stage_stats produces valid JSON with required keys."""
    from voxkitchen.pipeline.runner import _write_stage_stats

    stage_dir = tmp_path / "01_vad"
    stage_dir.mkdir()

    _write_stage_stats(
        stage_dir=stage_dir,
        stage_name="vad",
        operator="silero_vad",
        wall_time=12.345,
        cuts_in=100,
        cuts_out=847,
    )

    stats_path = stage_dir / "_stats.json"
    assert stats_path.exists()
    stats = json.loads(stats_path.read_text())
    assert stats["stage_name"] == "vad"
    assert stats["operator"] == "silero_vad"
    assert stats["wall_time_seconds"] == 12.35  # rounded to 2 decimals
    assert stats["cuts_in"] == 100
    assert stats["cuts_out"] == 847


def test_stats_json_includes_throughput(tmp_path: Path) -> None:
    from voxkitchen.pipeline.runner import _write_stage_stats

    stage_dir = tmp_path / "01_vad"
    stage_dir.mkdir()

    _write_stage_stats(
        stage_dir=stage_dir,
        stage_name="vad",
        operator="silero_vad",
        wall_time=10.0,
        cuts_in=100,
        cuts_out=200,
    )

    stats = json.loads((stage_dir / "_stats.json").read_text())
    assert stats["throughput_cuts_per_sec"] == 20.0  # 200 / 10


def test_stats_json_zero_time_no_division_error(tmp_path: Path) -> None:
    from voxkitchen.pipeline.runner import _write_stage_stats

    stage_dir = tmp_path / "01_fast"
    stage_dir.mkdir()

    _write_stage_stats(
        stage_dir=stage_dir,
        stage_name="fast",
        operator="identity",
        wall_time=0.0,
        cuts_in=5,
        cuts_out=5,
    )

    stats = json.loads((stage_dir / "_stats.json").read_text())
    assert stats["throughput_cuts_per_sec"] == 0.0
