"""Tests for the AISHELL-1 ingest recipe."""

from __future__ import annotations

from pathlib import Path

from voxkitchen.ingest.recipes import get_recipe
from voxkitchen.pipeline.context import RunContext


def _make_ctx(tmp_path: Path) -> RunContext:
    return RunContext(
        work_dir=tmp_path,
        pipeline_run_id="run-test",
        stage_index=0,
        stage_name="ingest",
        num_gpus=0,
        num_cpu_workers=1,
        gc_mode="aggressive",
        device="cpu",
    )


def test_aishell_recipe_is_registered() -> None:
    recipe = get_recipe("aishell")
    assert recipe.name == "aishell"


def test_aishell_parses_mock_data(mock_aishell: Path, tmp_path: Path) -> None:
    recipe = get_recipe("aishell")
    cutset = recipe.prepare(mock_aishell, ["train"], _make_ctx(tmp_path))
    assert len(cutset) == 2


def test_aishell_transcript_joined(mock_aishell: Path, tmp_path: Path) -> None:
    recipe = get_recipe("aishell")
    cutset = recipe.prepare(mock_aishell, ["train"], _make_ctx(tmp_path))
    texts = {c.supervisions[0].text for c in cutset}
    assert "你好世界" in texts
    assert "再见世界" in texts
    # Ensure space-separated form is NOT present
    assert "你 好 世 界" not in texts
