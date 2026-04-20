"""Tests for the AISHELL-1 ingest recipe."""

from __future__ import annotations

from pathlib import Path

from voxkitchen.ingest.recipes import get_recipe


def test_aishell_recipe_is_registered() -> None:
    recipe = get_recipe("aishell")
    assert recipe.name == "aishell"


def test_aishell_parses_mock_data(mock_aishell: Path, tmp_path: Path, make_run_context) -> None:
    recipe = get_recipe("aishell")
    cutset = recipe.prepare(mock_aishell, ["train"], make_run_context("ingest", stage_index=0))
    assert len(cutset) == 2


def test_aishell_transcript_joined(mock_aishell: Path, tmp_path: Path, make_run_context) -> None:
    recipe = get_recipe("aishell")
    cutset = recipe.prepare(mock_aishell, ["train"], make_run_context("ingest", stage_index=0))
    texts = {c.supervisions[0].text for c in cutset}
    assert "你好世界" in texts
    assert "再见世界" in texts
    # Ensure space-separated form is NOT present
    assert "你 好 世 界" not in texts
