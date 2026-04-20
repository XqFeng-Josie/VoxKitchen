"""Tests for the CommonVoice ingest recipe."""

from __future__ import annotations

from pathlib import Path

from voxkitchen.ingest.recipes import get_recipe


def test_commonvoice_recipe_is_registered() -> None:
    recipe = get_recipe("commonvoice")
    assert recipe.name == "commonvoice"


def test_commonvoice_parses_mock_data(
    mock_commonvoice: Path, tmp_path: Path, make_run_context
) -> None:
    recipe = get_recipe("commonvoice")
    cutset = recipe.prepare(mock_commonvoice, ["train"], make_run_context("ingest", stage_index=0))
    assert len(cutset) == 2
    texts = {c.supervisions[0].text for c in cutset}
    assert "hello world" in texts
    assert "goodbye world" in texts


def test_commonvoice_cuts_have_text_and_language(
    mock_commonvoice: Path, tmp_path: Path, make_run_context
) -> None:
    recipe = get_recipe("commonvoice")
    cutset = recipe.prepare(mock_commonvoice, ["train"], make_run_context("ingest", stage_index=0))
    for cut in cutset:
        sup = cut.supervisions[0]
        assert sup.text is not None
        assert sup.language == "en"
