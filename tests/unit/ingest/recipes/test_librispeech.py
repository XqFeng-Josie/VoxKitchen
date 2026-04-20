"""Tests for the LibriSpeech ingest recipe."""

from __future__ import annotations

from pathlib import Path

import pytest
from voxkitchen.ingest.recipes import get_recipe


def test_librispeech_recipe_is_registered() -> None:
    recipe = get_recipe("librispeech")
    assert recipe.name == "librispeech"


def test_librispeech_parses_mock_data(
    mock_librispeech: Path, tmp_path: Path, make_run_context
) -> None:
    recipe = get_recipe("librispeech")
    cutset = recipe.prepare(
        mock_librispeech, ["train-clean-100"], make_run_context("ingest", stage_index=0)
    )
    assert len(cutset) == 2
    ids = {c.id for c in cutset}
    assert "1089-134686-0001" in ids
    assert "1089-134686-0002" in ids
    texts = {c.supervisions[0].text for c in cutset}
    assert "HELLO WORLD" in texts
    assert "GOODBYE WORLD" in texts


def test_librispeech_cuts_have_speaker(
    mock_librispeech: Path, tmp_path: Path, make_run_context
) -> None:
    recipe = get_recipe("librispeech")
    cutset = recipe.prepare(
        mock_librispeech, ["train-clean-100"], make_run_context("ingest", stage_index=0)
    )
    for cut in cutset:
        assert cut.supervisions[0].speaker == "1089"


def test_librispeech_subset_filter(
    mock_librispeech: Path, tmp_path: Path, make_run_context
) -> None:
    recipe = get_recipe("librispeech")
    with pytest.raises(FileNotFoundError):
        recipe.prepare(mock_librispeech, ["nonexistent"], make_run_context("ingest", stage_index=0))
