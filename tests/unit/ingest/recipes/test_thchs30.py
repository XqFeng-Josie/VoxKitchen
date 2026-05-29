"""Tests for the THCHS-30 ingest recipe."""

from __future__ import annotations

from pathlib import Path

import pytest
from voxkitchen.ingest.recipes import get_recipe


def test_thchs30_recipe_is_registered() -> None:
    recipe = get_recipe("thchs30")
    assert recipe.name == "thchs30"


def test_thchs30_parses_mock_data(mock_thchs30: Path, make_run_context) -> None:
    recipe = get_recipe("thchs30")
    cutset = recipe.prepare(mock_thchs30, None, make_run_context("ingest", stage_index=0))
    ids = sorted(c.id for c in cutset)
    assert ids == ["A11_0", "A11_1", "B22_0"]


def test_thchs30_speaker_inferred_from_filename(mock_thchs30: Path, make_run_context) -> None:
    """``A11_0.wav`` → speaker ``A11``; ``B22_0.wav`` → speaker ``B22``."""
    recipe = get_recipe("thchs30")
    cutset = recipe.prepare(mock_thchs30, None, make_run_context("ingest", stage_index=0))
    by_id = {c.id: c for c in cutset}
    assert by_id["A11_0"].supervisions[0].speaker == "A11"
    assert by_id["A11_1"].supervisions[0].speaker == "A11"
    assert by_id["B22_0"].supervisions[0].speaker == "B22"


def test_thchs30_text_and_subset(mock_thchs30: Path, make_run_context) -> None:
    """Text is the first line of the .wav.trn; subset is the directory name."""
    recipe = get_recipe("thchs30")
    cutset = recipe.prepare(mock_thchs30, None, make_run_context("ingest", stage_index=0))
    by_id = {c.id: c for c in cutset}
    assert by_id["A11_0"].supervisions[0].text == "你 好 世 界"
    assert by_id["A11_0"].custom["subset"] == "train"
    assert by_id["B22_0"].custom["subset"] == "test"
    assert by_id["B22_0"].supervisions[0].language == "zh"


def test_thchs30_short_trn_tolerated(mock_thchs30: Path, make_run_context) -> None:
    """A 1-line trn (text only, missing pinyin/phonemes) must NOT skip the cut."""
    recipe = get_recipe("thchs30")
    cutset = recipe.prepare(mock_thchs30, None, make_run_context("ingest", stage_index=0))
    by_id = {c.id: c for c in cutset}
    a11_1 = by_id["A11_1"]
    assert a11_1.supervisions[0].text == "再 见"
    # pinyin / phonemes fall back to "" rather than dropping the row.
    assert a11_1.custom["pinyin"] == ""
    assert a11_1.custom["phonemes"] == ""


def test_thchs30_explicit_subset(mock_thchs30: Path, make_run_context) -> None:
    """Asking for only ``train`` yields the two train utts, not the test one."""
    recipe = get_recipe("thchs30")
    cutset = recipe.prepare(mock_thchs30, ["train"], make_run_context("ingest", stage_index=0))
    ids = sorted(c.id for c in cutset)
    assert ids == ["A11_0", "A11_1"]


def test_thchs30_unknown_subset_raises(mock_thchs30: Path, make_run_context) -> None:
    recipe = get_recipe("thchs30")
    with pytest.raises(ValueError, match="unknown subset"):
        recipe.prepare(mock_thchs30, ["bogus"], make_run_context("ingest", stage_index=0))
