"""Tests for the TED-LIUM 3 ingest recipe."""

from __future__ import annotations

from pathlib import Path

import pytest
from voxkitchen.ingest.recipes import get_recipe


def test_tedlium3_recipe_is_registered() -> None:
    recipe = get_recipe("tedlium3")
    assert recipe.name == "tedlium3"


def test_tedlium3_parses_train_subset(mock_tedlium3: Path, make_run_context) -> None:
    """The train STM has 2 valid utterances; the 2 padding rows must be dropped."""
    recipe = get_recipe("tedlium3")
    cutset = recipe.prepare(mock_tedlium3, ["train"], make_run_context("ingest", stage_index=0))
    assert len(cutset) == 2
    ids = sorted(c.id for c in cutset)
    assert ids == [
        "AaronHuey_2010X-5.00-8.40",
        "AaronHuey_2010X-10.00-14.20",
    ] or ids == sorted(
        [
            "AaronHuey_2010X-5.00-8.40",
            "AaronHuey_2010X-10.00-14.20",
        ]
    )


def test_tedlium3_default_subsets_covers_train_dev_test(
    mock_tedlium3: Path, make_run_context
) -> None:
    """`subsets=None` walks train + dev + test; missing splits are skipped."""
    recipe = get_recipe("tedlium3")
    cutset = recipe.prepare(mock_tedlium3, None, make_run_context("ingest", stage_index=0))
    # 2 utterances in train + 1 utterance in dev (no test split in fixture)
    assert len(cutset) == 3
    talks = {c.custom["talk"] for c in cutset}
    assert talks == {"AaronHuey_2010X", "BenSaunders_2014"}


def test_tedlium3_cut_carries_correct_slice_into_talk(
    mock_tedlium3: Path, make_run_context
) -> None:
    """Each utterance Cut points to a [start, start+duration) slice of the talk."""
    recipe = get_recipe("tedlium3")
    cutset = recipe.prepare(mock_tedlium3, ["train"], make_run_context("ingest", stage_index=0))
    by_id = {c.id: c for c in cutset}

    cut = by_id["AaronHuey_2010X-5.00-8.40"]
    assert cut.start == pytest.approx(5.00)
    assert cut.duration == pytest.approx(3.40)
    sup = cut.supervisions[0]
    assert sup.start == pytest.approx(5.00)
    assert sup.duration == pytest.approx(3.40)
    assert sup.text == "hello world this is a talk"
    assert sup.speaker == "AaronHuey_2010X"
    assert sup.language == "en"


def test_tedlium3_keeps_raw_transcript_tokens(mock_tedlium3: Path, make_run_context) -> None:
    """TED-LIUM noise / disfluency tokens are kept verbatim — not stripped.

    Downstream pipelines that want normalized text should add a
    normalization stage; the recipe shouldn't lossily edit the source.
    """
    recipe = get_recipe("tedlium3")
    cutset = recipe.prepare(mock_tedlium3, ["train"], make_run_context("ingest", stage_index=0))
    texts = {c.supervisions[0].text for c in cutset}
    assert any("{NOISE}" in t for t in texts)


def test_tedlium3_skips_inter_segment_gap_rows(mock_tedlium3: Path, make_run_context) -> None:
    """Padding rows (speaker=inter_segment_gap) are not emitted as Cuts."""
    recipe = get_recipe("tedlium3")
    cutset = recipe.prepare(mock_tedlium3, ["train"], make_run_context("ingest", stage_index=0))
    speakers = {c.supervisions[0].speaker for c in cutset}
    assert "inter_segment_gap" not in speakers


def test_tedlium3_missing_legacy_dir_raises(tmp_path: Path, make_run_context) -> None:
    """A directory that's not an extracted TED-LIUM tree should fail loudly."""
    recipe = get_recipe("tedlium3")
    with pytest.raises(FileNotFoundError, match="legacy"):
        recipe.prepare(tmp_path, None, make_run_context("ingest", stage_index=0))
