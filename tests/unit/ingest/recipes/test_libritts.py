"""Tests for the LibriTTS ingest recipe."""

from __future__ import annotations

from pathlib import Path

import pytest
from voxkitchen.ingest.recipes import get_recipe


def test_libritts_recipe_is_registered() -> None:
    recipe = get_recipe("libritts")
    assert recipe.name == "libritts"


def test_libritts_parses_mock_data(mock_libritts: Path, make_run_context) -> None:
    recipe = get_recipe("libritts")
    cutset = recipe.prepare(
        mock_libritts, ["train-clean-100"], make_run_context("ingest", stage_index=0)
    )
    ids = sorted(c.id for c in cutset)
    assert ids == ["1089_134686_000001_000000", "2289_200000_000001_000000"]


def test_libritts_prefers_normalized_when_both_present(
    mock_libritts: Path, make_run_context
) -> None:
    """When *.normalized.txt and *.original.txt both exist, normalized wins.

    The TTS-friendly normalization (case, punctuation) is the whole point
    of LibriTTS relative to plain LibriSpeech.
    """
    recipe = get_recipe("libritts")
    cutset = recipe.prepare(
        mock_libritts, ["train-clean-100"], make_run_context("ingest", stage_index=0)
    )
    by_id = {c.id: c for c in cutset}
    assert by_id["1089_134686_000001_000000"].supervisions[0].text == "Hello, world."


def test_libritts_falls_back_to_original_text(mock_libritts: Path, make_run_context) -> None:
    """When only *.original.txt exists, that text is used (no skipping)."""
    recipe = get_recipe("libritts")
    cutset = recipe.prepare(
        mock_libritts, ["train-clean-100"], make_run_context("ingest", stage_index=0)
    )
    by_id = {c.id: c for c in cutset}
    assert by_id["2289_200000_000001_000000"].supervisions[0].text == "Goodbye world."


def test_libritts_speaker_chapter_and_language(mock_libritts: Path, make_run_context) -> None:
    """Speaker / chapter / language are sourced from the directory layout."""
    recipe = get_recipe("libritts")
    cutset = recipe.prepare(
        mock_libritts, ["train-clean-100"], make_run_context("ingest", stage_index=0)
    )
    by_id = {c.id: c for c in cutset}

    cut1 = by_id["1089_134686_000001_000000"]
    assert cut1.supervisions[0].speaker == "1089"
    assert cut1.supervisions[0].language == "en"
    assert cut1.custom["chapter"] == "134686"
    assert cut1.custom["subset"] == "train-clean-100"


def test_libritts_gender_from_speakers_tsv(mock_libritts: Path, make_run_context) -> None:
    """Gender is enriched from speakers.tsv and normalized to schema codes."""
    recipe = get_recipe("libritts")
    cutset = recipe.prepare(
        mock_libritts, ["train-clean-100"], make_run_context("ingest", stage_index=0)
    )
    by_id = {c.id: c for c in cutset}
    assert by_id["1089_134686_000001_000000"].supervisions[0].gender == "f"
    assert by_id["2289_200000_000001_000000"].supervisions[0].gender == "m"


def test_libritts_discovery_skips_non_subset_dirs(mock_libritts: Path, make_run_context) -> None:
    """Auto-discovery (no `subsets` arg) only picks official subset names."""
    # Plant a sibling directory that isn't a known LibriTTS subset.
    (mock_libritts / "LibriTTS" / "extra_notes").mkdir()
    recipe = get_recipe("libritts")
    cutset = recipe.prepare(mock_libritts, None, make_run_context("ingest", stage_index=0))
    # We still find the two utterances under train-clean-100; the sibling
    # directory is ignored.
    assert len(cutset) == 2


def test_libritts_missing_subset_raises(mock_libritts: Path, make_run_context) -> None:
    """An explicitly requested subset that doesn't exist surfaces a clear error.

    LibriTTS subsets are independently downloadable, but asking for one
    you didn't fetch is a configuration mistake worth raising on (unlike
    AISHELL-3 where partial extracts are routine).
    """
    recipe = get_recipe("libritts")
    with pytest.raises(FileNotFoundError):
        recipe.prepare(
            mock_libritts, ["train-other-500"], make_run_context("ingest", stage_index=0)
        )
