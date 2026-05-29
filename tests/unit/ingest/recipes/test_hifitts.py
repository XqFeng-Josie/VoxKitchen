"""Tests for the Hi-Fi TTS ingest recipe."""

from __future__ import annotations

from pathlib import Path

import pytest
from voxkitchen.ingest.recipes import get_recipe


def test_hifitts_recipe_is_registered() -> None:
    recipe = get_recipe("hifitts")
    assert recipe.name == "hifitts"


def test_hifitts_parses_clean_train(mock_hifitts: Path, make_run_context) -> None:
    """Both readers' clean-train manifests are aggregated; the missing-file
    row is silently skipped."""
    recipe = get_recipe("hifitts")
    cutset = recipe.prepare(
        mock_hifitts, ["clean-train"], make_run_context("ingest", stage_index=0)
    )
    ids = sorted(c.id for c in cutset)
    assert ids == ["6097_moby_0001", "6097_moby_0002", "92_alice_0001", "92_alice_0002"]


def test_hifitts_speaker_and_language(mock_hifitts: Path, make_run_context) -> None:
    recipe = get_recipe("hifitts")
    cutset = recipe.prepare(
        mock_hifitts, ["clean-train"], make_run_context("ingest", stage_index=0)
    )
    by_id = {c.id: c for c in cutset}
    assert by_id["92_alice_0001"].supervisions[0].speaker == "92"
    assert by_id["6097_moby_0001"].supervisions[0].speaker == "6097"
    for c in cutset:
        assert c.supervisions[0].language == "en"


def test_hifitts_prefers_normalized_text(mock_hifitts: Path, make_run_context) -> None:
    recipe = get_recipe("hifitts")
    cutset = recipe.prepare(
        mock_hifitts, ["clean-train"], make_run_context("ingest", stage_index=0)
    )
    by_id = {c.id: c for c in cutset}
    # 0001 has text_normalized → that wins.
    assert by_id["92_alice_0001"].supervisions[0].text == "Hello, Alice."
    # 0002 has no text_normalized → falls back to text.
    assert by_id["92_alice_0002"].supervisions[0].text == "goodbye alice"


def test_hifitts_custom_carries_subset_and_reader(mock_hifitts: Path, make_run_context) -> None:
    recipe = get_recipe("hifitts")
    cutset = recipe.prepare(
        mock_hifitts, ["clean-train"], make_run_context("ingest", stage_index=0)
    )
    by_id = {c.id: c for c in cutset}
    assert by_id["92_alice_0001"].custom["subset"] == "clean-train"
    assert by_id["92_alice_0001"].custom["reader_id"] == "92"


def test_hifitts_dev_subset_selectable(mock_hifitts: Path, make_run_context) -> None:
    """Asking only for clean-dev yields just the dev manifest row."""
    recipe = get_recipe("hifitts")
    cutset = recipe.prepare(mock_hifitts, ["clean-dev"], make_run_context("ingest", stage_index=0))
    ids = sorted(c.id for c in cutset)
    assert ids == ["92_alice_dev_0001"]


def test_hifitts_default_subsets(mock_hifitts: Path, make_run_context) -> None:
    """No subset arg → all six subsets walked; only those with manifests yield cuts."""
    recipe = get_recipe("hifitts")
    cutset = recipe.prepare(mock_hifitts, None, make_run_context("ingest", stage_index=0))
    # 4 clean-train + 1 clean-dev = 5; other-* subsets have no manifests in fixture.
    assert len(cutset) == 5


def test_hifitts_unknown_subset_raises(mock_hifitts: Path, make_run_context) -> None:
    recipe = get_recipe("hifitts")
    with pytest.raises(ValueError, match="unknown subset"):
        recipe.prepare(mock_hifitts, ["bogus"], make_run_context("ingest", stage_index=0))
