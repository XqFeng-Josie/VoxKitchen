"""Tests for the Thorsten-Voice ingest recipe."""

from __future__ import annotations

from pathlib import Path

from voxkitchen.ingest.recipes import get_recipe


def test_thorsten_recipe_is_registered() -> None:
    recipe = get_recipe("thorsten_voice")
    assert recipe.name == "thorsten_voice"


def test_thorsten_parses_mock_data(mock_thorsten_voice: Path, make_run_context) -> None:
    recipe = get_recipe("thorsten_voice")
    cutset = recipe.prepare(mock_thorsten_voice, None, make_run_context("ingest", stage_index=0))
    ids = sorted(c.id for c in cutset)
    assert ids == ["sample_0001", "sample_0002"]


def test_thorsten_prefers_normalized_text(mock_thorsten_voice: Path, make_run_context) -> None:
    recipe = get_recipe("thorsten_voice")
    cutset = recipe.prepare(mock_thorsten_voice, None, make_run_context("ingest", stage_index=0))
    by_id = {c.id: c for c in cutset}
    # Row 1: raw == normalized, so normalized wins (they're equal).
    assert by_id["sample_0001"].supervisions[0].text == "Hallo Welt."
    # Row 2: normalized expands "Dr." → "Doktor"; the recipe must use that.
    assert by_id["sample_0002"].supervisions[0].text == "Hi Doktor Meier."


def test_thorsten_raw_text_preserved_in_custom(mock_thorsten_voice: Path, make_run_context) -> None:
    """When raw and normalized differ, raw is kept in cut.custom['raw_text']."""
    recipe = get_recipe("thorsten_voice")
    cutset = recipe.prepare(mock_thorsten_voice, None, make_run_context("ingest", stage_index=0))
    by_id = {c.id: c for c in cutset}
    # Row 1 raw == normalized — no need to preserve.
    assert "raw_text" not in by_id["sample_0001"].custom
    # Row 2 differs — raw_text preserved.
    assert by_id["sample_0002"].custom.get("raw_text") == "Hi Dr. Meier."


def test_thorsten_speaker_language_gender(mock_thorsten_voice: Path, make_run_context) -> None:
    """Mono-speaker corpus: single speaker label + de + male voice."""
    recipe = get_recipe("thorsten_voice")
    cutset = recipe.prepare(mock_thorsten_voice, None, make_run_context("ingest", stage_index=0))
    for c in cutset:
        sup = c.supervisions[0]
        assert sup.speaker == "thorsten"
        assert sup.language == "de"
        assert sup.gender == "m"


def test_thorsten_missing_metadata_raises(tmp_path: Path, make_run_context) -> None:
    """Empty root → FileNotFoundError with a clear hint to extract the tarball."""
    import pytest

    recipe = get_recipe("thorsten_voice")
    with pytest.raises(FileNotFoundError):
        recipe.prepare(tmp_path, None, make_run_context("ingest", stage_index=0))
