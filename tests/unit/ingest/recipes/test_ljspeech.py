"""Tests for the LJSpeech ingest recipe."""

from __future__ import annotations

from pathlib import Path

import pytest
from voxkitchen.ingest.recipes import get_recipe


def test_ljspeech_recipe_is_registered() -> None:
    recipe = get_recipe("ljspeech")
    assert recipe.name == "ljspeech"


def test_ljspeech_parses_mock_data(mock_ljspeech: Path, make_run_context) -> None:
    recipe = get_recipe("ljspeech")
    cutset = recipe.prepare(mock_ljspeech, None, make_run_context("ingest", stage_index=0))
    assert len(cutset) == 2
    ids = sorted(c.id for c in cutset)
    assert ids == ["LJ001-0001", "LJ001-0002"]


def test_ljspeech_prefers_normalized_text(mock_ljspeech: Path, make_run_context) -> None:
    """Each cut's supervision must carry the *normalized* (TTS-ready) text."""
    recipe = get_recipe("ljspeech")
    cutset = recipe.prepare(mock_ljspeech, None, make_run_context("ingest", stage_index=0))
    by_id = {c.id: c for c in cutset}

    assert by_id["LJ001-0001"].supervisions[0].text == "Hello world."
    assert by_id["LJ001-0002"].supervisions[0].text == "Hi Mister Smith."


def test_ljspeech_keeps_raw_text_in_custom_when_different(
    mock_ljspeech: Path, make_run_context
) -> None:
    """If normalization actually changed the text, the raw form is preserved."""
    recipe = get_recipe("ljspeech")
    cutset = recipe.prepare(mock_ljspeech, None, make_run_context("ingest", stage_index=0))
    by_id = {c.id: c for c in cutset}

    # Row 1 had identical raw/normalized — `custom` stays empty to avoid noise.
    assert "raw_text" not in by_id["LJ001-0001"].custom
    # Row 2's raw text differs from the normalized form, so we keep it.
    assert by_id["LJ001-0002"].custom["raw_text"] == "Hi Mr. Smith."


def test_ljspeech_tags_speaker_and_language(mock_ljspeech: Path, make_run_context) -> None:
    """All LJSpeech cuts share the single speaker `LJ` and language `en`."""
    recipe = get_recipe("ljspeech")
    cutset = recipe.prepare(mock_ljspeech, None, make_run_context("ingest", stage_index=0))
    for cut in cutset:
        sup = cut.supervisions[0]
        assert sup.speaker == "LJ"
        assert sup.language == "en"


def test_ljspeech_missing_metadata_raises(tmp_path: Path, make_run_context) -> None:
    """A directory without metadata.csv should surface a clear error."""
    (tmp_path / "wavs").mkdir()  # presence of wavs without metadata still errors
    recipe = get_recipe("ljspeech")
    with pytest.raises(FileNotFoundError, match=r"metadata\.csv"):
        recipe.prepare(tmp_path, None, make_run_context("ingest", stage_index=0))
