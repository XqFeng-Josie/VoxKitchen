"""Tests for the AISHELL-3 ingest recipe."""

from __future__ import annotations

from pathlib import Path

from voxkitchen.ingest.recipes import get_recipe


def test_aishell3_recipe_is_registered() -> None:
    recipe = get_recipe("aishell3")
    assert recipe.name == "aishell3"


def test_aishell3_parses_mock_data(mock_aishell3: Path, make_run_context) -> None:
    recipe = get_recipe("aishell3")
    cutset = recipe.prepare(mock_aishell3, ["train"], make_run_context("ingest", stage_index=0))
    ids = sorted(c.id for c in cutset)
    assert ids == ["SSB00050001", "SSB00050002", "SSB00090001"]


def test_aishell3_extracts_characters_not_pinyin_as_text(
    mock_aishell3: Path, make_run_context
) -> None:
    """The supervision text must be the Mandarin characters only.

    Pinyin tokens belong in ``cut.custom["pinyin"]``. Leaking them into
    the text field would break downstream ASR comparison and TTS
    training that expects plain characters.
    """
    recipe = get_recipe("aishell3")
    cutset = recipe.prepare(mock_aishell3, ["train"], make_run_context("ingest", stage_index=0))
    by_id = {c.id: c for c in cutset}

    assert by_id["SSB00050001"].supervisions[0].text == "你好世界"
    assert by_id["SSB00050001"].custom["pinyin"] == "ni3 hao3 shi4 jie4"
    assert by_id["SSB00050002"].supervisions[0].text == "再见"
    assert by_id["SSB00050002"].custom["pinyin"] == "zai4 jian4"


def test_aishell3_tags_speaker_and_language(mock_aishell3: Path, make_run_context) -> None:
    """Speaker comes from the parent directory name; language is always zh."""
    recipe = get_recipe("aishell3")
    cutset = recipe.prepare(mock_aishell3, ["train"], make_run_context("ingest", stage_index=0))
    by_id = {c.id: c for c in cutset}

    assert by_id["SSB00050001"].supervisions[0].speaker == "SSB0005"
    assert by_id["SSB00090001"].supervisions[0].speaker == "SSB0009"
    for cut in cutset:
        assert cut.supervisions[0].language == "zh"


def test_aishell3_gender_from_spk_info(mock_aishell3: Path, make_run_context) -> None:
    """Gender is enriched from spk-info.txt and normalized to schema codes."""
    recipe = get_recipe("aishell3")
    cutset = recipe.prepare(mock_aishell3, ["train"], make_run_context("ingest", stage_index=0))
    by_id = {c.id: c for c in cutset}

    assert by_id["SSB00050001"].supervisions[0].gender == "m"  # "male" → "m"
    assert by_id["SSB00090001"].supervisions[0].gender == "f"  # "female" → "f"


def test_aishell3_missing_subset_is_tolerated(mock_aishell3: Path, make_run_context) -> None:
    """A requested subset that isn't on disk is silently skipped, not raised.

    AISHELL-3 partial extractions (just `train/` or just `test/`) are
    common; aborting on the missing half would block legitimate runs.
    """
    recipe = get_recipe("aishell3")
    # `test/` was never created by the fixture.
    cutset = recipe.prepare(
        mock_aishell3, ["train", "test"], make_run_context("ingest", stage_index=0)
    )
    # Still parses the train cuts.
    assert len(cutset) == 3
