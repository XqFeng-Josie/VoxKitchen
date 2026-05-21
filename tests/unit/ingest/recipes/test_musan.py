"""Tests for the MUSAN ingest recipe."""

from __future__ import annotations

from pathlib import Path

import pytest
from voxkitchen.ingest.recipes import get_recipe


def test_musan_recipe_is_registered() -> None:
    recipe = get_recipe("musan")
    assert recipe.name == "musan"


def test_musan_parses_all_categories_by_default(mock_musan: Path, make_run_context) -> None:
    recipe = get_recipe("musan")
    cutset = recipe.prepare(mock_musan, None, make_run_context("ingest", stage_index=0))
    categories = {c.custom["musan_category"] for c in cutset}
    # All three top-level categories show up in the default listing.
    assert categories == {"noise", "music", "speech"}
    assert len(cutset) == 4  # 2 noise + 1 music + 1 speech in the fixture


def test_musan_subset_filter_restricts_to_one_category(mock_musan: Path, make_run_context) -> None:
    """`subsets=['noise']` should restrict to noise utterances only."""
    recipe = get_recipe("musan")
    cutset = recipe.prepare(mock_musan, ["noise"], make_run_context("ingest", stage_index=0))
    assert len(cutset) == 2  # both noise files in the fixture
    for cut in cutset:
        assert cut.custom["musan_category"] == "noise"


def test_musan_records_subcategory_from_directory(mock_musan: Path, make_run_context) -> None:
    """`musan_subcategory` must reflect the immediate parent directory.

    Filtering by subcategory downstream (e.g. only `librivox` background
    speech, not `us-gov`) is a common operator workflow, so the tag must
    survive ingest.
    """
    recipe = get_recipe("musan")
    cutset = recipe.prepare(mock_musan, ["noise"], make_run_context("ingest", stage_index=0))
    subcategories = {c.custom["musan_subcategory"] for c in cutset}
    assert subcategories == {"free-sound", "sound-bible"}


def test_musan_emits_no_supervisions(mock_musan: Path, make_run_context) -> None:
    """MUSAN has no transcripts by design — emit Cuts with an empty supervisions list.

    Downstream operators that need text (ASR, alignment) should not be
    accidentally invoked on MUSAN cuts; consumers like `noise_augment`
    only need the audio.
    """
    recipe = get_recipe("musan")
    cutset = recipe.prepare(mock_musan, None, make_run_context("ingest", stage_index=0))
    for cut in cutset:
        assert cut.supervisions == []


def test_musan_unknown_subset_raises(mock_musan: Path, make_run_context) -> None:
    """A typo in the subset name fails fast with a list of valid options."""
    recipe = get_recipe("musan")
    with pytest.raises(ValueError, match="unknown MUSAN subset"):
        recipe.prepare(mock_musan, ["NOT_A_CATEGORY"], make_run_context("ingest", stage_index=0))


def test_musan_missing_category_dir_silently_skipped(tmp_path: Path, make_run_context) -> None:
    """A partial extract (only `noise/` present) yields whatever exists.

    Users who only need the noise subtree commonly extract just that
    directory; aborting on a missing `music/` would block them.
    """
    import numpy as np
    import soundfile as sf

    ds = tmp_path / "musan"
    noise_dir = ds / "noise" / "free-sound"
    noise_dir.mkdir(parents=True)
    audio = np.sin(np.linspace(0, 1, 16000)).astype(np.float32) * 0.5
    sf.write(noise_dir / "noise-free-sound-0000.wav", audio, 16000)

    recipe = get_recipe("musan")
    cutset = recipe.prepare(tmp_path, None, make_run_context("ingest", stage_index=0))
    # Default subsets include music/speech but they aren't on disk;
    # the recipe returns just the noise cut without raising.
    assert len(cutset) == 1
    assert next(iter(cutset)).custom["musan_category"] == "noise"
