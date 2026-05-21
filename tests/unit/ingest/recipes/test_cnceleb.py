"""Tests for the CN-Celeb 1 ingest recipe."""

from __future__ import annotations

from pathlib import Path

import pytest
from voxkitchen.ingest.recipes import get_recipe


def test_cnceleb_recipe_is_registered() -> None:
    recipe = get_recipe("cnceleb")
    assert recipe.name == "cnceleb"


def test_cnceleb_data_subset_walks_all_utterances(mock_cnceleb: Path, make_run_context) -> None:
    """`subsets=None` (defaults to ['data']) yields every FLAC under data/."""
    recipe = get_recipe("cnceleb")
    cutset = recipe.prepare(mock_cnceleb, None, make_run_context("ingest", stage_index=0))
    # 3 fixture files spread across 2 speakers.
    assert len(cutset) == 3
    speakers = {c.supervisions[0].speaker for c in cutset}
    assert speakers == {"id00000", "id00001"}


def test_cnceleb_dev_subset_reads_lst_file(mock_cnceleb: Path, make_run_context) -> None:
    """`subsets=['dev']` only emits the utterances listed in dev/dev.lst."""
    recipe = get_recipe("cnceleb")
    cutset = recipe.prepare(mock_cnceleb, ["dev"], make_run_context("ingest", stage_index=0))
    assert len(cutset) == 1
    assert next(iter(cutset)).id == "interview-01-001"
    assert next(iter(cutset)).custom["subset"] == "dev"


def test_cnceleb_eval_subset_merges_enroll_and_test_lists(
    mock_cnceleb: Path, make_run_context
) -> None:
    """`subsets=['eval']` concatenates enroll.lst + test.lst.

    Also exercises the recipe's tolerance for lines with or without the
    .flac suffix: the fixture's enroll.lst has it, test.lst doesn't.
    """
    recipe = get_recipe("cnceleb")
    cutset = recipe.prepare(mock_cnceleb, ["eval"], make_run_context("ingest", stage_index=0))
    # 1 line in enroll.lst + 1 line in test.lst
    assert len(cutset) == 2
    for cut in cutset:
        assert cut.custom["subset"] == "eval"


def test_cnceleb_speaker_and_language_tags(mock_cnceleb: Path, make_run_context) -> None:
    """Speaker is the parent dir name; language is hard-coded zh."""
    recipe = get_recipe("cnceleb")
    cutset = recipe.prepare(mock_cnceleb, None, make_run_context("ingest", stage_index=0))
    for cut in cutset:
        sup = cut.supervisions[0]
        assert sup.speaker.startswith("id000"), f"unexpected speaker: {sup.speaker!r}"
        assert sup.language == "zh"
        # CN-Celeb is non-transcribed; supervision text stays empty.
        assert sup.text == ""


def test_cnceleb_overlap_in_subsets_deduplicates(mock_cnceleb: Path, make_run_context) -> None:
    """Passing both ['data', 'dev'] must NOT double-count the dev utterance.

    'data' is a superset of 'dev', and the user may legitimately ask for
    both (e.g. tagging which utterances are in the dev split). The recipe
    deduplicates on the FLAC path so the same audio doesn't appear twice.
    """
    recipe = get_recipe("cnceleb")
    cutset = recipe.prepare(
        mock_cnceleb, ["data", "dev"], make_run_context("ingest", stage_index=0)
    )
    # Still 3 distinct utterances, no dup.
    assert len(cutset) == 3


def test_cnceleb_unknown_subset_raises(mock_cnceleb: Path, make_run_context) -> None:
    """A typo'd subset value fails fast."""
    recipe = get_recipe("cnceleb")
    with pytest.raises(ValueError, match="unknown CN-Celeb subset"):
        recipe.prepare(mock_cnceleb, ["train"], make_run_context("ingest", stage_index=0))


def test_cnceleb_handles_missing_lst_file(tmp_path: Path, make_run_context) -> None:
    """`subsets=['dev']` on a tree without `dev/dev.lst` returns empty.

    Partial extracts that only carry `data/` shouldn't abort when the
    user happens to ask for a missing split — they should produce a
    zero-Cut CutSet so downstream stages see "no data" and behave
    sensibly.
    """
    import numpy as np
    import soundfile as sf

    # Build a data-only tree (no dev/ or eval/ directories).
    spk = tmp_path / "CN-Celeb_flac" / "data" / "id00000"
    spk.mkdir(parents=True)
    audio = np.sin(np.linspace(0, 1, 16000)).astype(np.float32) * 0.5
    sf.write(spk / "vlog-01-001.flac", audio, 16000, format="FLAC")

    recipe = get_recipe("cnceleb")
    cutset = recipe.prepare(tmp_path, ["dev"], make_run_context("ingest", stage_index=0))
    assert len(cutset) == 0
