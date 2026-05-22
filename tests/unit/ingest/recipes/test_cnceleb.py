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
    # 3 fixture files in data/, spread across 2 speakers.
    assert len(cutset) == 3
    speakers = {c.supervisions[0].speaker for c in cutset}
    assert speakers == {"id00000", "id00001"}


def test_cnceleb_dev_subset_expands_speaker_ids_to_utterances(
    mock_cnceleb: Path, make_run_context
) -> None:
    """`subsets=['dev']` reads speaker IDs from dev.lst, then walks data/<spk>/.

    Real-data verification of the recipe against the OpenSLR/82 tarball
    showed that ``dev/dev.lst`` lists speaker IDs (``id00000``), not
    utterance paths. The recipe expands each ID to that speaker's data/
    utterances. The fixture lists only id00000 in dev.lst, so the 2
    utterances under data/id00000/ are returned and the 1 utterance
    under data/id00001/ is filtered out.
    """
    recipe = get_recipe("cnceleb")
    cutset = recipe.prepare(mock_cnceleb, ["dev"], make_run_context("ingest", stage_index=0))
    assert len(cutset) == 2
    for cut in cutset:
        assert cut.supervisions[0].speaker == "id00000"
        assert cut.custom["subset"] == "dev"


def test_cnceleb_eval_subset_walks_separate_enrol_and_test_dirs(
    mock_cnceleb: Path, make_run_context
) -> None:
    """`subsets=['eval']` walks eval/enroll/ + eval/test/.

    Those directories hold FLAC files that are **separate** from data/
    — they are the verification trial recordings, not pointers into
    data/. The recipe walks the directories directly because the
    accompanying .lst files spell paths with ``.wav`` while the on-disk
    files are ``.flac``.
    """
    recipe = get_recipe("cnceleb")
    cutset = recipe.prepare(mock_cnceleb, ["eval"], make_run_context("ingest", stage_index=0))
    # 2 enroll + 2 test in the fixture.
    assert len(cutset) == 4
    for cut in cutset:
        assert cut.custom["subset"] == "eval"


def test_cnceleb_eval_speaker_extracted_from_filename_prefix(
    mock_cnceleb: Path, make_run_context
) -> None:
    """eval FLACs sit in flat enroll/ and test/ dirs; speaker is in the filename.

    ``eval/enroll/id00800-enroll.flac`` → speaker is ``id00800``;
    ``eval/test/id00800-singing-01-001.flac`` → also ``id00800``. The
    parent directory name is ``enroll`` or ``test``, not a speaker id.
    """
    recipe = get_recipe("cnceleb")
    cutset = recipe.prepare(mock_cnceleb, ["eval"], make_run_context("ingest", stage_index=0))
    speakers = {c.supervisions[0].speaker for c in cutset}
    assert speakers == {"id00800", "id00801"}


def test_cnceleb_speaker_and_language_tags(mock_cnceleb: Path, make_run_context) -> None:
    """Language is hard-coded zh; supervision text stays empty."""
    recipe = get_recipe("cnceleb")
    cutset = recipe.prepare(mock_cnceleb, None, make_run_context("ingest", stage_index=0))
    for cut in cutset:
        sup = cut.supervisions[0]
        assert sup.speaker.startswith("id000"), f"unexpected speaker: {sup.speaker!r}"
        assert sup.language == "zh"
        assert sup.text == ""


def test_cnceleb_overlap_in_subsets_deduplicates(mock_cnceleb: Path, make_run_context) -> None:
    """Passing both ['data', 'dev'] must NOT double-count dev utterances.

    dev is a filter on the data/ audio (same FLACs), so a user who
    asks for both legitimately wants the union, which is just data.
    """
    recipe = get_recipe("cnceleb")
    cutset = recipe.prepare(
        mock_cnceleb, ["data", "dev"], make_run_context("ingest", stage_index=0)
    )
    assert len(cutset) == 3  # still 3 unique data/ utterances


def test_cnceleb_data_plus_eval_combines_disjoint_audio(
    mock_cnceleb: Path, make_run_context
) -> None:
    """data + eval are disjoint audio sets — should be additive.

    data/ has 3 utterances; eval/ has 4 separate audio files. The
    combined CutSet should have 7 cuts.
    """
    recipe = get_recipe("cnceleb")
    cutset = recipe.prepare(
        mock_cnceleb, ["data", "eval"], make_run_context("ingest", stage_index=0)
    )
    assert len(cutset) == 7


def test_cnceleb_unknown_subset_raises(mock_cnceleb: Path, make_run_context) -> None:
    """A typo'd subset value fails fast."""
    recipe = get_recipe("cnceleb")
    with pytest.raises(ValueError, match="unknown CN-Celeb subset"):
        recipe.prepare(mock_cnceleb, ["train"], make_run_context("ingest", stage_index=0))


def test_cnceleb_handles_missing_dev_lst(tmp_path: Path, make_run_context) -> None:
    """`subsets=['dev']` on a data-only tree (no dev/) returns empty.

    Partial extracts shouldn't abort when the user happens to ask for a
    missing split — they should produce a zero-Cut CutSet so downstream
    stages see "no data" and behave sensibly.
    """
    import numpy as np
    import soundfile as sf

    spk = tmp_path / "CN-Celeb_flac" / "data" / "id00000"
    spk.mkdir(parents=True)
    audio = np.sin(np.linspace(0, 1, 16000)).astype(np.float32) * 0.5
    sf.write(spk / "vlog-01-001.flac", audio, 16000, format="FLAC")

    recipe = get_recipe("cnceleb")
    cutset = recipe.prepare(tmp_path, ["dev"], make_run_context("ingest", stage_index=0))
    assert len(cutset) == 0
