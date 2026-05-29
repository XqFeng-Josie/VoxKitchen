"""Tests for the LibriTTS-R ingest recipe.

LibriTTS-R reuses the entire LibriTTS parser via subclassing and only
overrides the top-level directory name and download URLs. These tests
confirm registration, that the ``LibriTTS_R/`` layout is honoured, and
that the provenance tag distinguishes a LibriTTS-R Cut from a LibriTTS
Cut at the data level.
"""

from __future__ import annotations

from pathlib import Path

from voxkitchen.ingest.recipes import get_recipe
from voxkitchen.ingest.recipes.libritts import LibriTTSRecipe
from voxkitchen.ingest.recipes.libritts_r import LibriTTSRRecipe


def test_libritts_r_recipe_is_registered() -> None:
    recipe = get_recipe("libritts_r")
    assert recipe.name == "libritts_r"
    # The subclass relationship is contract — we promise to track the
    # LibriTTS parsing rules. Drift from that is a behaviour change.
    assert isinstance(recipe, LibriTTSRecipe)
    assert isinstance(recipe, LibriTTSRRecipe)


def test_libritts_r_parses_mock_data(mock_libritts_r: Path, make_run_context) -> None:
    recipe = get_recipe("libritts_r")
    cutset = recipe.prepare(
        mock_libritts_r, ["train-clean-100"], make_run_context("ingest", stage_index=0)
    )
    ids = sorted(c.id for c in cutset)
    assert ids == ["1089_134686_000001_000000", "2289_200000_000001_000000"]


def test_libritts_r_prefers_normalized_text(mock_libritts_r: Path, make_run_context) -> None:
    """Same normalized > original preference as LibriTTS — proves we
    inherit the parser unchanged."""
    recipe = get_recipe("libritts_r")
    cutset = recipe.prepare(
        mock_libritts_r, ["train-clean-100"], make_run_context("ingest", stage_index=0)
    )
    by_id = {c.id: c for c in cutset}
    assert by_id["1089_134686_000001_000000"].supervisions[0].text == "Hello, world."
    assert by_id["2289_200000_000001_000000"].supervisions[0].text == "Goodbye world."


def test_libritts_r_provenance_tag_differs(mock_libritts_r: Path, make_run_context) -> None:
    """A LibriTTS-R Cut must be traceable to the libritts_r recipe (not libritts)."""
    recipe = get_recipe("libritts_r")
    cutset = recipe.prepare(
        mock_libritts_r, ["train-clean-100"], make_run_context("ingest", stage_index=0)
    )
    tags = {c.provenance.generated_by for c in cutset}
    assert tags == {"libritts_r_recipe@1"}


def test_libritts_r_honours_top_dir_override(mock_libritts_r: Path, make_run_context) -> None:
    """The recipe must look inside ``LibriTTS_R/`` (not ``LibriTTS/``)."""
    recipe = get_recipe("libritts_r")
    # Sanity: parent has LibriTTS_R/, no LibriTTS/.
    assert (mock_libritts_r / "LibriTTS_R").is_dir()
    assert not (mock_libritts_r / "LibriTTS").exists()
    cutset = recipe.prepare(
        mock_libritts_r, ["train-clean-100"], make_run_context("ingest", stage_index=0)
    )
    assert len(cutset) == 2


def test_libritts_r_download_urls_point_at_slr141() -> None:
    """Every download URL must be on OpenSLR resource 141, never 60."""
    recipe = get_recipe("libritts_r")
    urls = [u for urls in recipe.download_urls.values() for u in urls]
    assert all("openslr.org/resources/141/" in u for u in urls), urls
    assert all("/60/" not in u for u in urls), urls
