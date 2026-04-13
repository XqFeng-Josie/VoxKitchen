"""Tests for the CommonVoice ingest recipe."""

from __future__ import annotations

from pathlib import Path

from voxkitchen.ingest.recipes import get_recipe
from voxkitchen.pipeline.context import RunContext


def _make_ctx(tmp_path: Path) -> RunContext:
    return RunContext(
        work_dir=tmp_path,
        pipeline_run_id="run-test",
        stage_index=0,
        stage_name="ingest",
        num_gpus=0,
        num_cpu_workers=1,
        gc_mode="aggressive",
        device="cpu",
    )


def test_commonvoice_recipe_is_registered() -> None:
    recipe = get_recipe("commonvoice")
    assert recipe.name == "commonvoice"


def test_commonvoice_parses_mock_data(mock_commonvoice: Path, tmp_path: Path) -> None:
    recipe = get_recipe("commonvoice")
    cutset = recipe.prepare(mock_commonvoice, ["train"], _make_ctx(tmp_path))
    assert len(cutset) == 2
    texts = {c.supervisions[0].text for c in cutset}
    assert "hello world" in texts
    assert "goodbye world" in texts


def test_commonvoice_cuts_have_text_and_language(mock_commonvoice: Path, tmp_path: Path) -> None:
    recipe = get_recipe("commonvoice")
    cutset = recipe.prepare(mock_commonvoice, ["train"], _make_ctx(tmp_path))
    for cut in cutset:
        sup = cut.supervisions[0]
        assert sup.text is not None
        assert sup.language == "en"
