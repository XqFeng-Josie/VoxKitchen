"""Unit tests for pack_webdataset operator."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

pytest.importorskip("webdataset")

from voxkitchen.operators.registry import get_operator  # noqa: E402
from voxkitchen.schema.cut import Cut  # noqa: E402
from voxkitchen.schema.cutset import CutSet  # noqa: E402
from voxkitchen.schema.provenance import Provenance  # noqa: E402
from voxkitchen.utils.audio import recording_from_file  # noqa: E402


def _cut_from_path(audio_path: Path) -> Cut:
    rec = recording_from_file(audio_path)
    return Cut(
        id=f"cut-{rec.id}",
        recording_id=rec.id,
        start=0.0,
        duration=rec.duration,
        recording=rec,
        supervisions=[],
        provenance=Provenance(
            source_cut_id=None,
            generated_by="fixture",
            stage_name="00_ingest",
            created_at=datetime(2026, 4, 11, tzinfo=timezone.utc),
            pipeline_run_id="run-test",
        ),
    )


def test_pack_wds_is_registered() -> None:
    from voxkitchen.operators.pack.pack_webdataset import PackWebDatasetOperator

    assert get_operator("pack_webdataset") is PackWebDatasetOperator


def test_pack_wds_creates_tar(mono_wav_16k: Path, tmp_path: Path) -> None:
    from voxkitchen.operators.pack.pack_webdataset import (
        PackWebDatasetConfig,
        PackWebDatasetOperator,
    )
    from voxkitchen.pipeline.context import RunContext

    ctx = RunContext(
        work_dir=tmp_path,
        pipeline_run_id="run-test",
        stage_index=0,
        stage_name="pack",
        num_gpus=0,
        num_cpu_workers=1,
        gc_mode="keep",
        device="cpu",
    )
    out_dir = tmp_path / "wds_out"
    config = PackWebDatasetConfig(output_dir=str(out_dir))
    op = PackWebDatasetOperator(config, ctx=ctx)

    cut = _cut_from_path(mono_wav_16k)
    cuts = CutSet([cut])
    result = op.process(cuts)

    tar_files = list(out_dir.glob("*.tar"))
    assert len(tar_files) >= 1

    # Returns cuts unchanged
    assert [c.id for c in result] == [cut.id]
