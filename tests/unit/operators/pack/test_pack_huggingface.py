"""Unit tests for pack_huggingface operator."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from voxkitchen.operators.registry import get_operator
from voxkitchen.schema.cut import Cut
from voxkitchen.schema.cutset import CutSet
from voxkitchen.schema.provenance import Provenance
from voxkitchen.utils.audio import recording_from_file


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


def test_pack_hf_is_registered() -> None:
    from voxkitchen.operators.pack.pack_huggingface import PackHuggingFaceOperator

    assert get_operator("pack_huggingface") is PackHuggingFaceOperator


def test_pack_hf_creates_dataset(mono_wav_16k: Path, tmp_path: Path) -> None:
    from voxkitchen.operators.pack.pack_huggingface import (
        PackHuggingFaceConfig,
        PackHuggingFaceOperator,
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
    out_dir = tmp_path / "hf_out"
    config = PackHuggingFaceConfig(output_dir=str(out_dir))
    op = PackHuggingFaceOperator(config, ctx=ctx)

    cut = _cut_from_path(mono_wav_16k)
    cuts = CutSet([cut])
    result = op.process(cuts)

    # HuggingFace datasets save_to_disk writes dataset_dict.json or dataset_info.json
    assert out_dir.exists()
    files = list(out_dir.iterdir())
    assert len(files) > 0

    # Returns cuts unchanged
    assert [c.id for c in result] == [cut.id]
