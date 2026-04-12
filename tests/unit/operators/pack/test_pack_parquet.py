"""Unit tests for pack_parquet operator."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from voxkitchen.operators.registry import get_operator
from voxkitchen.schema.cut import Cut
from voxkitchen.schema.cutset import CutSet
from voxkitchen.schema.provenance import Provenance
from voxkitchen.schema.recording import AudioSource, Recording


def _cut(cid: str, duration: float = 1.0) -> Cut:
    rec = Recording(
        id=f"rec-{cid}",
        sources=[AudioSource(type="file", channels=[0], source=f"/fake/{cid}.wav")],
        sampling_rate=16000,
        num_samples=int(16000 * duration),
        duration=duration,
        num_channels=1,
    )
    return Cut(
        id=cid,
        recording_id=rec.id,
        start=0.0,
        duration=duration,
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


def test_pack_parquet_is_registered() -> None:
    from voxkitchen.operators.pack.pack_parquet import PackParquetOperator

    assert get_operator("pack_parquet") is PackParquetOperator


def test_pack_parquet_writes_file(tmp_path: Path) -> None:
    from voxkitchen.operators.pack.pack_parquet import PackParquetConfig, PackParquetOperator
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
    out_dir = tmp_path / "parquet_out"
    config = PackParquetConfig(output_dir=str(out_dir))
    op = PackParquetOperator(config, ctx=ctx)

    cuts = CutSet([_cut("c0"), _cut("c1")])
    op.process(cuts)

    assert (out_dir / "metadata.parquet").exists()


def test_pack_parquet_has_columns(tmp_path: Path) -> None:
    import pyarrow.parquet as pq

    from voxkitchen.operators.pack.pack_parquet import PackParquetConfig, PackParquetOperator
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
    out_dir = tmp_path / "parquet_out"
    config = PackParquetConfig(output_dir=str(out_dir))
    op = PackParquetOperator(config, ctx=ctx)

    cuts = CutSet([_cut("c0"), _cut("c1")])
    result = op.process(cuts)

    table = pq.read_table(out_dir / "metadata.parquet")
    assert "id" in table.schema.names
    assert "duration" in table.schema.names

    # Verify cut IDs are actually in the parquet
    ids = table.column("id").to_pylist()
    assert "c0" in ids
    assert "c1" in ids

    # Returns cuts unchanged
    assert [c.id for c in result] == ["c0", "c1"]
