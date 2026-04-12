"""Unit tests for pack_kaldi operator."""

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


def test_pack_kaldi_is_registered() -> None:
    from voxkitchen.operators.pack.pack_kaldi import PackKaldiOperator

    assert get_operator("pack_kaldi") is PackKaldiOperator


def test_pack_kaldi_writes_files(tmp_path: Path) -> None:
    from voxkitchen.operators.pack.pack_kaldi import PackKaldiConfig, PackKaldiOperator
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
    out_dir = tmp_path / "kaldi_out"
    config = PackKaldiConfig(output_dir=str(out_dir))
    op = PackKaldiOperator(config, ctx=ctx)

    cuts = CutSet([_cut("utt1"), _cut("utt2")])
    op.process(cuts)

    assert (out_dir / "wav.scp").exists()
    assert (out_dir / "text").exists()
    assert (out_dir / "utt2spk").exists()


def test_pack_kaldi_content(tmp_path: Path) -> None:
    from voxkitchen.operators.pack.pack_kaldi import PackKaldiConfig, PackKaldiOperator
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
    out_dir = tmp_path / "kaldi_out"
    config = PackKaldiConfig(output_dir=str(out_dir))
    op = PackKaldiOperator(config, ctx=ctx)

    cuts = CutSet([_cut("utt-a"), _cut("utt-b")])
    result = op.process(cuts)

    wav_scp_lines = (out_dir / "wav.scp").read_text().splitlines()
    assert any("utt-a" in line for line in wav_scp_lines)
    assert any("utt-b" in line for line in wav_scp_lines)
    # Check audio path is present
    assert any("/fake/utt-a.wav" in line for line in wav_scp_lines)

    # Operator returns cuts unchanged
    assert [c.id for c in result] == ["utt-a", "utt-b"]
