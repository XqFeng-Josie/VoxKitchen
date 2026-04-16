"""Unit tests for mel_extract operator."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from voxkitchen.operators.annotate.mel_extract import (
    MelExtractConfig,
    MelExtractOperator,
)
from voxkitchen.operators.registry import get_operator
from voxkitchen.pipeline.context import RunContext
from voxkitchen.schema.cut import Cut
from voxkitchen.schema.cutset import CutSet
from voxkitchen.schema.provenance import Provenance
from voxkitchen.utils.audio import recording_from_file


def _ctx(tmp_path: Path) -> RunContext:
    return RunContext(
        work_dir=tmp_path,
        pipeline_run_id="run-test",
        stage_index=1,
        stage_name="mel",
        num_gpus=0,
        num_cpu_workers=1,
        gc_mode="aggressive",
        device="cpu",
    )


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
            created_at=datetime(2026, 4, 15, tzinfo=timezone.utc),
            pipeline_run_id="run-test",
        ),
    )


def test_mel_extract_is_registered() -> None:
    assert get_operator("mel_extract") is MelExtractOperator


def test_mel_extract_does_not_produce_audio() -> None:
    assert MelExtractOperator.produces_audio is False


def test_mel_extract_writes_npy(mono_wav_16k: Path, tmp_path: Path) -> None:
    ctx = _ctx(tmp_path)
    cs = CutSet([_cut_from_path(mono_wav_16k)])
    config = MelExtractConfig(n_mels=80, hop_length=256)
    op = MelExtractOperator(config, ctx)
    op.setup()
    result = op.process(cs)
    op.teardown()

    out = next(iter(result))
    mel_path = out.custom.get("mel_path")
    assert mel_path is not None
    assert Path(mel_path).exists()

    mel = np.load(mel_path)
    assert mel.shape[0] == 80  # n_mels
    assert mel.shape[1] > 0  # time frames
    assert mel.dtype == np.float32


def test_mel_extract_records_shape_and_frames(mono_wav_16k: Path, tmp_path: Path) -> None:
    ctx = _ctx(tmp_path)
    cs = CutSet([_cut_from_path(mono_wav_16k)])
    config = MelExtractConfig()
    op = MelExtractOperator(config, ctx)
    op.setup()
    result = op.process(cs)
    op.teardown()

    out = next(iter(result))
    assert "mel_shape" in out.custom
    assert out.custom["mel_shape"][0] == 80
    assert "mel_frames" in out.metrics
    assert out.metrics["mel_frames"] > 0
