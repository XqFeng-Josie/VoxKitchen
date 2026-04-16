"""Unit tests for snr_estimate operator."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import soundfile as sf
from voxkitchen.operators.registry import get_operator
from voxkitchen.schema.cut import Cut
from voxkitchen.schema.cutset import CutSet
from voxkitchen.schema.provenance import Provenance
from voxkitchen.schema.recording import AudioSource, Recording


def _make_cut(path: Path) -> Cut:
    """Build a Cut from an audio file path."""
    info = sf.info(str(path))
    rec = Recording(
        id=path.stem,
        sources=[AudioSource(type="file", channels=[0], source=str(path))],
        sampling_rate=info.samplerate,
        num_samples=info.frames,
        duration=info.duration,
        num_channels=info.channels,
    )
    return Cut(
        id=path.stem,
        recording_id=rec.id,
        start=0.0,
        duration=info.duration,
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


def test_snr_estimate_is_registered() -> None:
    from voxkitchen.operators.quality.snr_estimate import SnrEstimateOperator

    assert get_operator("snr_estimate") is SnrEstimateOperator


def test_snr_estimate_adds_metric(mono_wav_16k: Path) -> None:
    """Running snr_estimate on a sine wave adds a positive float snr metric."""
    from voxkitchen.operators.quality.snr_estimate import SnrEstimateConfig, SnrEstimateOperator

    cut = _make_cut(mono_wav_16k)
    config = SnrEstimateConfig()
    op = SnrEstimateOperator(config, ctx=object())  # type: ignore[arg-type]
    result = list(op.process(CutSet([cut])))

    assert len(result) == 1
    assert "snr" in result[0].metrics
    assert isinstance(result[0].metrics["snr"], float)
    assert result[0].metrics["snr"] > 0.0


def test_snr_estimate_preserves_other_metrics(mono_wav_16k: Path) -> None:
    """snr_estimate merges into existing metrics without dropping them."""
    from voxkitchen.operators.quality.snr_estimate import SnrEstimateConfig, SnrEstimateOperator

    cut = _make_cut(mono_wav_16k)
    cut = cut.model_copy(update={"metrics": {"existing": 1.0}})
    config = SnrEstimateConfig()
    op = SnrEstimateOperator(config, ctx=object())  # type: ignore[arg-type]
    result = list(op.process(CutSet([cut])))

    assert len(result) == 1
    assert result[0].metrics["existing"] == 1.0
    assert "snr" in result[0].metrics
