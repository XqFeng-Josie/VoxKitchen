"""Unit tests for paraformer_asr operator."""

from __future__ import annotations

try:
    from funasr import AutoModel  # noqa: F401
except ImportError:
    import pytest

    pytest.skip("funasr not available", allow_module_level=True)

from datetime import datetime, timezone
from pathlib import Path

import pytest

from voxkitchen.operators.annotate.paraformer_asr import (
    ParaformerAsrConfig,
    ParaformerAsrOperator,
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
        stage_name="asr",
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
            created_at=datetime(2026, 4, 11, tzinfo=timezone.utc),
            pipeline_run_id="run-test",
        ),
    )


# ---------------------------------------------------------------------------
# Fast (no model download)
# ---------------------------------------------------------------------------


def test_paraformer_asr_is_registered() -> None:
    assert get_operator("paraformer_asr") is ParaformerAsrOperator


def test_paraformer_asr_class_attrs() -> None:
    assert ParaformerAsrOperator.device == "gpu"
    assert ParaformerAsrOperator.produces_audio is False
    assert ParaformerAsrOperator.reads_audio_bytes is True
    assert "funasr" in ParaformerAsrOperator.required_extras


# ---------------------------------------------------------------------------
# Slow (downloads Paraformer model)
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_paraformer_asr_transcribes(mono_wav_16k: Path, tmp_path: Path) -> None:
    """Real model: a sine wave should complete without error and return 1 cut."""
    cut = _cut_from_path(mono_wav_16k)
    config = ParaformerAsrConfig()
    op = ParaformerAsrOperator(config, ctx=_ctx(tmp_path))
    op.setup()
    result = list(op.process(CutSet([cut])))
    op.teardown()

    assert len(result) == 1
    assert isinstance(result[0].supervisions, list)
