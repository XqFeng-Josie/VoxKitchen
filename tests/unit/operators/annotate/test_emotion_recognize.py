"""Unit tests for emotion_recognize operator."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

try:
    from funasr import AutoModel  # noqa: F401
except ImportError:
    pytest.skip("funasr not available", allow_module_level=True)

from voxkitchen.operators.annotate.emotion_recognize import (
    EmotionRecognizeConfig,
    EmotionRecognizeOperator,
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
        stage_name="emotion",
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


def test_emotion_recognize_is_registered() -> None:
    assert get_operator("emotion_recognize") is EmotionRecognizeOperator


def test_emotion_recognize_does_not_produce_audio() -> None:
    assert EmotionRecognizeOperator.produces_audio is False


@pytest.mark.slow
def test_emotion_recognize_returns_emotion_label(
    mono_wav_16k: Path, tmp_path: Path
) -> None:
    ctx = _ctx(tmp_path)
    cs = CutSet([_cut_from_path(mono_wav_16k)])
    config = EmotionRecognizeConfig(model="iic/emotion2vec_plus_base")
    op = EmotionRecognizeOperator(config, ctx)
    op.setup()
    result = op.process(cs)
    op.teardown()

    out_cut = next(iter(result))
    assert "emotion" in out_cut.custom
    assert out_cut.custom["emotion"] in [
        "angry", "disgusted", "fearful", "happy",
        "neutral", "other", "sad", "surprised", "unknown",
    ]
    assert "emotion_scores" in out_cut.custom
    assert "emotion_model" in out_cut.custom
