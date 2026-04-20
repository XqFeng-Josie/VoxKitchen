"""Unit tests for tts_kokoro operator."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest
import soundfile as sf

try:
    import kokoro  # noqa: F401
except ImportError:
    pytest.skip("kokoro not available", allow_module_level=True)

from voxkitchen.operators.registry import get_operator
from voxkitchen.operators.synthesize.tts_kokoro import (
    TtsKokoroConfig,
    TtsKokoroOperator,
)
from voxkitchen.schema.cut import Cut
from voxkitchen.schema.cutset import CutSet
from voxkitchen.schema.provenance import Provenance
from voxkitchen.schema.supervision import Supervision


def _text_cut(cid: str, text: str) -> Cut:
    return Cut(
        id=cid,
        recording_id=f"text-{cid}",
        start=0.0,
        duration=0.0,
        supervisions=[
            Supervision(
                id=f"sup-{cid}",
                recording_id=f"text-{cid}",
                start=0.0,
                duration=0.0,
                text=text,
            )
        ],
        provenance=Provenance(
            source_cut_id=None,
            generated_by="test",
            stage_name="00_ingest",
            created_at=datetime(2026, 4, 15, tzinfo=timezone.utc),
            pipeline_run_id="run-test",
        ),
    )


def test_tts_kokoro_is_registered() -> None:
    assert get_operator("tts_kokoro") is TtsKokoroOperator


def test_tts_kokoro_produces_audio() -> None:
    assert TtsKokoroOperator.produces_audio is True
    assert TtsKokoroOperator.reads_audio_bytes is False


def test_tts_kokoro_config_defaults() -> None:
    config = TtsKokoroConfig()
    assert config.voice == "af_heart"
    assert config.lang_code == "a"
    assert config.speed == 1.0


@pytest.mark.slow
def test_tts_kokoro_synthesizes_audio(tmp_path: Path, make_run_context) -> None:
    ctx = make_run_context("tts")
    cs = CutSet([_text_cut("c0", "Hello, this is a test.")])
    config = TtsKokoroConfig(voice="af_heart", lang_code="a")
    op = TtsKokoroOperator(config, ctx)
    op.setup()
    result = op.process(cs)
    op.teardown()

    out = next(iter(result))
    assert out.recording is not None
    audio_path = out.recording.sources[0].source
    assert Path(audio_path).exists()
    info = sf.info(audio_path)
    assert info.samplerate == 24000
    assert info.duration > 0.1
    assert out.duration > 0.1
    assert out.provenance.generated_by == "tts_kokoro"
    assert out.provenance.source_cut_id == "c0"


@pytest.mark.slow
def test_tts_kokoro_skips_cut_without_text(tmp_path: Path, make_run_context) -> None:
    cut_no_text = Cut(
        id="c-empty",
        recording_id="text-empty",
        start=0.0,
        duration=0.0,
        supervisions=[],
        provenance=Provenance(
            source_cut_id=None,
            generated_by="test",
            stage_name="00_ingest",
            created_at=datetime(2026, 4, 15, tzinfo=timezone.utc),
            pipeline_run_id="run-test",
        ),
    )
    ctx = make_run_context("tts")
    cs = CutSet([cut_no_text])
    config = TtsKokoroConfig()
    op = TtsKokoroOperator(config, ctx)
    op.setup()
    result = op.process(cs)
    op.teardown()

    cuts = list(result)
    assert len(cuts) == 0
