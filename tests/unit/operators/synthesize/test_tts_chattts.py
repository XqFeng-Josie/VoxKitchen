"""Unit tests for tts_chattts operator."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest
import soundfile as sf

try:
    import ChatTTS  # noqa: F401
except ImportError:
    pytest.skip("ChatTTS not available", allow_module_level=True)

from voxkitchen.operators.registry import get_operator
from voxkitchen.operators.synthesize.tts_chattts import (
    TtsChatTTSConfig,
    TtsChatTTSOperator,
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


def test_tts_chattts_is_registered() -> None:
    assert get_operator("tts_chattts") is TtsChatTTSOperator


def test_tts_chattts_produces_audio() -> None:
    assert TtsChatTTSOperator.produces_audio is True
    assert TtsChatTTSOperator.reads_audio_bytes is False
    assert TtsChatTTSOperator.device == "gpu"


def test_tts_chattts_config_defaults() -> None:
    config = TtsChatTTSConfig()
    assert config.seed is None
    assert config.temperature == 0.3
    assert config.top_p == 0.7


@pytest.mark.slow
def test_tts_chattts_synthesizes_audio(tmp_path: Path, make_run_context) -> None:
    ctx = make_run_context("tts")
    cs = CutSet([_text_cut("c0", "你好，这是一个测试。")])  # noqa: RUF001
    config = TtsChatTTSConfig(seed=42)
    op = TtsChatTTSOperator(config, ctx)
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
    assert out.provenance.generated_by == "tts_chattts"


@pytest.mark.slow
def test_tts_chattts_skips_cut_without_text(tmp_path: Path, make_run_context) -> None:
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
    config = TtsChatTTSConfig()
    op = TtsChatTTSOperator(config, ctx)
    op.setup()
    result = op.process(cs)
    op.teardown()

    assert len(list(result)) == 0


@pytest.mark.slow
def test_tts_chattts_reproducible_with_seed(tmp_path: Path, make_run_context) -> None:
    ctx = make_run_context("tts")
    text = "测试可复现性。"
    config = TtsChatTTSConfig(seed=42)

    op = TtsChatTTSOperator(config, ctx)
    op.setup()
    r1 = op.process(CutSet([_text_cut("c0", text)]))
    out1 = next(iter(r1))

    r2 = op.process(CutSet([_text_cut("c1", text)]))
    out2 = next(iter(r2))
    op.teardown()

    assert out1.recording is not None
    assert out2.recording is not None
