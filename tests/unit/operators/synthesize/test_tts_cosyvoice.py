"""Unit tests for tts_cosyvoice operator."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest
import soundfile as sf

try:
    from cosyvoice.cli.cosyvoice import AutoModel  # noqa: F401
except ImportError:
    pytest.skip("cosyvoice not available", allow_module_level=True)

from voxkitchen.operators.registry import get_operator
from voxkitchen.operators.synthesize.tts_cosyvoice import (
    TtsCosyVoiceConfig,
    TtsCosyVoiceOperator,
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


def test_tts_cosyvoice_is_registered() -> None:
    assert get_operator("tts_cosyvoice") is TtsCosyVoiceOperator


def test_tts_cosyvoice_produces_audio() -> None:
    assert TtsCosyVoiceOperator.produces_audio is True
    assert TtsCosyVoiceOperator.reads_audio_bytes is False
    assert TtsCosyVoiceOperator.device == "gpu"


def test_tts_cosyvoice_config_defaults() -> None:
    config = TtsCosyVoiceConfig()
    assert config.model_id == "FunAudioLLM/CosyVoice2-0.5B"
    assert config.mode == "sft"
    assert config.spk_id == "default"
    assert config.reference_audio is None


@pytest.mark.slow
def test_tts_cosyvoice_sft_mode(tmp_path: Path, make_run_context) -> None:
    ctx = make_run_context("tts")
    cs = CutSet([_text_cut("c0", "你好，这是一个语音合成测试。")])  # noqa: RUF001
    config = TtsCosyVoiceConfig(mode="sft")
    op = TtsCosyVoiceOperator(config, ctx)
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
    assert out.provenance.generated_by == "tts_cosyvoice"


@pytest.mark.slow
def test_tts_cosyvoice_skips_cut_without_text(tmp_path: Path, make_run_context) -> None:
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
    config = TtsCosyVoiceConfig()
    op = TtsCosyVoiceOperator(config, ctx)
    op.setup()
    result = op.process(cs)
    op.teardown()

    assert len(list(result)) == 0
