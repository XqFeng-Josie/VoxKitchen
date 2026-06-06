"""Unit tests for tts_fish_speech operator."""

from __future__ import annotations

import sys
import types
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
from voxkitchen.operators.registry import get_operator
from voxkitchen.operators.synthesize.tts_fish_speech import (
    TtsFishSpeechConfig,
    TtsFishSpeechOperator,
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


def test_tts_fish_speech_is_registered() -> None:
    assert get_operator("tts_fish_speech") is TtsFishSpeechOperator


def test_tts_fish_speech_produces_audio() -> None:
    assert TtsFishSpeechOperator.produces_audio is True
    assert TtsFishSpeechOperator.reads_audio_bytes is False
    assert TtsFishSpeechOperator.device == "gpu"


def test_tts_fish_speech_config_defaults() -> None:
    config = TtsFishSpeechConfig()
    assert config.model_id == "fishaudio/s2-pro"
    assert config.reference_audio is None
    assert config.reference_text is None


def test_tts_fish_speech_loads_s2_engine(tmp_path: Path, monkeypatch, make_run_context) -> None:
    checkpoint = tmp_path / "s2-pro"
    checkpoint.mkdir()
    (checkpoint / "codec.pth").write_bytes(b"fake")
    calls = {}

    class FakeCuda:
        @staticmethod
        def is_available() -> bool:
            return True

    torch_mod = types.ModuleType("torch")
    torch_mod.float16 = "float16"
    torch_mod.bfloat16 = "bfloat16"
    torch_mod.cuda = FakeCuda()

    class FakeEngine:
        def __init__(self, *, llama_queue, decoder_model, precision, compile):
            calls["engine"] = (llama_queue, decoder_model, precision, compile)

    class FakeQueue:
        def put(self, item):
            calls["queue_stop"] = item

    def fake_queue(**kwargs):
        calls["queue"] = kwargs
        return FakeQueue()

    def fake_codec(*args):
        calls["codec"] = args
        return "fake-codec"

    inference_engine_mod = types.ModuleType("fish_speech.inference_engine")
    inference_engine_mod.TTSInferenceEngine = FakeEngine
    text2semantic_mod = types.ModuleType("fish_speech.models.text2semantic.inference")
    text2semantic_mod.launch_thread_safe_queue = fake_queue
    text2semantic_mod.load_codec_model = fake_codec
    hf_mod = types.ModuleType("huggingface_hub")
    hf_mod.snapshot_download = lambda model_id: checkpoint

    monkeypatch.setitem(sys.modules, "torch", torch_mod)
    monkeypatch.setitem(sys.modules, "fish_speech.inference_engine", inference_engine_mod)
    monkeypatch.setitem(
        sys.modules, "fish_speech.models.text2semantic.inference", text2semantic_mod
    )
    monkeypatch.setitem(sys.modules, "huggingface_hub", hf_mod)

    ctx = replace(make_run_context("fish"), device="cuda:0", num_gpus=1)
    op = TtsFishSpeechOperator(TtsFishSpeechConfig(model_id=str(checkpoint)), ctx)
    op.setup()

    assert calls["queue"]["checkpoint_path"] == checkpoint.resolve()
    assert calls["queue"]["device"] == "cuda:0"
    assert calls["queue"]["precision"] == "bfloat16"
    assert calls["queue"]["compile"] is False
    assert calls["codec"] == (checkpoint.resolve() / "codec.pth", "cuda:0", "bfloat16")
    assert calls["engine"][1:] == ("fake-codec", "bfloat16", False)
    op.teardown()
    assert calls["queue_stop"] is None


def test_tts_fish_speech_infer_uses_engine_request(tmp_path: Path, monkeypatch, make_run_context):
    ref = tmp_path / "ref.wav"
    ref.write_bytes(b"fake wav")
    seen = {}

    class FakeReferenceAudio:
        def __init__(self, *, audio, text):
            seen["reference"] = (audio, text)

    class FakeRequest:
        def __init__(self, **kwargs):
            seen["request"] = kwargs

    schema_mod = types.ModuleType("fish_speech.utils.schema")
    schema_mod.ServeReferenceAudio = FakeReferenceAudio
    schema_mod.ServeTTSRequest = FakeRequest
    monkeypatch.setitem(sys.modules, "fish_speech.utils.schema", schema_mod)

    class FakeEngine:
        def inference(self, req):
            yield types.SimpleNamespace(
                code="final",
                audio=(32000, np.array([0.1, -0.2], dtype=np.float32)),
                error=None,
            )

    config = TtsFishSpeechConfig(
        reference_audio=str(ref),
        reference_text="reference text",
        seed=123,
        chunk_length=300,
    )
    op = TtsFishSpeechOperator(config, make_run_context("fish"))
    op._inference = FakeEngine()
    audio = op._infer("hello")

    assert np.allclose(audio, np.array([0.1, -0.2], dtype=np.float32))
    assert op._sample_rate == 32000
    assert seen["reference"] == (b"fake wav", "reference text")
    assert seen["request"]["text"] == "hello"
    assert seen["request"]["seed"] == 123
    assert seen["request"]["chunk_length"] == 300
    assert len(seen["request"]["references"]) == 1


def test_tts_fish_speech_infer_wraps_inference_with_autocast(
    tmp_path: Path, monkeypatch, make_run_context
):
    """Regression test: _infer must call torch.autocast so that float32
    reference audio does not trigger a bfloat16 conv1d dtype mismatch.
    """
    import torch

    autocast_calls: list[dict] = []
    _real_autocast = torch.autocast

    class _TrackingAutocast:
        def __init__(self, **kwargs):
            self._kwargs = kwargs
            self._ctx = _real_autocast(**kwargs)

        def __enter__(self):
            autocast_calls.append(self._kwargs)
            return self._ctx.__enter__()

        def __exit__(self, *args):
            return self._ctx.__exit__(*args)

    monkeypatch.setattr(torch, "autocast", lambda **kw: _TrackingAutocast(**kw))

    class FakeEngine:
        def inference(self, req):
            yield types.SimpleNamespace(
                code="final",
                audio=(32000, np.array([0.1, -0.2], dtype=np.float32)),
                error=None,
            )

    schema_mod = types.ModuleType("fish_speech.utils.schema")
    schema_mod.ServeReferenceAudio = lambda **kw: kw
    schema_mod.ServeTTSRequest = lambda **kw: kw
    monkeypatch.setitem(sys.modules, "fish_speech.utils.schema", schema_mod)

    config = TtsFishSpeechConfig()
    op = TtsFishSpeechOperator(config, make_run_context("fish"))
    op._inference = FakeEngine()
    audio = op._infer("hello")

    assert audio is not None
    assert len(autocast_calls) == 1, "torch.autocast should be called exactly once per _infer"
    assert autocast_calls[0]["device_type"] == "cuda"
    assert autocast_calls[0]["dtype"] == torch.bfloat16


def test_tts_fish_speech_infer_failure_propagates(tmp_path: Path, monkeypatch, make_run_context):
    """_infer() raising must propagate out of process() — not be silently swallowed.

    Regression guard: previously _infer() caught all exceptions and returned None,
    causing process() to silently drop the cut. Now failures must propagate so the
    executor's per-cut fallback can record them to _errors.jsonl.
    """
    import types

    schema_mod = types.ModuleType("fish_speech.utils.schema")
    schema_mod.ServeReferenceAudio = lambda **kw: kw
    schema_mod.ServeTTSRequest = lambda **kw: kw
    monkeypatch.setitem(sys.modules, "fish_speech.utils.schema", schema_mod)

    def _exploding_infer(self, text: str):
        raise RuntimeError("simulated inference crash")

    monkeypatch.setattr(TtsFishSpeechOperator, "_infer", _exploding_infer)

    op = TtsFishSpeechOperator(TtsFishSpeechConfig(), make_run_context("fish"))
    cut = _text_cut("c0", "hello")
    with pytest.raises(RuntimeError, match="simulated inference crash"):
        op.process(CutSet([cut]))


@pytest.mark.slow
def test_tts_fish_speech_synthesizes_audio(tmp_path: Path, make_run_context) -> None:
    pytest.importorskip("fish_speech")

    ctx = make_run_context("tts")
    cs = CutSet([_text_cut("c0", "Hello, this is a test.")])
    config = TtsFishSpeechConfig()
    op = TtsFishSpeechOperator(config, ctx)
    op.setup()
    result = op.process(cs)
    op.teardown()

    out = next(iter(result))
    assert out.recording is not None
    audio_path = out.recording.sources[0].source
    assert Path(audio_path).exists()
    info = sf.info(audio_path)
    assert info.samplerate > 0
    assert info.duration > 0.1
    assert out.provenance.generated_by == "tts_fish_speech"


@pytest.mark.slow
def test_tts_fish_speech_skips_cut_without_text(tmp_path: Path, make_run_context) -> None:
    pytest.importorskip("fish_speech")

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
    config = TtsFishSpeechConfig()
    op = TtsFishSpeechOperator(config, ctx)
    op.setup()
    result = op.process(cs)
    op.teardown()

    assert len(list(result)) == 0
