"""Unit tests for channel_merge operator."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import soundfile as sf
from voxkitchen.operators.basic.channel_merge import ChannelMergeConfig, ChannelMergeOperator
from voxkitchen.operators.registry import get_operator
from voxkitchen.schema.cut import Cut
from voxkitchen.schema.cutset import CutSet
from voxkitchen.schema.provenance import Provenance
from voxkitchen.utils.audio import recording_from_file


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


def test_channel_merge_is_registered() -> None:
    assert get_operator("channel_merge") is ChannelMergeOperator


def test_channel_merge_produces_audio() -> None:
    assert ChannelMergeOperator.produces_audio is True


def test_channel_merge_stereo_to_mono(
    stereo_wav_44k: Path, tmp_path: Path, make_run_context
) -> None:
    ctx = make_run_context("channel_merge")
    cs = CutSet([_cut_from_path(stereo_wav_44k)])
    config = ChannelMergeConfig(target_channels=1)
    op = ChannelMergeOperator(config, ctx)
    op.setup()
    result = op.process(cs)
    op.teardown()

    out_cut = next(iter(result))
    assert out_cut.recording is not None
    derived_path = Path(out_cut.recording.sources[0].source)
    assert derived_path.exists()
    info = sf.info(str(derived_path))
    assert info.channels == 1
    assert info.samplerate == 44100


def test_channel_merge_mono_stays_mono(
    mono_wav_16k: Path, tmp_path: Path, make_run_context
) -> None:
    ctx = make_run_context("channel_merge")
    cs = CutSet([_cut_from_path(mono_wav_16k)])
    config = ChannelMergeConfig(target_channels=1)
    op = ChannelMergeOperator(config, ctx)
    op.setup()
    result = op.process(cs)
    op.teardown()

    out_cut = next(iter(result))
    assert out_cut.recording is not None
    derived_path = Path(out_cut.recording.sources[0].source)
    assert derived_path.exists()
    info = sf.info(str(derived_path))
    assert info.channels == 1
    assert info.samplerate == 16000
