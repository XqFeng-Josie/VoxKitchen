"""Unit tests for speaker_similarity operator."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pytest
from voxkitchen.operators.quality.speaker_similarity import (
    SpeakerSimilarityConfig,
    SpeakerSimilarityOperator,
)
from voxkitchen.operators.registry import get_operator
from voxkitchen.pipeline.context import RunContext
from voxkitchen.schema.cut import Cut
from voxkitchen.schema.cutset import CutSet
from voxkitchen.schema.provenance import Provenance


def _ctx(tmp_path: Path) -> RunContext:
    return RunContext(
        work_dir=tmp_path,
        pipeline_run_id="run-test",
        stage_index=1,
        stage_name="sim",
        num_gpus=0,
        num_cpu_workers=1,
        gc_mode="aggressive",
        device="cpu",
    )


def _cut_with_embedding(cid: str, embedding: list[float]) -> Cut:
    return Cut(
        id=cid,
        recording_id="rec-1",
        start=0.0,
        duration=1.0,
        supervisions=[],
        custom={"speaker_embedding": embedding},
        provenance=Provenance(
            source_cut_id=None,
            generated_by="test",
            stage_name="00_ingest",
            created_at=datetime(2026, 4, 15, tzinfo=timezone.utc),
            pipeline_run_id="run-test",
        ),
    )


def _cut_no_embedding(cid: str) -> Cut:
    return Cut(
        id=cid,
        recording_id="rec-1",
        start=0.0,
        duration=1.0,
        supervisions=[],
        provenance=Provenance(
            source_cut_id=None,
            generated_by="test",
            stage_name="00_ingest",
            created_at=datetime(2026, 4, 15, tzinfo=timezone.utc),
            pipeline_run_id="run-test",
        ),
    )


def test_speaker_similarity_is_registered() -> None:
    assert get_operator("speaker_similarity") is SpeakerSimilarityOperator


def test_speaker_similarity_identical_embedding(tmp_path: Path) -> None:
    ref = [1.0, 0.0, 0.0, 0.0]
    np.save(tmp_path / "ref.npy", np.array(ref, dtype=np.float32))

    ctx = _ctx(tmp_path)
    cs = CutSet([_cut_with_embedding("c0", ref)])
    config = SpeakerSimilarityConfig(reference_path=str(tmp_path / "ref.npy"))
    op = SpeakerSimilarityOperator(config, ctx)
    op.setup()
    result = op.process(cs)
    op.teardown()

    out = next(iter(result))
    assert out.metrics["speaker_similarity"] == pytest.approx(1.0, abs=0.01)


def test_speaker_similarity_orthogonal_embedding(tmp_path: Path) -> None:
    ref = [1.0, 0.0, 0.0, 0.0]
    np.save(tmp_path / "ref.npy", np.array(ref, dtype=np.float32))

    ctx = _ctx(tmp_path)
    cs = CutSet([_cut_with_embedding("c0", [0.0, 1.0, 0.0, 0.0])])
    config = SpeakerSimilarityConfig(reference_path=str(tmp_path / "ref.npy"))
    op = SpeakerSimilarityOperator(config, ctx)
    op.setup()
    result = op.process(cs)
    op.teardown()

    out = next(iter(result))
    assert out.metrics["speaker_similarity"] == pytest.approx(0.0, abs=0.01)


def test_speaker_similarity_no_embedding_returns_zero(tmp_path: Path) -> None:
    ref = [1.0, 0.0, 0.0, 0.0]
    np.save(tmp_path / "ref.npy", np.array(ref, dtype=np.float32))

    ctx = _ctx(tmp_path)
    cs = CutSet([_cut_no_embedding("c0")])
    config = SpeakerSimilarityConfig(reference_path=str(tmp_path / "ref.npy"))
    op = SpeakerSimilarityOperator(config, ctx)
    op.setup()
    result = op.process(cs)
    op.teardown()

    out = next(iter(result))
    assert out.metrics["speaker_similarity"] == 0.0
