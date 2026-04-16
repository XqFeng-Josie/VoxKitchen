"""Unit tests for voxkitchen.pipeline.gc."""

from __future__ import annotations

from collections.abc import Generator
from pathlib import Path

import pytest
from voxkitchen.operators.base import Operator, OperatorConfig
from voxkitchen.operators.registry import _REGISTRY, register_operator
from voxkitchen.pipeline.checkpoint import stage_dir_name
from voxkitchen.pipeline.gc import (
    GcPlan,  # noqa: F401 - imported to verify public export
    compute_gc_plan,
    empty_trash,
    run_gc,
)
from voxkitchen.pipeline.spec import IngestSpec, PipelineSpec, StageSpec
from voxkitchen.schema.cutset import CutSet


@pytest.fixture(autouse=True)
def _registry_snapshot() -> Generator[None, None, None]:
    saved = dict(_REGISTRY)
    _REGISTRY.clear()
    yield
    _REGISTRY.clear()
    _REGISTRY.update(saved)


def _register_ops() -> None:
    class _Config(OperatorConfig):
        pass

    class MaterializeOp(Operator):
        name = "materialize"
        config_cls = _Config
        produces_audio = True
        reads_audio_bytes = True

        def process(self, cuts: CutSet) -> CutSet:
            return cuts

    class AudioReaderOp(Operator):
        name = "audio_reader"
        config_cls = _Config
        produces_audio = False
        reads_audio_bytes = True

        def process(self, cuts: CutSet) -> CutSet:
            return cuts

    class MetricOnlyOp(Operator):
        name = "metric_only"
        config_cls = _Config
        produces_audio = False
        reads_audio_bytes = False

        def process(self, cuts: CutSet) -> CutSet:
            return cuts

    class PackOp(Operator):
        name = "pack"
        config_cls = _Config
        produces_audio = True
        reads_audio_bytes = True

        def process(self, cuts: CutSet) -> CutSet:
            return cuts

    register_operator(MaterializeOp)
    register_operator(AudioReaderOp)
    register_operator(MetricOnlyOp)
    register_operator(PackOp)


def _spec(*stage_ops: str) -> PipelineSpec:
    return PipelineSpec(
        version="0.1",
        name="demo",
        work_dir="/tmp/work",
        ingest=IngestSpec(source="manifest", args={}),
        stages=[StageSpec(name=f"s{i}", op=op) for i, op in enumerate(stage_ops)],
    )


def test_gc_plan_empty_for_pipeline_without_materializers() -> None:
    _register_ops()
    spec = _spec("audio_reader", "metric_only")
    plan = compute_gc_plan(spec)
    assert plan.last_consumer == {}


def test_gc_plan_maps_materializer_to_last_consumer() -> None:
    _register_ops()
    spec = _spec("materialize", "audio_reader", "metric_only")
    plan = compute_gc_plan(spec)
    # materialize @ index 0 is consumed by audio_reader @ index 1
    assert plan.last_consumer == {0: 1}


def test_gc_plan_for_consecutive_materializers() -> None:
    _register_ops()
    spec = _spec("materialize", "materialize", "audio_reader")
    plan = compute_gc_plan(spec)
    # s0 is consumed by s1 (materialize reads audio); s1 is consumed by s2
    assert plan.last_consumer == {0: 1, 1: 2}


def test_gc_plan_final_pack_stage_is_excluded_from_gc() -> None:
    _register_ops()
    spec = _spec("materialize", "audio_reader", "pack")
    plan = compute_gc_plan(spec)
    # pack is the last stage and its output is the user-facing artifact, never GC'd
    assert 2 not in plan.last_consumer


def test_gc_plan_materializer_with_no_downstream_consumer_has_no_entry() -> None:
    _register_ops()
    # materialize is followed only by a metric-only stage, so nothing "consumes" it
    spec = _spec("materialize", "metric_only")
    plan = compute_gc_plan(spec)
    assert plan.last_consumer == {}


def test_run_gc_moves_derived_to_trash(tmp_path: Path) -> None:
    _register_ops()
    spec = _spec("materialize", "audio_reader")
    plan = compute_gc_plan(spec)
    stage_names = [s.name for s in spec.stages]

    # Lay out simulated work_dir
    s0 = tmp_path / stage_dir_name(0, "s0")
    s1 = tmp_path / stage_dir_name(1, "s1")
    (s0 / "derived").mkdir(parents=True)
    (s0 / "derived" / "out.wav").write_bytes(b"fake")
    s1.mkdir(parents=True)

    # After s1 (index 1) completes, s0's derived/ is eligible for GC
    run_gc(
        plan,
        work_dir=tmp_path,
        just_completed_idx=1,
        gc_mode="aggressive",
        stage_names=stage_names,
    )

    trashed = tmp_path / "derived_trash" / stage_dir_name(0, "s0") / "derived"
    assert trashed.exists()
    assert (trashed / "out.wav").exists()
    assert not (s0 / "derived").exists()


def test_run_gc_is_noop_in_keep_mode(tmp_path: Path) -> None:
    _register_ops()
    spec = _spec("materialize", "audio_reader")
    plan = compute_gc_plan(spec)
    stage_names = [s.name for s in spec.stages]

    s0 = tmp_path / stage_dir_name(0, "s0")
    (s0 / "derived").mkdir(parents=True)
    (s0 / "derived" / "out.wav").write_bytes(b"fake")

    run_gc(
        plan,
        work_dir=tmp_path,
        just_completed_idx=1,
        gc_mode="keep",
        stage_names=stage_names,
    )
    assert (s0 / "derived" / "out.wav").exists()
    assert not (tmp_path / "derived_trash").exists()


def test_run_gc_does_nothing_if_stage_not_producer(tmp_path: Path) -> None:
    _register_ops()
    spec = _spec("audio_reader", "materialize", "audio_reader")
    plan = compute_gc_plan(spec)
    stage_names = [s.name for s in spec.stages]

    s1 = tmp_path / stage_dir_name(1, "s1")
    (s1 / "derived").mkdir(parents=True)
    (s1 / "derived" / "out.wav").write_bytes(b"fake")

    # After s0 completes, nothing should be GC'd (s0 doesn't produce audio)
    run_gc(
        plan,
        work_dir=tmp_path,
        just_completed_idx=0,
        gc_mode="aggressive",
        stage_names=stage_names,
    )
    assert (s1 / "derived" / "out.wav").exists()


def test_empty_trash_removes_trash_dir(tmp_path: Path) -> None:
    trash = tmp_path / "derived_trash"
    (trash / "sub").mkdir(parents=True)
    (trash / "sub" / "file").write_bytes(b"x")
    empty_trash(tmp_path)
    assert not trash.exists()


def test_empty_trash_is_noop_when_trash_missing(tmp_path: Path) -> None:
    empty_trash(tmp_path)  # must not raise
