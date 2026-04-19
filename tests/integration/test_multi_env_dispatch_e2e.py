"""End-to-end verification of the multi-env subprocess dispatch.

There is no Docker in CI, so we fake the "another env" by creating a
sandbox directory ``<tmp>/envs/sandbox/bin/python`` that symlinks to
``sys.executable``. Combined with an ``op_env_map.json`` pinning a chosen
operator to the ``sandbox`` env, this exercises the full cross-env code
path:

  parent runner → resolve_env(op) → "sandbox" != current_env ("core")
  → dispatch_stage_to_env → subprocess python -m ...stage_runner
  → subprocess writes cuts.jsonl.gz + _stats.json
  → parent re-loads output → next stage runs in-process

What we don't cover: the actual Python env *isolation* (we're using the
same interpreter). For that you need an image build. This test verifies
the plumbing between parent and subprocess works, including argument
passing, context serialization, error propagation, and file roundtrip.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest
from voxkitchen.pipeline.loader import load_pipeline_spec
from voxkitchen.pipeline.runner import run_pipeline
from voxkitchen.runtime import env_resolver
from voxkitchen.schema.cutset import CutSet


def _prepare_fake_sandbox(tmp_path: Path, op_name: str) -> tuple[Path, Path]:
    """Write the fake envs layout + op_env_map.json. Return (envs_dir, map_path)."""
    envs_dir = tmp_path / "envs"
    bin_dir = envs_dir / "sandbox" / "bin"
    bin_dir.mkdir(parents=True)
    (bin_dir / "python").symlink_to(sys.executable)

    map_path = tmp_path / "op_env_map.json"
    # Leave every other operator unmapped; resolve_env() falls back to
    # the in-process registry and picks "core" for them (the fallback path).
    map_path.write_text(json.dumps({op_name: "sandbox"}), encoding="utf-8")
    return envs_dir, map_path


@pytest.fixture(autouse=True)
def _reset_env_resolver_caches():
    """env_resolver uses lru_cache for op_env_map / derived registry.
    Tests here monkeypatch VKIT_OP_ENV_MAP and ENVS_DIR, so we must flush
    the cache on BOTH sides of each test — otherwise a test that loads a
    fake map pollutes every subsequent test in the same pytest process.
    """
    env_resolver.reset_caches()
    yield
    env_resolver.reset_caches()


def test_cross_env_dispatch_happy_path(
    tmp_path: Path, sample_manifest_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Force 'identity' to dispatch to a synthetic "sandbox" env whose python
    # happens to be this very interpreter. current_env defaults to "core".
    envs_dir, map_path = _prepare_fake_sandbox(tmp_path, op_name="identity")
    monkeypatch.setattr(env_resolver, "ENVS_DIR", envs_dir)
    monkeypatch.setenv("VKIT_OP_ENV_MAP", str(map_path))
    monkeypatch.setenv("VKIT_ENV", "core")
    env_resolver.reset_caches()

    work_dir = tmp_path / "work"
    yaml_path = tmp_path / "pipeline.yaml"
    yaml_path.write_text(
        f"""
version: "0.1"
name: cross-env-e2e
work_dir: {work_dir}
num_cpu_workers: 1
ingest:
  source: manifest
  args:
    path: {sample_manifest_path}
stages:
  - {{ name: dispatched, op: identity }}           # → sandbox (subprocess)
  - {{ name: local,      op: duration_filter }}    # → core (in-process)
""",
        encoding="utf-8",
    )

    spec = load_pipeline_spec(yaml_path, run_id="run-cross-env")
    run_pipeline(spec)

    # Both stages produced cuts + success marker.
    disp_dir = work_dir / "00_dispatched"
    local_dir = work_dir / "01_local"
    for d in (disp_dir, local_dir):
        assert (d / "cuts.jsonl.gz").exists(), f"missing output in {d}"
        assert (d / "_SUCCESS").exists(), f"missing _SUCCESS in {d}"
        assert (d / "_stats.json").exists(), f"missing _stats.json in {d}"

    # The dispatched stage's _stats.json was written by the subprocess,
    # which tags the env. This is the tell-tale that we actually crossed
    # the process boundary rather than falling through to in-process.
    disp_stats = json.loads((disp_dir / "_stats.json").read_text())
    assert disp_stats["env"] == "sandbox", disp_stats
    # The in-process stage doesn't record 'env' in _stats.json (that's
    # only written by stage_runner). Verify that difference holds.
    local_stats = json.loads((local_dir / "_stats.json").read_text())
    assert "env" not in local_stats

    # Cut contents round-trip through both stages unchanged.
    final = CutSet.from_jsonl_gz(local_dir / "cuts.jsonl.gz")
    assert sorted(c.id for c in final) == [f"c{i}" for i in range(5)]


def test_cross_env_subprocess_failure_surfaces_as_stage_failed(
    tmp_path: Path, sample_manifest_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """If the subprocess exits non-zero, the parent must raise StageFailedError."""
    envs_dir, map_path = _prepare_fake_sandbox(tmp_path, op_name="identity")
    monkeypatch.setattr(env_resolver, "ENVS_DIR", envs_dir)
    monkeypatch.setenv("VKIT_OP_ENV_MAP", str(map_path))
    monkeypatch.setenv("VKIT_ENV", "core")
    env_resolver.reset_caches()

    # Replace the sandbox python with a script that always exits 1. This
    # simulates a subprocess that crashes before writing its output file.
    sandbox_python = envs_dir / "sandbox" / "bin" / "python"
    sandbox_python.unlink()
    sandbox_python.write_text("#!/bin/sh\nexit 1\n", encoding="utf-8")
    sandbox_python.chmod(0o755)

    work_dir = tmp_path / "work"
    yaml_path = tmp_path / "pipeline.yaml"
    yaml_path.write_text(
        f"""
version: "0.1"
name: cross-env-fail
work_dir: {work_dir}
num_cpu_workers: 1
ingest:
  source: manifest
  args:
    path: {sample_manifest_path}
stages:
  - {{ name: will_fail, op: identity }}
""",
        encoding="utf-8",
    )
    spec = load_pipeline_spec(yaml_path, run_id="run-fail")

    from voxkitchen.pipeline.runner import StageFailedError

    with pytest.raises(StageFailedError) as exc_info:
        run_pipeline(spec)
    assert exc_info.value.stage_name == "will_fail"


def test_stage_runner_binary_entry_point(
    tmp_path: Path, sample_manifest_path: Path
) -> None:
    """`python -m voxkitchen.runtime.stage_runner` runs a stage end-to-end
    when invoked directly — independent of the parent runner. This is the
    contract the Docker subprocess relies on."""
    import subprocess

    from voxkitchen.pipeline.context import RunContext
    from dataclasses import asdict

    output_path = tmp_path / "out.jsonl.gz"
    ctx = RunContext(
        work_dir=tmp_path,
        pipeline_run_id="direct-test",
        stage_index=0,
        stage_name="stage",
        num_gpus=0,
        num_cpu_workers=1,
        gc_mode="keep",
        device="cpu",
    )

    cmd = [
        sys.executable,
        "-m",
        "voxkitchen.runtime.stage_runner",
        "--op",
        "identity",
        "--config-json",
        "{}",
        "--input",
        str(sample_manifest_path),
        "--output",
        str(output_path),
        "--ctx-json",
        json.dumps(asdict(ctx), default=str),
    ]
    env = os.environ.copy()
    env["VKIT_ENV"] = "core"
    # text=True + default locale decodes subprocess output as ASCII under
    # pytest; tqdm's block-char progress ("█" = 0xe2 0x96 0x88) crashes
    # the decoder. Pin to UTF-8.
    result = subprocess.run(
        cmd, env=env, capture_output=True, encoding="utf-8", errors="replace", check=False
    )
    assert result.returncode == 0, result.stderr
    assert output_path.exists()

    cuts = CutSet.from_jsonl_gz(output_path)
    assert len(cuts) == 5

    stats_path = tmp_path / "_stats.json"
    assert stats_path.exists()
    stats = json.loads(stats_path.read_text())
    assert stats["operator"] == "identity"
    assert stats["env"] == "core"
