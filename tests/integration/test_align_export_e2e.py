"""End-to-end test: resample → faster_whisper_asr → normalize_text → forced_align → pack_jsonl."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from voxkitchen.pipeline.loader import load_pipeline_spec
from voxkitchen.pipeline.preflight import preflight_spec
from voxkitchen.pipeline.runner import run_pipeline

# Absolute path so the spec works regardless of cwd inside pytest.
_DEMO_DATA = Path(__file__).parent.parent.parent / "examples" / "demo_data"

_PIPELINE_YAML = """\
version: "0.1"
name: align-e2e
work_dir: {work_dir}
num_gpus: 0
num_cpu_workers: 1
ingest:
  source: dir
  args:
    root: {data_dir}
stages:
  - name: rs
    op: resample
    args:
      target_sr: 16000
  - name: asr
    op: faster_whisper_asr
    args:
      model: tiny
  - name: norm
    op: normalize_text
    args: {{}}
  - name: align
    op: forced_align
    args: {{}}
  - name: pack
    op: pack_jsonl
    args: {{}}
"""


def _build_spec(tmp_path: Path):
    yaml_path = tmp_path / "pipeline.yaml"
    yaml_path.write_text(
        _PIPELINE_YAML.format(
            work_dir=tmp_path / "work",
            data_dir=_DEMO_DATA,
        ),
        encoding="utf-8",
    )
    return load_pipeline_spec(yaml_path, run_id="run-align-e2e")


def test_align_chain_passes_preflight(tmp_path: Path) -> None:
    """Pure static check — no models, no audio, always runs."""
    result = preflight_spec(_build_spec(tmp_path))
    assert result.errors == [], f"preflight errors: {result.errors}"


@pytest.mark.slow
def test_asr_align_export_runs(tmp_path: Path) -> None:
    """Full pipeline run on demo audio. Requires asr + align extras and network."""
    pytest.importorskip("faster_whisper")  # skip cleanly when the asr extra is absent
    spec = _build_spec(tmp_path)
    run_pipeline(spec)

    work_dir = tmp_path / "work"
    manifests = list(work_dir.rglob("manifest.jsonl"))
    assert manifests, "no manifest.jsonl found under work_dir"

    manifest = manifests[0]
    rows = [
        json.loads(line)
        for line in manifest.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert rows, "manifest.jsonl is empty"
    assert any(r.get("text") for r in rows), "no row has non-empty text"
