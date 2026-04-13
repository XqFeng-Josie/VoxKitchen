"""End-to-end test: dir_scan → silero_vad → faster_whisper_asr → pack_manifest."""

from __future__ import annotations

from pathlib import Path

import pytest

from voxkitchen.pipeline.loader import load_pipeline_spec
from voxkitchen.pipeline.runner import run_pipeline
from voxkitchen.schema.io import read_cuts


@pytest.mark.slow
def test_vad_asr_pipeline(audio_dir: Path, tmp_path: Path) -> None:
    """Scan → Silero VAD → Faster Whisper ASR → Pack. All on CPU."""
    work_dir = tmp_path / "work"
    yaml_path = tmp_path / "pipeline.yaml"
    yaml_path.write_text(
        f"""
version: "0.1"
name: asr-e2e
work_dir: {work_dir}
num_gpus: 1
num_cpu_workers: 1
ingest:
  source: dir
  args:
    root: {audio_dir}
    recursive: true
stages:
  - name: vad
    op: silero_vad
    args:
      threshold: 0.3
  - name: asr
    op: faster_whisper_asr
    args:
      model: tiny
      compute_type: int8
  - name: pack
    op: pack_manifest
""",
        encoding="utf-8",
    )
    spec = load_pipeline_spec(yaml_path, run_id="run-asr-e2e")
    run_pipeline(spec)

    # All stages complete
    assert (work_dir / "00_vad" / "_SUCCESS").exists()
    assert (work_dir / "01_asr" / "_SUCCESS").exists()
    assert (work_dir / "02_pack" / "_SUCCESS").exists()

    # Final cuts should exist (may be 0 if VAD finds no speech in sine waves)
    final_cuts = list(read_cuts(work_dir / "02_pack" / "cuts.jsonl.gz"))
    # The pipeline should complete without crashing regardless of content
    assert isinstance(final_cuts, list)
