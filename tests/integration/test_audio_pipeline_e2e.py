"""End-to-end test: dir_scan → resample → pack_manifest on real audio."""

from __future__ import annotations

from pathlib import Path

import soundfile as sf

from voxkitchen.pipeline.loader import load_pipeline_spec
from voxkitchen.pipeline.runner import run_pipeline
from voxkitchen.schema.io import read_cuts


def test_dir_scan_to_resample_to_pack(audio_dir: Path, tmp_path: Path) -> None:
    """Full pipeline: scan dir → resample 44k→16k → pack manifest."""
    work_dir = tmp_path / "work"
    yaml_path = tmp_path / "pipeline.yaml"
    yaml_path.write_text(
        f"""
version: "0.1"
name: audio-e2e
work_dir: {work_dir}
num_cpu_workers: 1
ingest:
  source: dir
  args:
    root: {audio_dir}
    recursive: true
stages:
  - name: resample
    op: resample
    args:
      target_sr: 16000
      target_channels: 1
  - name: pack
    op: pack_manifest
""",
        encoding="utf-8",
    )
    spec = load_pipeline_spec(yaml_path, run_id="run-audio-e2e")
    run_pipeline(spec)

    # Verify resample stage produced derived audio
    resample_dir = work_dir / "00_resample"
    assert (resample_dir / "_SUCCESS").exists()
    derived = resample_dir / "derived"
    assert derived.exists()
    derived_files = list(derived.glob("*.wav"))
    assert len(derived_files) == 3  # one per input file

    # Verify all resampled files are 16kHz
    for f in derived_files:
        info = sf.info(str(f))
        assert info.samplerate == 16000

    # Verify pack stage completed
    assert (work_dir / "01_pack" / "_SUCCESS").exists()
    final_cuts = list(read_cuts(work_dir / "01_pack" / "cuts.jsonl.gz"))
    assert len(final_cuts) == 3

    # Each final cut should have a recording pointing to a derived file
    for cut in final_cuts:
        assert cut.recording is not None
        assert Path(cut.recording.sources[0].source).exists()


def test_audio_pipeline_with_resume(audio_dir: Path, tmp_path: Path) -> None:
    """Run pipeline, delete final stage, re-run → resample not re-run."""
    work_dir = tmp_path / "work"
    yaml_path = tmp_path / "pipeline.yaml"
    yaml_path.write_text(
        f"""
version: "0.1"
name: resume-test
work_dir: {work_dir}
num_cpu_workers: 1
ingest:
  source: dir
  args:
    root: {audio_dir}
    recursive: true
stages:
  - name: resample
    op: resample
    args:
      target_sr: 16000
      target_channels: 1
  - name: pack
    op: pack_manifest
""",
        encoding="utf-8",
    )
    spec = load_pipeline_spec(yaml_path, run_id="run-resume-audio")

    # First run
    run_pipeline(spec)
    assert (work_dir / "01_pack" / "_SUCCESS").exists()

    # Delete pack output
    (work_dir / "01_pack" / "_SUCCESS").unlink()
    (work_dir / "01_pack" / "cuts.jsonl.gz").unlink()

    # Record resample mtime (should not change on re-run)
    resample_manifest = work_dir / "00_resample" / "cuts.jsonl.gz"
    original_mtime = resample_manifest.stat().st_mtime

    # Resume
    run_pipeline(spec)

    assert (work_dir / "01_pack" / "_SUCCESS").exists()
    assert resample_manifest.stat().st_mtime == original_mtime  # not re-run
