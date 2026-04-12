"""End-to-end test: dir_scan → fixed_segment → duration_filter → pack_kaldi."""

from __future__ import annotations

from pathlib import Path

from voxkitchen.pipeline.loader import load_pipeline_spec
from voxkitchen.pipeline.runner import run_pipeline
from voxkitchen.schema.io import read_cuts


def test_segment_filter_pack_pipeline(audio_dir: Path, tmp_path: Path) -> None:
    work_dir = tmp_path / "work"
    yaml_path = tmp_path / "pipeline.yaml"
    yaml_path.write_text(
        f"""
version: "0.1"
name: quality-e2e
work_dir: {work_dir}
num_cpu_workers: 1
ingest:
  source: dir
  args:
    root: {audio_dir}
    recursive: true
stages:
  - name: segment
    op: fixed_segment
    args:
      segment_duration: 0.3
      min_remaining: 0.1
  - name: filter
    op: duration_filter
    args:
      min_duration: 0.2
  - name: pack
    op: pack_kaldi
""",
        encoding="utf-8",
    )
    spec = load_pipeline_spec(yaml_path, run_id="run-quality-e2e")
    run_pipeline(spec)

    # All stages complete
    for i, name in enumerate(["segment", "filter", "pack"]):
        assert (work_dir / f"{i:02d}_{name}" / "_SUCCESS").exists()

    # Segmentation should produce more cuts than input (3 files x ~1s / 0.3s ~ 9)
    segment_cuts = list(read_cuts(work_dir / "00_segment" / "cuts.jsonl.gz"))
    assert len(segment_cuts) > 3

    # Duration filter may drop short segments
    filter_cuts = list(read_cuts(work_dir / "01_filter" / "cuts.jsonl.gz"))
    assert len(filter_cuts) <= len(segment_cuts)
    assert len(filter_cuts) > 0  # at least some pass

    # Pack should write Kaldi files
    kaldi_dir = work_dir / "02_pack" / "kaldi_output"
    assert (kaldi_dir / "wav.scp").exists()
    assert (kaldi_dir / "text").exists()
    assert (kaldi_dir / "utt2spk").exists()

    # wav.scp should have one line per filtered cut
    wav_scp_lines = (kaldi_dir / "wav.scp").read_text().strip().split("\n")
    assert len(wav_scp_lines) == len(filter_cuts)
