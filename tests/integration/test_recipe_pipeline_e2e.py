"""End-to-end test: LibriSpeech recipe → pack_manifest pipeline."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf
from voxkitchen.pipeline.loader import load_pipeline_spec
from voxkitchen.pipeline.runner import run_pipeline
from voxkitchen.schema.io import read_cuts


def _make_mock_librispeech(tmp_path: Path) -> Path:
    """Create a tiny LibriSpeech-like directory."""
    subset = tmp_path / "libri" / "train-clean-100" / "1089" / "134686"
    subset.mkdir(parents=True)
    for utt_id in ["0001", "0002"]:
        fname = f"1089-134686-{utt_id}.flac"
        audio = np.sin(np.linspace(0, 1, 16000)).astype(np.float32) * 0.5
        sf.write(subset / fname, audio, 16000)
    (subset / "1089-134686.trans.txt").write_text(
        "1089-134686-0001 HELLO WORLD\n1089-134686-0002 GOODBYE WORLD\n"
    )
    return tmp_path / "libri"


def test_librispeech_recipe_pipeline(tmp_path: Path) -> None:
    mock_root = _make_mock_librispeech(tmp_path)
    work_dir = tmp_path / "work"
    yaml_path = tmp_path / "pipeline.yaml"
    yaml_path.write_text(
        f"""
version: "0.1"
name: recipe-e2e
work_dir: {work_dir}
num_cpu_workers: 1
ingest:
  source: recipe
  recipe: librispeech
  args:
    root: {mock_root}
    subsets: [train-clean-100]
stages:
  - name: pack
    op: pack_manifest
""",
        encoding="utf-8",
    )
    spec = load_pipeline_spec(yaml_path, run_id="run-recipe-e2e")
    run_pipeline(spec)

    assert (work_dir / "00_pack" / "_SUCCESS").exists()
    final_cuts = list(read_cuts(work_dir / "00_pack" / "cuts.jsonl.gz"))
    assert len(final_cuts) == 2
    texts = {c.supervisions[0].text for c in final_cuts if c.supervisions}
    assert "HELLO WORLD" in texts
    assert "GOODBYE WORLD" in texts
