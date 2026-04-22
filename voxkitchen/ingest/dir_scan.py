"""DirScanIngestSource: scan a directory of audio files → CutSet."""

from __future__ import annotations

from pathlib import Path

from voxkitchen.ingest.base import IngestConfig, IngestSource
from voxkitchen.schema.cut import Cut
from voxkitchen.schema.cutset import CutSet
from voxkitchen.schema.provenance import Provenance
from voxkitchen.utils.audio import detect_audio_files, recording_from_file
from voxkitchen.utils.time import now_utc


class DirScanConfig(IngestConfig):
    root: str
    recursive: bool = True
    reference_text_glob: str | None = None  # e.g. "*.txt" — load paired reference for CER


class DirScanIngestSource(IngestSource):
    name = "dir"
    config_cls = DirScanConfig

    def run(self) -> CutSet:
        assert isinstance(self.config, DirScanConfig)
        root = Path(self.config.root)
        if not root.is_dir():
            raise FileNotFoundError(f"audio directory not found: {root}")

        audio_files = detect_audio_files(root, recursive=self.config.recursive)
        cuts: list[Cut] = []
        for audio_path in audio_files:
            rec = recording_from_file(audio_path)
            custom: dict[str, str] = {}
            if self.config.reference_text_glob:
                ref_text = _find_reference_text(audio_path, self.config.reference_text_glob)
                if ref_text is not None:
                    custom["reference_text"] = ref_text
            cut = Cut(
                id=rec.id,
                recording_id=rec.id,
                start=0.0,
                duration=rec.duration,
                recording=rec,
                supervisions=[],
                custom=custom,
                provenance=Provenance(
                    source_cut_id=None,
                    generated_by="dir_scan",
                    stage_name=self.ctx.stage_name,
                    created_at=now_utc(),
                    pipeline_run_id=self.ctx.pipeline_run_id,
                ),
            )
            cuts.append(cut)
        return CutSet(cuts)


def _find_reference_text(audio_path: Path, glob_pattern: str) -> str | None:
    """Look for a reference text file alongside ``audio_path``.

    Searches in the same directory as the audio file. If multiple files match
    the glob, the first alphabetically is used. Returns the file's text content
    with leading/trailing whitespace stripped, or ``None`` if nothing matches.
    """
    candidates = sorted(audio_path.parent.glob(glob_pattern))
    if not candidates:
        return None
    return candidates[0].read_text(encoding="utf-8").strip()
