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
            cut = Cut(
                id=rec.id,
                recording_id=rec.id,
                start=0.0,
                duration=rec.duration,
                recording=rec,
                supervisions=[],
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
