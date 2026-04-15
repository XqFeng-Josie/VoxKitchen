"""LibriSpeech recipe: parse a local LibriSpeech directory into a CutSet."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

from voxkitchen.ingest.recipes import register_recipe
from voxkitchen.ingest.recipes.base import Recipe
from voxkitchen.schema.cut import Cut
from voxkitchen.schema.cutset import CutSet
from voxkitchen.schema.provenance import Provenance
from voxkitchen.schema.supervision import Supervision
from voxkitchen.utils.audio import recording_from_file

if TYPE_CHECKING:
    from voxkitchen.pipeline.context import RunContext


class LibriSpeechRecipe(Recipe):
    name = "librispeech"
    download_urls = {
        "dev-clean": ["https://www.openslr.org/resources/12/dev-clean.tar.gz"],
        "dev-other": ["https://www.openslr.org/resources/12/dev-other.tar.gz"],
        "test-clean": ["https://www.openslr.org/resources/12/test-clean.tar.gz"],
        "test-other": ["https://www.openslr.org/resources/12/test-other.tar.gz"],
        "train-clean-100": ["https://www.openslr.org/resources/12/train-clean-100.tar.gz"],
        "train-clean-360": ["https://www.openslr.org/resources/12/train-clean-360.tar.gz"],
        "train-other-500": ["https://www.openslr.org/resources/12/train-other-500.tar.gz"],
    }

    def prepare(self, root: Path, subsets: list[str] | None, ctx: RunContext) -> CutSet:
        # openslr tarballs extract to root/LibriSpeech/<subset>/...
        ls_dir = root / "LibriSpeech"
        effective_root = ls_dir if ls_dir.is_dir() else root
        target_subsets = subsets or self._discover_subsets(effective_root)
        cuts: list[Cut] = []
        for subset_name in target_subsets:
            subset_dir = effective_root / subset_name
            if not subset_dir.is_dir():
                raise FileNotFoundError(f"subset not found: {subset_dir}")
            for trans_file in sorted(subset_dir.rglob("*.trans.txt")):
                chapter_dir = trans_file.parent
                for utt_id, text in self._parse_transcript(trans_file).items():
                    audio_path = chapter_dir / f"{utt_id}.flac"
                    if not audio_path.exists():
                        continue
                    rec = recording_from_file(audio_path, recording_id=utt_id)
                    cuts.append(
                        Cut(
                            id=utt_id,
                            recording_id=rec.id,
                            start=0.0,
                            duration=rec.duration,
                            recording=rec,
                            supervisions=[
                                Supervision(
                                    id=f"{utt_id}__text",
                                    recording_id=rec.id,
                                    start=0.0,
                                    duration=rec.duration,
                                    text=text,
                                    speaker=utt_id.split("-")[0],
                                )
                            ],
                            provenance=Provenance(
                                source_cut_id=None,
                                generated_by="librispeech_recipe@1",
                                stage_name=ctx.stage_name,
                                created_at=datetime.now(timezone.utc),
                                pipeline_run_id=ctx.pipeline_run_id,
                            ),
                            custom={"subset": subset_name},
                        )
                    )
        return CutSet(cuts)

    def _discover_subsets(self, root: Path) -> list[str]:
        return sorted(p.name for p in root.iterdir() if p.is_dir())

    def _parse_transcript(self, path: Path) -> dict[str, str]:
        result: dict[str, str] = {}
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            utt_id, text = line.split(" ", 1)
            result[utt_id] = text
        return result


register_recipe(LibriSpeechRecipe())
