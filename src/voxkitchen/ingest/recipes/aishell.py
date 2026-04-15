"""AISHELL-1 recipe: parse a local AISHELL-1 directory into a CutSet."""

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

_DEFAULT_SUBSETS = ["train", "dev", "test"]


class AishellRecipe(Recipe):
    name = "aishell"
    download_urls = {
        "data_aishell": [
            "https://www.openslr.org/resources/33/data_aishell.tgz",
        ],
        "resource_aishell": [
            "https://www.openslr.org/resources/33/resource_aishell.tgz",
        ],
    }

    def download(self, root: Path, subsets: list[str] | None) -> None:
        """Download AISHELL-1 and handle nested tgz extraction.

        data_aishell.tgz contains data_aishell/wav/wav.tgz which needs
        a second extraction step.
        """
        import tarfile

        from voxkitchen.utils.download import download_file, extract_tar

        # Always download both parts
        for subset, urls in self.download_urls.items():
            for url in urls:
                filename = url.rsplit("/", 1)[-1]
                archive_path = root / filename
                download_file(url, archive_path, desc=f"aishell/{subset}")
                extract_tar(archive_path, root)

        # Handle nested wav.tgz inside data_aishell/wav/
        nested_tgz = root / "data_aishell" / "wav" / "wav.tgz"
        if nested_tgz.exists():
            wav_dir = root / "data_aishell" / "wav"
            with tarfile.open(nested_tgz, "r:gz") as tar:
                tar.extractall(path=wav_dir, filter="data")

    def prepare(self, root: Path, subsets: list[str] | None, ctx: RunContext) -> CutSet:
        transcripts = self._parse_transcript(root)
        target_subsets = subsets or _DEFAULT_SUBSETS
        wav_root = root / "data_aishell" / "wav"
        cuts: list[Cut] = []
        for subset_name in target_subsets:
            subset_dir = wav_root / subset_name
            if not subset_dir.is_dir():
                continue
            for audio_path in sorted(subset_dir.rglob("*.wav")):
                utt_id = audio_path.stem
                if utt_id not in transcripts:
                    continue
                text = transcripts[utt_id]
                speaker = audio_path.parent.name
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
                                speaker=speaker,
                                language="zh",
                            )
                        ],
                        provenance=Provenance(
                            source_cut_id=None,
                            generated_by="aishell_recipe@1",
                            stage_name=ctx.stage_name,
                            created_at=datetime.now(timezone.utc),
                            pipeline_run_id=ctx.pipeline_run_id,
                        ),
                        custom={"subset": subset_name},
                    )
                )
        return CutSet(cuts)

    def _parse_transcript(self, root: Path) -> dict[str, str]:
        trans_dir = root / "data_aishell" / "transcript"
        trans_files = list(trans_dir.glob("*.txt"))
        if not trans_files:
            return {}
        result: dict[str, str] = {}
        for line in trans_files[0].read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            utt_id = parts[0]
            text = "".join(parts[1:])
            result[utt_id] = text
        return result


register_recipe(AishellRecipe())
