"""LibriTTS recipe: parse a local LibriTTS directory into a CutSet.

LibriTTS is a multi-speaker English TTS corpus derived from LibriSpeech
audiobooks, but resegmented at sentence boundaries and shipped with
TTS-friendly text normalization. The canonical mirror is OpenSLR
resource 60. Subset sizes (compressed): dev-clean 1.3 GB, dev-other 1.3
GB, test-clean 1.2 GB, test-other 1.4 GB, train-clean-100 6.0 GB,
train-clean-360 24 GB, train-other-500 33 GB.

Each subset's tarball expands to::

    LibriTTS/<subset>/<speaker_id>/<chapter_id>/
        <utt>.wav
        <utt>.normalized.txt       # TTS-normalized text
        <utt>.original.txt         # original LibriSpeech text
        <utt>.normalized.txt
    LibriTTS/speakers.tsv          # READER\tGENDER\tSUBSET\tNAME

The recipe emits one Cut per ``*.wav`` and prefers the **normalized**
text (matching the convention from the LJSpeech recipe). Speaker IDs
are taken from the directory hierarchy; gender is enriched from
``speakers.tsv`` when present.
"""

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

_LANGUAGE = "en"


class LibriTTSRecipe(Recipe):
    """Parse LibriTTS into a CutSet."""

    name = "libritts"
    # Same OpenSLR mirror as LibriSpeech; resource 60.
    download_urls = {
        "dev-clean": ["https://www.openslr.org/resources/60/dev-clean.tar.gz"],
        "dev-other": ["https://www.openslr.org/resources/60/dev-other.tar.gz"],
        "test-clean": ["https://www.openslr.org/resources/60/test-clean.tar.gz"],
        "test-other": ["https://www.openslr.org/resources/60/test-other.tar.gz"],
        "train-clean-100": ["https://www.openslr.org/resources/60/train-clean-100.tar.gz"],
        "train-clean-360": ["https://www.openslr.org/resources/60/train-clean-360.tar.gz"],
        "train-other-500": ["https://www.openslr.org/resources/60/train-other-500.tar.gz"],
    }
    # HEAD-probed Content-Length values (2026-05) — compressed tarball
    # size per subset.
    download_sizes = {
        "dev-clean": 1_291_469_655,
        "dev-other": 924_804_676,
        "test-clean": 1_230_670_113,
        "test-other": 964_502_297,
        "train-clean-100": 7_723_686_890,
        "train-clean-360": 27_504_073_644,
        "train-other-500": 44_565_031_479,
    }

    def prepare(self, root: Path, subsets: list[str] | None, ctx: RunContext) -> CutSet:
        # Tarballs extract to <root>/LibriTTS/<subset>/.... Tolerate both
        # "user pointed at parent" and "user pointed at inner directory".
        lt_dir = root / "LibriTTS"
        effective_root = lt_dir if lt_dir.is_dir() else root

        spk_gender = self._parse_speakers_tsv(effective_root / "speakers.tsv")
        target_subsets = subsets or self._discover_subsets(effective_root)

        cuts: list[Cut] = []
        for subset_name in target_subsets:
            subset_dir = effective_root / subset_name
            if not subset_dir.is_dir():
                raise FileNotFoundError(f"subset not found: {subset_dir}")
            for audio_path in sorted(subset_dir.rglob("*.wav")):
                utt_id = audio_path.stem
                text = self._read_text(audio_path)
                if text is None:
                    # No normalized.txt and no original.txt — skip this wav
                    # instead of aborting. Partial LibriTTS trees happen.
                    continue
                # Directory layout: <subset>/<speaker>/<chapter>/<utt>.wav
                chapter_dir = audio_path.parent
                speaker = chapter_dir.parent.name
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
                                language=_LANGUAGE,
                                gender=spk_gender.get(speaker),
                            )
                        ],
                        provenance=Provenance(
                            source_cut_id=None,
                            generated_by="libritts_recipe@1",
                            stage_name=ctx.stage_name,
                            created_at=datetime.now(timezone.utc),
                            pipeline_run_id=ctx.pipeline_run_id,
                        ),
                        custom={"subset": subset_name, "chapter": chapter_dir.name},
                    )
                )
        return CutSet(cuts)

    @staticmethod
    def _read_text(audio_path: Path) -> str | None:
        """Prefer the normalized transcript; fall back to the original."""
        normalized = audio_path.with_suffix(".normalized.txt")
        if normalized.is_file():
            return normalized.read_text(encoding="utf-8").strip()
        original = audio_path.with_suffix(".original.txt")
        if original.is_file():
            return original.read_text(encoding="utf-8").strip()
        return None

    @staticmethod
    def _discover_subsets(root: Path) -> list[str]:
        """Return subset directory names present under *root*.

        We restrict to the official LibriTTS subset names so unrelated
        sibling directories (``speakers.tsv``, README files, etc.) don't
        get picked up as subsets.
        """
        official = {
            "dev-clean",
            "dev-other",
            "test-clean",
            "test-other",
            "train-clean-100",
            "train-clean-360",
            "train-other-500",
        }
        return sorted(p.name for p in root.iterdir() if p.is_dir() and p.name in official)

    @staticmethod
    def _parse_speakers_tsv(path: Path) -> dict[str, str]:
        """Return ``{speaker_id: gender}`` from speakers.tsv when present.

        The official file is tab-separated: ``READER\tGENDER\tSUBSET\tNAME``.
        We accept either ``M`` / ``F`` or full words and normalize to the
        schema's ``m`` / ``f`` codes. Rows we can't parse are skipped.
        """
        if not path.is_file():
            return {}
        result: dict[str, str] = {}
        for i, line in enumerate(path.read_text(encoding="utf-8").splitlines()):
            line = line.strip()
            if not line:
                continue
            # Skip an optional header row that uses the literal field names.
            if i == 0 and line.upper().startswith("READER"):
                continue
            parts = line.split("\t")
            if len(parts) < 2:
                continue
            spk, raw_gender = parts[0], parts[1].lower()
            if raw_gender.startswith("m"):
                result[spk] = "m"
            elif raw_gender.startswith("f"):
                result[spk] = "f"
        return result


register_recipe(LibriTTSRecipe())
