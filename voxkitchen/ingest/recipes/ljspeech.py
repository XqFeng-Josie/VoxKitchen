"""LJSpeech recipe: parse a local LJSpeech-1.1 directory into a CutSet.

LJSpeech is the canonical single-speaker English TTS dataset (~24 hours,
13,100 utterances read by one female speaker). The recipe targets the
v1.1 release distributed by keithito.com.

Directory layout produced by extracting ``LJSpeech-1.1.tar.bz2``::

    LJSpeech-1.1/
      metadata.csv         # pipe-separated: id|raw_text|normalized_text
      wavs/
        LJ001-0001.wav
        LJ001-0002.wav
        ...

The recipe emits one Cut per row of ``metadata.csv``. Each Cut carries
the **normalized** text as the supervision (preferred for TTS training);
the original raw text is preserved in ``cut.custom["raw_text"]``.

All cuts share a single speaker label (``LJ``) because the dataset is
mono-speaker by design.
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

_SPEAKER_ID = "LJ"
_LANGUAGE = "en"


class LJSpeechRecipe(Recipe):
    """Parse LJSpeech-1.1 into a CutSet."""

    name = "ljspeech"
    # The dataset ships as a single ~2.6 GB archive; no per-subset selection.
    # Using the canonical mirror at data.keithito.com. The same archive is
    # also re-hosted on OpenSLR/12 as resource 109 but with rate limits.
    download_urls = {
        "default": ["https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2"],
    }

    def prepare(self, root: Path, subsets: list[str] | None, ctx: RunContext) -> CutSet:
        # The tarball extracts to <root>/LJSpeech-1.1/ ; tolerate both that
        # layout and a "user already pointed at the inner directory" layout.
        ls_dir = root / "LJSpeech-1.1"
        effective_root = ls_dir if ls_dir.is_dir() else root

        metadata = effective_root / "metadata.csv"
        if not metadata.is_file():
            raise FileNotFoundError(
                f"LJSpeech metadata.csv not found under {effective_root}. "
                "Extract LJSpeech-1.1.tar.bz2 into the recipe root first."
            )

        wav_dir = effective_root / "wavs"
        if not wav_dir.is_dir():
            raise FileNotFoundError(f"LJSpeech wavs/ directory not found under {effective_root}")

        # ``subsets`` is accepted for API symmetry with other recipes but
        # LJSpeech has no native subset partitioning. Quietly ignore it; the
        # signature is part of the Recipe contract.
        cuts: list[Cut] = []
        for row in self._parse_metadata(metadata):
            utt_id, raw_text, normalized_text = row
            audio_path = wav_dir / f"{utt_id}.wav"
            if not audio_path.exists():
                # Skip rows whose audio is missing rather than abort — partial
                # extractions during testing are recoverable.
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
                            text=normalized_text or raw_text,
                            speaker=_SPEAKER_ID,
                            language=_LANGUAGE,
                        )
                    ],
                    provenance=Provenance(
                        source_cut_id=None,
                        generated_by="ljspeech_recipe@1",
                        stage_name=ctx.stage_name,
                        created_at=datetime.now(timezone.utc),
                        pipeline_run_id=ctx.pipeline_run_id,
                    ),
                    custom={"raw_text": raw_text} if raw_text != normalized_text else {},
                )
            )
        return CutSet(cuts)

    @staticmethod
    def _parse_metadata(path: Path) -> list[tuple[str, str, str]]:
        """Parse metadata.csv into ``(utt_id, raw_text, normalized_text)`` rows.

        The CSV uses ``|`` as separator and has no header. Empty lines are
        skipped silently; rows with fewer than three fields are also skipped
        so a partially-truncated file doesn't blow up the whole prepare step.
        """
        rows: list[tuple[str, str, str]] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            parts = line.split("|")
            if len(parts) < 3:
                continue
            utt_id, raw_text, normalized_text = parts[0], parts[1], parts[2]
            rows.append((utt_id, raw_text, normalized_text))
        return rows


register_recipe(LJSpeechRecipe())
