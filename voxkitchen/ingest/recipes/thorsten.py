"""Thorsten-Voice recipe: parse the German single-speaker TTS corpus.

Thorsten-Voice is a ~23-hour single-speaker German read-speech corpus
recorded by Thorsten Müller for open TTS training, released under CC0-1.0
(2021.02 "neutral" release on OpenSLR resource 95). It is the de facto
open German TTS dataset — used by Coqui, Piper, and Home Assistant.

Directory layout inside ``thorsten-de_v02.tgz``::

    thorsten-de_v02/
      metadata.csv    # LJSpeech-style: id|text|text_normalized
      wavs/<id>.wav   # 22.05 kHz mono, single male speaker

We emit one Cut per metadata row and prefer the **normalized** transcript
(matching the LJSpeech recipe convention). Speaker label is constant
(``thorsten``) because the corpus is mono-speaker by design.
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

_SPEAKER_ID = "thorsten"
_LANGUAGE = "de"


class ThorstenVoiceRecipe(Recipe):
    """Parse Thorsten-Voice (2021.02 neutral) into a CutSet."""

    name = "thorsten_voice"
    # OpenSLR resource 95, 2021.02 neutral release. Single tarball; no
    # native subset partitioning, mirroring ``ljspeech``.
    download_urls = {
        "default": ["https://www.openslr.org/resources/95/thorsten-de_v02.tgz"],
    }
    # HEAD-probed Content-Length (2026-05).
    download_sizes = {
        "default": 3_002_716_611,
    }

    def prepare(self, root: Path, subsets: list[str] | None, ctx: RunContext) -> CutSet:
        # The tarball extracts to <root>/thorsten-de_v02/ ; tolerate the
        # "user already pointed at the inner directory" layout.
        ds_dir = root / "thorsten-de_v02"
        effective_root = ds_dir if ds_dir.is_dir() else root

        metadata = effective_root / "metadata.csv"
        if not metadata.is_file():
            raise FileNotFoundError(
                f"thorsten-voice metadata.csv not found under {effective_root}. "
                "Extract thorsten-de_v02.tgz into the recipe root first."
            )

        wav_dir = effective_root / "wavs"
        if not wav_dir.is_dir():
            raise FileNotFoundError(
                f"thorsten-voice wavs/ directory not found under {effective_root}"
            )

        # ``subsets`` accepted for API symmetry with other recipes; the
        # 2021.02 release has no native partitions.
        cuts: list[Cut] = []
        for utt_id, raw_text, normalized_text in self._parse_metadata(metadata):
            audio_path = wav_dir / f"{utt_id}.wav"
            if not audio_path.exists():
                # Skip rows whose audio is missing rather than abort —
                # partial extracts are recoverable.
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
                            gender="m",
                        )
                    ],
                    provenance=Provenance(
                        source_cut_id=None,
                        generated_by="thorsten_voice_recipe@1",
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

        Same ``|``-separated, no-header layout as LJSpeech. The 2021.02
        Thorsten release sometimes ships rows with only two fields (the
        normalized column is omitted when it matches the raw text); treat
        those as ``normalized_text == raw_text`` rather than skip.
        """
        rows: list[tuple[str, str, str]] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            parts = line.split("|")
            if len(parts) < 2:
                continue
            utt_id = parts[0]
            raw_text = parts[1]
            normalized_text = parts[2] if len(parts) >= 3 else raw_text
            rows.append((utt_id, raw_text, normalized_text))
        return rows


register_recipe(ThorstenVoiceRecipe())
