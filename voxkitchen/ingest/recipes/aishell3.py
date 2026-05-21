"""AISHELL-3 recipe: parse a local AISHELL-3 directory into a CutSet.

AISHELL-3 is a multi-speaker Mandarin TTS corpus: 218 speakers across
roughly 85 hours of clean studio recordings. Distinct from AISHELL-1
(read ASR, single channel) — AISHELL-3 is purpose-built for TTS and
voice cloning research.

Directory layout produced by extracting ``data_aishell3.tgz`` from
OpenSLR resource 93::

    data_aishell3/
      spk-info.txt              # speaker_id<TAB>age<TAB>gender<TAB>region
      train/
        wav/
          SSB0005/SSB00050001.wav
          SSB0005/SSB00050002.wav
          ...
        content.txt             # utt_id<TAB>char1 pinyin1 char2 pinyin2 ...
      test/
        wav/
          SSBNNNN/SSBNNNNNNNN.wav
          ...
        content.txt

Each row of ``content.txt`` interleaves a Mandarin character with its
pinyin token, e.g. ``SSB00050001.wav<TAB>请 qing3 选 xuan3 ...``.
The recipe peels the characters off as the supervision text and stores
the pinyin sequence in ``cut.custom["pinyin"]`` for downstream operators
that want phonetic input (e.g. CosyVoice zero-shot).

Speaker IDs are taken from the directory name (``SSB0005`` →
``SSB0005``); gender is enriched from ``spk-info.txt`` when present.
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

_DEFAULT_SUBSETS = ["train", "test"]
_LANGUAGE = "zh"


class Aishell3Recipe(Recipe):
    """Parse AISHELL-3 into a CutSet."""

    name = "aishell3"
    download_urls = {
        "data_aishell3": [
            "https://www.openslr.org/resources/93/data_aishell3.tgz",
        ],
    }

    def prepare(self, root: Path, subsets: list[str] | None, ctx: RunContext) -> CutSet:
        # Tarball extracts to <root>/data_aishell3/. Tolerate the user
        # pointing either at the parent (with the tarball alongside) or at
        # the extracted directory itself.
        ds_dir = root / "data_aishell3"
        effective_root = ds_dir if ds_dir.is_dir() else root

        spk_gender = self._parse_speaker_info(effective_root / "spk-info.txt")
        target_subsets = subsets or _DEFAULT_SUBSETS

        cuts: list[Cut] = []
        for subset_name in target_subsets:
            subset_dir = effective_root / subset_name
            if not subset_dir.is_dir():
                # Don't abort — partial extracts often have only one subset.
                continue
            transcripts = self._parse_content(subset_dir / "content.txt")
            wav_root = subset_dir / "wav"
            if not wav_root.is_dir():
                continue
            for audio_path in sorted(wav_root.rglob("*.wav")):
                utt_key = audio_path.name  # transcripts are keyed by filename
                entry = transcripts.get(utt_key)
                if entry is None:
                    continue
                text, pinyin = entry
                speaker = audio_path.parent.name
                utt_id = audio_path.stem
                rec = recording_from_file(audio_path, recording_id=utt_id)
                custom: dict[str, object] = {"subset": subset_name}
                if pinyin:
                    custom["pinyin"] = pinyin
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
                            generated_by="aishell3_recipe@1",
                            stage_name=ctx.stage_name,
                            created_at=datetime.now(timezone.utc),
                            pipeline_run_id=ctx.pipeline_run_id,
                        ),
                        custom=custom,
                    )
                )
        return CutSet(cuts)

    @staticmethod
    def _parse_content(path: Path) -> dict[str, tuple[str, str]]:
        """Return ``{wav_filename: (characters, pinyin)}`` from content.txt.

        Each row looks like ``SSB00050001.wav<TAB>请 qing3 选 xuan3 ...``.
        We split on whitespace after the tab and take every even index as
        the character and every odd index as the pinyin token.
        """
        if not path.is_file():
            return {}
        result: dict[str, tuple[str, str]] = {}
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or "\t" not in line:
                continue
            wav_name, body = line.split("\t", 1)
            tokens = body.split()
            chars = "".join(tokens[0::2])
            pinyin = " ".join(tokens[1::2])
            result[wav_name] = (chars, pinyin)
        return result

    @staticmethod
    def _parse_speaker_info(path: Path) -> dict[str, str]:
        """Return ``{speaker_id: gender}`` from spk-info.txt when present.

        File is tab-separated; the gender column uses ``male`` / ``female``
        — we map to the schema's compact codes ``m`` / ``f``. Missing or
        unparseable rows are silently skipped.
        """
        if not path.is_file():
            return {}
        result: dict[str, str] = {}
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) < 3:
                # Some releases use space separation; fall back.
                parts = line.split()
            if len(parts) < 3:
                continue
            spk = parts[0]
            raw_gender = parts[2].lower()
            if raw_gender.startswith("m"):
                result[spk] = "m"
            elif raw_gender.startswith("f"):
                result[spk] = "f"
        return result


register_recipe(Aishell3Recipe())
