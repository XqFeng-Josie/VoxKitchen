"""THCHS-30 recipe: parse the Tsinghua Chinese 30-hour Database into a CutSet.

THCHS-30 is a 30-hour Mandarin read-speech corpus from CSLT Tsinghua
University, published on OpenSLR resource 18. It is the canonical small,
permissive (Apache-2.0) Mandarin ASR baseline — 40 speakers, ~13k
utterances at 16 kHz, with character / pinyin / phoneme transcripts.

The corpus ships as three archives:

* ``data_thchs30.tgz`` — main 30-hour speech corpus (~6.4 GB).
* ``test-noise.tgz``   — noisy test conditions (~1.9 GB; *not* parsed
  here — the recipe targets the main clean corpus).
* ``resource.tgz``     — language model + lexicon resources (~24 MB).

Directory layout inside ``data_thchs30.tgz``::

    data_thchs30/
      data/             # all *.wav and *.wav.trn (transcripts)
      train/            # symlinks back into data/
      dev/              # symlinks back into data/
      test/             # symlinks back into data/
      lm_word/ lm_phone/   # language model files

Each transcript file ``<utt>.wav.trn`` is three lines:

    1. Space-separated Chinese characters (the text)
    2. Pinyin (with tone digits)
    3. Phoneme sequence

Utterance IDs encode the speaker (e.g. ``A11_0.wav`` → speaker ``A11``).

The recipe parses the ``train`` / ``dev`` / ``test`` symlink directories
to honour the official splits and treats them as subsets.
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

_LANGUAGE = "zh"
_OFFICIAL_SUBSETS = ("train", "dev", "test")


class Thchs30Recipe(Recipe):
    """Parse THCHS-30 into a CutSet."""

    name = "thchs30"
    # OpenSLR resource 18. We only fetch the main corpus by default;
    # ``test-noise`` and ``resource`` are optional add-ons exposed as
    # explicit subset names so users can ``--subsets test-noise`` if
    # they actually need them.
    download_urls = {
        "main": ["https://www.openslr.org/resources/18/data_thchs30.tgz"],
        "test-noise": ["https://www.openslr.org/resources/18/test-noise.tgz"],
        "resource": ["https://www.openslr.org/resources/18/resource.tgz"],
    }
    # HEAD-probed Content-Length values (2026-05).
    download_sizes = {
        "main": 6_453_425_169,
        "test-noise": 1_971_460_210,
        "resource": 24_813_708,
    }

    def prepare(self, root: Path, subsets: list[str] | None, ctx: RunContext) -> CutSet:
        # The main tarball extracts to <root>/data_thchs30/. Tolerate
        # "user pointed at the inner directory" as well.
        ds_dir = root / "data_thchs30"
        effective_root = ds_dir if ds_dir.is_dir() else root

        # Subset semantics matching LibriTTS: if the user *explicitly*
        # requests a subset that doesn't exist, that's a config error
        # worth surfacing. If the user didn't ask, silently auto-discover
        # whichever of the official splits is on disk — partial extracts
        # (e.g. ``train`` only) are a routine first-time-user scenario.
        if subsets is None:
            target_subsets = [s for s in _OFFICIAL_SUBSETS if (effective_root / s).is_dir()]
        else:
            target_subsets = subsets
            for s in target_subsets:
                if s not in _OFFICIAL_SUBSETS:
                    raise ValueError(
                        f"unknown subset {s!r} for thchs30. Available: {list(_OFFICIAL_SUBSETS)}"
                    )

        cuts: list[Cut] = []
        for subset_name in target_subsets:
            subset_dir = effective_root / subset_name
            if not subset_dir.is_dir():
                raise FileNotFoundError(f"thchs30 subset not found: {subset_dir}")
            for audio_path in sorted(subset_dir.glob("*.wav")):
                trn_path = audio_path.with_suffix(".wav.trn")
                if not trn_path.is_file():
                    # No transcript → skip rather than abort; partial
                    # extracts shouldn't break a whole subset.
                    continue
                text, pinyin, phonemes = self._read_trn(trn_path)
                if text is None:
                    continue
                # Filenames like ``A11_0.wav`` → speaker ``A11``.
                speaker = audio_path.stem.split("_", 1)[0]
                rec = recording_from_file(audio_path, recording_id=audio_path.stem)
                cuts.append(
                    Cut(
                        id=audio_path.stem,
                        recording_id=rec.id,
                        start=0.0,
                        duration=rec.duration,
                        recording=rec,
                        supervisions=[
                            Supervision(
                                id=f"{audio_path.stem}__text",
                                recording_id=rec.id,
                                start=0.0,
                                duration=rec.duration,
                                text=text,
                                speaker=speaker,
                                language=_LANGUAGE,
                            )
                        ],
                        provenance=Provenance(
                            source_cut_id=None,
                            generated_by="thchs30_recipe@1",
                            stage_name=ctx.stage_name,
                            created_at=datetime.now(timezone.utc),
                            pipeline_run_id=ctx.pipeline_run_id,
                        ),
                        custom={
                            "subset": subset_name,
                            "pinyin": pinyin,
                            "phonemes": phonemes,
                        },
                    )
                )
        return CutSet(cuts)

    @staticmethod
    def _read_trn(path: Path) -> tuple[str | None, str, str]:
        """Read a ``*.wav.trn`` and return ``(text, pinyin, phonemes)``.

        Returns ``(None, "", "")`` if the file is too short to parse —
        callers skip such utterances. Spaces between Chinese characters
        in the text line are preserved as the recipe ships them (the
        canonical form for downstream LM building).
        """
        lines = [ln.strip() for ln in path.read_text(encoding="utf-8").splitlines()]
        # Drop blanks; the official corpus always has exactly three
        # non-empty lines but some forks ship with a trailing newline.
        lines = [ln for ln in lines if ln]
        if not lines:
            return None, "", ""
        text = lines[0]
        pinyin = lines[1] if len(lines) >= 2 else ""
        phonemes = lines[2] if len(lines) >= 3 else ""
        return text, pinyin, phonemes


register_recipe(Thchs30Recipe())
