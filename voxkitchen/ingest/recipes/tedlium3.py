"""TED-LIUM release-3 recipe: parse English TED-talk ASR data into a CutSet.

TED-LIUM 3 (Hernandez et al., 2018) is the de-facto open benchmark for
ASR on long-form English speech — 452 hours of TED talks with
utterance-aligned transcripts, distributed from OpenSLR resource 51 as
a single ~54 GB tarball.

Distinct from LibriSpeech in two ways that matter for the recipe:

1. **Audio is NIST SPHERE** (``.sph``), not WAV. ``soundfile`` reads
   SPHERE natively via libsndfile, so `recording_from_file` works
   unchanged — but pipelines that re-write audio (resample, augment,
   pack) will need to materialize WAVs first; the existing audio
   operators already handle that.
2. **One ``.sph`` per talk, many utterances per ``.sph``.** Each
   utterance is a ``[start, end]`` slice into the parent talk audio.
   We emit one Cut per STM row, with ``cut.start`` / ``cut.duration``
   carrying the slice and ``cut.recording`` pointing to the talk-level
   audio file. The Cut model supports this natively.

Directory layout produced by extracting ``TEDLIUM_release-3.tgz``::

    TEDLIUM_release-3/
      data/sph/<talk>.sph         # complete talks, one .sph each
      data/stm/<talk>.stm         # transcripts aligned to the talks
      legacy/
        train/sph/, train/stm/    # train split (≈ 268 h)
        dev/sph/,   dev/stm/      # dev split (≈ 1.6 h)
        test/sph/,  test/stm/     # test split (≈ 2.6 h)

The recipe walks the legacy split that matches each requested subset
(default ``["train", "dev", "test"]``). The `data/` flat directories
are intentionally NOT scanned — they contain the same talks as the
legacy splits, and including them would double-count.

STM format (whitespace-separated, one utterance per row)::

    <talk> <chan> <speaker> <start> <end> <attribs> <transcript>

Rows whose ``speaker`` is ``inter_segment_gap`` (or whose transcript is
``ignore_time_segment_in_scoring``) are padding markers and are
silently dropped — they don't correspond to spoken material.

Text is emitted **raw** from the STM. TED-LIUM uses tokens like
``{NOISE}``, ``<unk>``, lowercase, apostrophes, and the special
``(2)`` disfluency marker; downstream pipelines that need normalized
text should add a normalization stage rather than hide the originals
here.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, NamedTuple

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
_LANGUAGE = "en"

# STM rows whose speaker column carries either of these sentinels are
# pause/padding markers, not utterances. Documented in the TED-LIUM 3
# README under "Annotation conventions".
_NON_UTTERANCE_SPEAKERS = {"inter_segment_gap"}
_NON_UTTERANCE_TEXTS = {"ignore_time_segment_in_scoring"}


class _StmRow(NamedTuple):
    speaker: str
    start: float
    end: float
    text: str


class Tedlium3Recipe(Recipe):
    """Parse TED-LIUM release-3 into a CutSet."""

    name = "tedlium3"
    # ``download_urls`` intentionally empty: the canonical OpenSLR/51 mirror
    # was de-listed at some point after 2024. ``www.openslr.org/51/`` now
    # returns "Resource not found: 51" and every ``resources/51/...`` path
    # returns 404. Verified against the live site. Until the LIUM team
    # restores a public tarball, ``vkit docker download tedlium3`` falls
    # back to the base-class NotImplementedError that ``commonvoice`` also
    # uses for manual-download datasets.
    #
    # The recipe's ``prepare()`` continues to work for users who already
    # have an extracted TED-LIUM 3 tree on disk — e.g. from an archived
    # tarball, a colleague's copy, or the HuggingFace ``LIUM/tedlium``
    # ``release3`` config repacked into the legacy/ structure. A
    # streaming-from-HF download path is tracked for a future revision.
    download_urls: dict[str, list[str]] = {}

    def prepare(self, root: Path, subsets: list[str] | None, ctx: RunContext) -> CutSet:
        # Tarball extracts to <root>/TEDLIUM_release-3/. Tolerate both
        # layouts (parent vs already-extracted-inner).
        ds_dir = root / "TEDLIUM_release-3"
        effective_root = ds_dir if ds_dir.is_dir() else root
        legacy_root = effective_root / "legacy"
        if not legacy_root.is_dir():
            raise FileNotFoundError(
                f"TED-LIUM 3 legacy/ directory not found under {effective_root}. "
                "Extract TEDLIUM_release-3.tgz first."
            )

        target_subsets = subsets or _DEFAULT_SUBSETS
        cuts: list[Cut] = []
        for subset_name in target_subsets:
            stm_dir = legacy_root / subset_name / "stm"
            sph_dir = legacy_root / subset_name / "sph"
            if not stm_dir.is_dir() or not sph_dir.is_dir():
                # Tolerate partial extracts — a user might only fetch
                # `dev` for benchmarking against a paper baseline.
                continue
            for stm_path in sorted(stm_dir.glob("*.stm")):
                talk = stm_path.stem
                sph_path = sph_dir / f"{talk}.sph"
                if not sph_path.is_file():
                    continue
                # One Recording per talk; reused by reference across each
                # utterance Cut. The Cut model carries the per-utterance
                # slice via (start, duration) into this shared recording.
                rec = recording_from_file(sph_path, recording_id=talk)
                for row in self._parse_stm(stm_path):
                    duration = row.end - row.start
                    if duration <= 0:
                        # Defensive: an STM with start >= end would yield
                        # a negative-length Cut; drop those silently
                        # rather than letting Pydantic raise on a
                        # validation we can't recover from at runtime.
                        continue
                    utt_id = f"{talk}-{row.start:.2f}-{row.end:.2f}"
                    cuts.append(
                        Cut(
                            id=utt_id,
                            recording_id=rec.id,
                            start=row.start,
                            duration=duration,
                            recording=rec,
                            supervisions=[
                                Supervision(
                                    id=f"{utt_id}__text",
                                    recording_id=rec.id,
                                    start=row.start,
                                    duration=duration,
                                    text=row.text,
                                    speaker=row.speaker,
                                    language=_LANGUAGE,
                                )
                            ],
                            provenance=Provenance(
                                source_cut_id=None,
                                generated_by="tedlium3_recipe@1",
                                stage_name=ctx.stage_name,
                                created_at=datetime.now(timezone.utc),
                                pipeline_run_id=ctx.pipeline_run_id,
                            ),
                            custom={"subset": subset_name, "talk": talk},
                        )
                    )
        return CutSet(cuts)

    @staticmethod
    def _parse_stm(path: Path) -> list[_StmRow]:
        """Parse one STM file into per-utterance rows.

        Each line is whitespace-delimited with seven leading columns
        (talk, channel, speaker, start, end, attribs, then transcript).
        Padding rows (``inter_segment_gap`` / ``ignore_time_segment_in_scoring``)
        are dropped. Malformed rows (fewer than seven tokens, non-numeric
        timestamps) are silently skipped rather than aborting the whole
        recipe — TED-LIUM occasionally ships single-bad-row files.
        """
        rows: list[_StmRow] = []
        for raw_line in path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith(";;"):
                continue
            parts = line.split(maxsplit=6)
            if len(parts) < 7:
                continue
            _talk, _channel, speaker, start_str, end_str, _attribs, text = parts
            if speaker in _NON_UTTERANCE_SPEAKERS:
                continue
            if text.strip() in _NON_UTTERANCE_TEXTS:
                continue
            try:
                start = float(start_str)
                end = float(end_str)
            except ValueError:
                continue
            rows.append(_StmRow(speaker=speaker, start=start, end=end, text=text.strip()))
        return rows


register_recipe(Tedlium3Recipe())
