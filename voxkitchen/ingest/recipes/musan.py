"""MUSAN recipe: parse a local MUSAN directory into a CutSet of background-audio cuts.

MUSAN (Music, Speech, and Noise) is a 109-hour open corpus of
non-transcribed audio specifically curated as a source of distractor
and augmentation material for speech systems. The standard reference
is Snyder, Chen, and Povey (2015); the corpus is distributed from
OpenSLR resource 17 as a single ~11 GB tarball.

The recipe pairs naturally with VoxKitchen's ``noise_augment`` operator:
download MUSAN once via ``vkit docker download musan``, then point a
``noise_augment`` stage at the resulting ``./data/musan/noise/`` (or
the whole corpus) to mix it into your speech pipeline.

Directory layout produced by extracting ``musan.tar.gz``::

    musan/
      noise/
        free-sound/noise-free-sound-0000.wav, …
        sound-bible/noise-sound-bible-0000.wav, …
      music/
        fma/, fma-western-art/, hd-classical/, jamendo/, rfm/
      speech/
        librivox/, us-gov/
      ANNOTATIONS, BACKGROUND_NOISE, LICENSE, README

The three top-level directories — ``noise``, ``music``, ``speech`` —
are the recipe's selectable "subsets". MUSAN has **no transcripts** by
design, so each Cut is emitted with an empty supervisions list. The
category and subcategory are preserved on ``cut.custom`` so downstream
operators (and `vkit inspect cuts`) can filter:

- ``cut.custom["musan_category"]`` ∈ {"noise", "music", "speech"}
- ``cut.custom["musan_subcategory"]`` is the immediate parent directory
  (e.g. ``"free-sound"``, ``"fma"``, ``"librivox"``)
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
from voxkitchen.utils.audio import recording_from_file

if TYPE_CHECKING:
    from voxkitchen.pipeline.context import RunContext

_CATEGORIES: tuple[str, ...] = ("noise", "music", "speech")


class MusanRecipe(Recipe):
    """Parse the MUSAN augmentation corpus into a CutSet."""

    name = "musan"
    download_urls = {
        # MUSAN ships as one ~10.3 GB tarball. Single-archive design — no
        # per-category download split is offered by OpenSLR.
        "musan": ["https://www.openslr.org/resources/17/musan.tar.gz"],
    }
    # HEAD-probed Content-Length (2026-05).
    download_sizes = {
        "musan": 11_086_114_085,
    }

    def prepare(self, root: Path, subsets: list[str] | None, ctx: RunContext) -> CutSet:
        # Tarball extracts to <root>/musan/. Tolerate both "user pointed at
        # the parent" and "user pointed at the extracted musan/ directory".
        ds_dir = root / "musan"
        effective_root = ds_dir if ds_dir.is_dir() else root

        # subsets are top-level category names; default is all three.
        target = subsets if subsets else list(_CATEGORIES)
        for s in target:
            if s not in _CATEGORIES:
                raise ValueError(
                    f"unknown MUSAN subset {s!r}; valid values are {list(_CATEGORIES)}"
                )

        cuts: list[Cut] = []
        for category in target:
            cat_dir = effective_root / category
            if not cat_dir.is_dir():
                # Tolerate partial extracts — a user might fetch only the
                # ``noise`` subtree for a noise-augmentation pipeline.
                continue
            for wav in sorted(cat_dir.rglob("*.wav")):
                subcategory = wav.parent.name  # free-sound, fma, librivox, …
                utt_id = wav.stem
                rec = recording_from_file(wav, recording_id=utt_id)
                cuts.append(
                    Cut(
                        id=utt_id,
                        recording_id=rec.id,
                        start=0.0,
                        duration=rec.duration,
                        recording=rec,
                        # MUSAN audio is intentionally non-transcribed.
                        supervisions=[],
                        provenance=Provenance(
                            source_cut_id=None,
                            generated_by="musan_recipe@1",
                            stage_name=ctx.stage_name,
                            created_at=datetime.now(timezone.utc),
                            pipeline_run_id=ctx.pipeline_run_id,
                        ),
                        custom={
                            "musan_category": category,
                            "musan_subcategory": subcategory,
                        },
                    )
                )
        return CutSet(cuts)


register_recipe(MusanRecipe())
