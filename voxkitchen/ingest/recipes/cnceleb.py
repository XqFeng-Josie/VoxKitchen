"""CN-Celeb 1 recipe: parse a local CN-Celeb directory into a CutSet.

CN-Celeb (Fan et al., 2020; v2 dataset card on OpenSLR resource 82) is
the standard open Chinese speaker-recognition benchmark and the
counterpart to the gated VoxCeleb. It contains roughly 130,000
utterances from ~1,000 speakers across 11 genres (interview, vlog,
singing, drama, …). The dataset is distributed as a single ~22 GB
tarball (``cn-celeb_v2.tar.gz``).

The recipe is intentionally lean: CN-Celeb is **non-transcribed**
(speaker-identity research target, not ASR), so each Cut is emitted
with an empty supervisions list and a populated speaker tag. Subset
selection follows the canonical splits the corpus ships:

- ``data`` (default) — every FLAC under ``data/``. Useful for
  speaker-embedding extraction across the whole corpus.
- ``dev`` — the official dev set listed in ``dev/dev.lst``.
- ``eval`` — the official eval set listed in ``eval/lists/enroll.lst``
  and ``eval/lists/test.lst`` (concatenated).

Directory layout produced by extracting ``cn-celeb_v2.tar.gz``::

    CN-Celeb_flac/
      README
      data/
        id00000/
          enroll-001-001.flac, vlog-01-001.flac, …
        id00001/
        …
        id00999/
      dev/
        dev.lst                       # one path per line, relative to data/
      eval/
        lists/
          enroll.lst, test.lst        # trial-pair lists
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
_VALID_SUBSETS: tuple[str, ...] = ("data", "dev", "eval")


class CnCelebRecipe(Recipe):
    """Parse CN-Celeb 1 into a CutSet for speaker-identity work."""

    name = "cnceleb"
    download_urls = {
        "cn-celeb_v2": [
            "https://www.openslr.org/resources/82/cn-celeb_v2.tar.gz",
        ],
    }

    def prepare(self, root: Path, subsets: list[str] | None, ctx: RunContext) -> CutSet:
        # The tarball historically extracts to a top-level directory
        # whose exact name has varied between releases. Probe the common
        # ones; fall back to the user-supplied root if none match.
        effective_root = root
        for candidate in ("CN-Celeb_flac", "CN-Celeb", "CN-Celeb_v2"):
            if (root / candidate).is_dir():
                effective_root = root / candidate
                break

        target = subsets or ["data"]
        for s in target:
            if s not in _VALID_SUBSETS:
                raise ValueError(
                    f"unknown CN-Celeb subset {s!r}; valid values are {list(_VALID_SUBSETS)}"
                )

        # Collect (path, source_subset_name) pairs to avoid double-emitting
        # the same utterance when the user passes overlapping subsets.
        seen: set[Path] = set()
        flac_paths: list[tuple[Path, str]] = []
        for subset_name in target:
            for flac in self._iter_subset(effective_root, subset_name):
                if flac in seen or not flac.is_file():
                    continue
                seen.add(flac)
                flac_paths.append((flac, subset_name))

        cuts: list[Cut] = []
        for flac, subset_name in flac_paths:
            # Layout invariant: data/<speaker_id>/<utterance>.flac
            speaker = flac.parent.name
            utt_id = flac.stem
            rec = recording_from_file(flac, recording_id=utt_id)
            cuts.append(
                Cut(
                    id=utt_id,
                    recording_id=rec.id,
                    start=0.0,
                    duration=rec.duration,
                    recording=rec,
                    supervisions=[
                        Supervision(
                            id=f"{utt_id}__speaker",
                            recording_id=rec.id,
                            start=0.0,
                            duration=rec.duration,
                            text="",  # speaker-id corpus: no transcripts
                            speaker=speaker,
                            language=_LANGUAGE,
                        )
                    ],
                    provenance=Provenance(
                        source_cut_id=None,
                        generated_by="cnceleb_recipe@1",
                        stage_name=ctx.stage_name,
                        created_at=datetime.now(timezone.utc),
                        pipeline_run_id=ctx.pipeline_run_id,
                    ),
                    custom={"subset": subset_name},
                )
            )
        return CutSet(cuts)

    @staticmethod
    def _iter_subset(root: Path, subset_name: str) -> list[Path]:
        """Return the FLAC paths that belong to *subset_name*.

        ``data`` walks the corpus root; ``dev`` and ``eval`` consult the
        canonical split files and resolve each line against ``data/``.
        Returns ``[]`` if the relevant on-disk structure is missing —
        partial extracts shouldn't abort the whole prepare step.
        """
        data_root = root / "data"
        if subset_name == "data":
            if not data_root.is_dir():
                return []
            return sorted(data_root.rglob("*.flac"))

        # dev / eval: read the .lst files and resolve relative paths.
        if subset_name == "dev":
            list_files = [root / "dev" / "dev.lst"]
        else:  # eval
            list_files = [
                root / "eval" / "lists" / "enroll.lst",
                root / "eval" / "lists" / "test.lst",
            ]

        paths: list[Path] = []
        for lf in list_files:
            if not lf.is_file():
                continue
            for line in lf.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                # Lines may be bare paths (vlog-01-001.flac) or include the
                # speaker prefix (id00001/vlog-01-001.flac). Be tolerant.
                candidate = data_root / line
                if candidate.suffix.lower() != ".flac":
                    candidate = candidate.with_suffix(".flac")
                paths.append(candidate)
        return paths


register_recipe(CnCelebRecipe())
