"""CN-Celeb 1 recipe: parse a local CN-Celeb directory into a CutSet.

CN-Celeb (Fan et al., 2020; v2 dataset card on OpenSLR resource 82) is
the standard open Chinese speaker-recognition benchmark and the
counterpart to the gated VoxCeleb. It contains roughly 130k utterances
from ~1000 speakers across 11 genres (interview, vlog, singing, drama,
…). The dataset is distributed as a single ~21 GB tarball
(``cn-celeb_v2.tar.gz``).

The recipe is intentionally lean: CN-Celeb is **non-transcribed**
(speaker-identity research target, not ASR), so each Cut is emitted
with an empty-text Supervision that carries only speaker + language
tags. Subset selection follows the canonical splits the corpus ships:

- ``data`` (default) — every FLAC under ``data/``. Useful for
  speaker-embedding extraction across the whole corpus.
- ``dev`` — the development split for training speaker-embedding
  models. ``dev/dev.lst`` lists ~800 **speaker IDs** (one per line);
  the recipe expands each ID to that speaker's utterances under
  ``data/<spk>/*.flac``. Note: dev shares audio with ``data`` — it is
  a filter on the same recordings, not a separate audio store.
- ``eval`` — the verification eval set. The corpus stores eval audio
  in ``eval/enroll/`` (one enrollment recording per eval speaker) and
  ``eval/test/`` (~17k test recordings); both directories are
  **separate** FLAC files from ``data/``, not pointers into it.

Directory layout produced by extracting ``cn-celeb_v2.tar.gz``::

    CN-Celeb_flac/
      README.TXT
      1911.01799.pdf                  # the CN-Celeb paper
      data/                           # training-side audio
        id00000/
          singing-01-001.flac, vlog-01-001.flac, …
        …
        id00999/
      dev/
        dev.lst                       # speaker IDs only (e.g. ``id00000``)
      eval/
        enroll/                       # flat dir of enrollment FLACs
          id00800-enroll.flac, …
        test/                         # flat dir of test trial FLACs
          id00800-singing-01-001.flac, …
        lists/
          enroll.lst                  # "<utt-id> <relative-path>" per line
          test.lst                    # "<relative-path>" per line
          trials.lst                  # speaker-verification trial pairs

Speakers are extracted from the path layout: under
``data/<spk>/<utt>.flac`` the parent dir name is the speaker id;
under ``eval/enroll/`` and ``eval/test/`` the speaker id is the
dash-prefix of the filename (``id00800-enroll.flac`` → ``id00800``).
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
    # HEAD-probed Content-Length (2026-05); 20.74 GB compressed.
    download_sizes = {
        "cn-celeb_v2": 22_264_439_915,
    }

    def prepare(self, root: Path, subsets: list[str] | None, ctx: RunContext) -> CutSet:
        # The tarball historically extracts to a top-level directory whose
        # exact name has varied between releases (CN-Celeb_flac, CN-Celeb,
        # CN-Celeb_v2). Probe the common ones; fall back to the user-supplied
        # root if none match.
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

        # Collect (path, source_subset_name) pairs. Dedup on resolved path so
        # ['data', 'dev'] doesn't emit dev utterances twice — they're a
        # subset of the data/ audio, not separate files.
        seen: set[Path] = set()
        flac_paths: list[tuple[Path, str]] = []
        for subset_name in target:
            for flac in self._iter_subset(effective_root, subset_name):
                resolved = flac.resolve()
                if resolved in seen or not flac.is_file():
                    continue
                seen.add(resolved)
                flac_paths.append((flac, subset_name))

        cuts: list[Cut] = []
        for flac, subset_name in flac_paths:
            speaker = self._speaker_from_path(flac)
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

        - ``data``: walk every FLAC under ``data/``.
        - ``dev``: read speaker IDs from ``dev/dev.lst``, then walk each
          ``data/<spk>/*.flac`` — dev is a filter on the data/ audio.
        - ``eval``: walk ``eval/enroll/*.flac`` plus ``eval/test/*.flac``.
          These are separate FLAC files from ``data/`` — enrolment and
          verification trial recordings. The ``eval/lists/*.lst`` files
          are NOT consulted because they reference ``.wav`` extensions
          while the on-disk files are ``.flac``; walking the directories
          directly is both more robust and matches the corpus invariant.

        Missing on-disk structure yields ``[]`` rather than raising — a
        partial extract should still be ingestable for whatever it does
        contain.
        """
        data_root = root / "data"
        if subset_name == "data":
            if not data_root.is_dir():
                return []
            return sorted(data_root.rglob("*.flac"))

        if subset_name == "dev":
            dev_lst = root / "dev" / "dev.lst"
            if not dev_lst.is_file() or not data_root.is_dir():
                return []
            paths: list[Path] = []
            for line in dev_lst.read_text(encoding="utf-8").splitlines():
                speaker = line.strip()
                if not speaker or speaker.startswith("#"):
                    continue
                spk_dir = data_root / speaker
                if spk_dir.is_dir():
                    paths.extend(sorted(spk_dir.glob("*.flac")))
            return paths

        # eval
        paths_eval: list[Path] = []
        for sub in ("enroll", "test"):
            d = root / "eval" / sub
            if d.is_dir():
                paths_eval.extend(sorted(d.glob("*.flac")))
        return paths_eval

    @staticmethod
    def _speaker_from_path(path: Path) -> str:
        """Extract the speaker id given a FLAC's position in the corpus tree.

        - ``data/<spk>/<utt>.flac`` → parent directory name (``<spk>``).
        - ``eval/enroll/<spk>-enroll.flac`` and
          ``eval/test/<spk>-<utt>.flac`` → the speaker id is encoded as
          the dash-prefix of the filename (``id00800-enroll`` →
          ``id00800``). The flat ``enroll/`` and ``test/`` directories
          don't carry the speaker in the path, only in the filename.
        """
        if path.parent.name in ("enroll", "test"):
            stem = path.stem
            dash = stem.find("-")
            return stem[:dash] if dash > 0 else stem
        return path.parent.name


register_recipe(CnCelebRecipe())
