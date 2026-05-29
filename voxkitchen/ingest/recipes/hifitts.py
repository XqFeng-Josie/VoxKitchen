"""Hi-Fi TTS recipe: parse the Hi-Fi Multi-Speaker English TTS corpus.

Hi-Fi TTS is a ~292-hour (clean) 44.1 kHz English TTS corpus from 11
LibriVox audiobook readers, published on OpenSLR resource 109 under
CC BY 4.0. It is the standard open multi-speaker English TTS dataset.

Directory layout inside ``hi_fi_tts_v0.tar.gz``::

    hi_fi_tts_v0/
      audio/<reader_id>/<book_name>/<utt>.flac
      <reader_id>_manifest_clean_train.json
      <reader_id>_manifest_clean_dev.json
      <reader_id>_manifest_clean_test.json
      <reader_id>_manifest_other_train.json   # only for "other" readers
      ...

Each manifest is JSON Lines; one utterance per line with keys::

    {
      "audio_filepath": "audio/<reader_id>/<book>/<utt>.flac",
      "duration": 4.5,
      "text": "...",
      "text_normalized": "...",
      "speaker": "<reader_id>"        # sometimes "reader_id" instead
    }

We expose three subsets — ``clean-train`` / ``clean-dev`` / ``clean-test``
— plus their ``other-*`` siblings (lower-quality four-reader partition).
The recipe globs across all per-reader manifests for the chosen subset
and emits one Cut per JSONL line.
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

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
# Subset names follow the LibriTTS-style hyphenated convention so the
# CLI stays consistent. They map to the upstream ``<partition>_<split>``
# manifest suffixes (``clean_train`` etc.).
_OFFICIAL_SUBSETS = (
    "clean-train",
    "clean-dev",
    "clean-test",
    "other-train",
    "other-dev",
    "other-test",
)

# Manifest filename pattern: ``<reader_id>_manifest_<partition>_<split>.json``
# reader_id is a numeric LibriVox ID.
_MANIFEST_RE = re.compile(
    r"^(?P<reader>\d+)_manifest_(?P<partition>clean|other)_(?P<split>train|dev|test)\.json$"
)


class HiFiTTSRecipe(Recipe):
    """Parse Hi-Fi TTS into a CutSet."""

    name = "hifitts"
    # OpenSLR resource 109. Single tarball (~41 GB) — no per-subset
    # selection at the download level, so we expose one entry.
    download_urls = {
        "default": ["https://www.openslr.org/resources/109/hi_fi_tts_v0.tar.gz"],
    }
    # HEAD-probed Content-Length (2026-05).
    download_sizes = {
        "default": 41_352_291_414,
    }

    def prepare(self, root: Path, subsets: list[str] | None, ctx: RunContext) -> CutSet:
        # Tarball extracts to <root>/hi_fi_tts_v0/. Tolerate "user
        # pointed at the inner directory" too.
        ds_dir = root / "hi_fi_tts_v0"
        effective_root = ds_dir if ds_dir.is_dir() else root

        target_subsets = subsets or list(_OFFICIAL_SUBSETS)
        for s in target_subsets:
            if s not in _OFFICIAL_SUBSETS:
                raise ValueError(
                    f"unknown subset {s!r} for hifitts. Available: {list(_OFFICIAL_SUBSETS)}"
                )

        # Index manifest files once: subset → list of (reader_id, manifest_path).
        manifests_by_subset: dict[str, list[tuple[str, Path]]] = {s: [] for s in _OFFICIAL_SUBSETS}
        for path in sorted(effective_root.glob("*.json")):
            m = _MANIFEST_RE.match(path.name)
            if not m:
                continue
            subset_key = f"{m['partition']}-{m['split']}"
            manifests_by_subset[subset_key].append((m["reader"], path))

        cuts: list[Cut] = []
        for subset_name in target_subsets:
            for reader_id, manifest_path in manifests_by_subset[subset_name]:
                for row in self._read_jsonl(manifest_path):
                    audio_rel = row.get("audio_filepath")
                    text = row.get("text_normalized") or row.get("text") or ""
                    if not audio_rel:
                        # Skip rows missing the only field we can't synthesize.
                        continue
                    audio_path = effective_root / audio_rel
                    if not audio_path.is_file():
                        # Skip rather than abort — partial extracts happen.
                        continue
                    utt_id = audio_path.stem
                    speaker = row.get("speaker") or row.get("reader_id") or reader_id
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
                                    speaker=str(speaker),
                                    language=_LANGUAGE,
                                )
                            ],
                            provenance=Provenance(
                                source_cut_id=None,
                                generated_by="hifitts_recipe@1",
                                stage_name=ctx.stage_name,
                                created_at=datetime.now(timezone.utc),
                                pipeline_run_id=ctx.pipeline_run_id,
                            ),
                            custom={
                                "subset": subset_name,
                                "reader_id": str(reader_id),
                            },
                        )
                    )
        return CutSet(cuts)

    @staticmethod
    def _read_jsonl(path: Path) -> list[dict[str, Any]]:
        """Parse a JSON Lines manifest, skipping unparsable rows.

        Hi-Fi TTS manifests are dense JSONL — one object per line — but
        the occasional trailing blank line or BOM crops up in mirrors.
        Skip rather than abort so a single bad line doesn't sink the
        whole subset.
        """
        rows: list[dict[str, Any]] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip().lstrip("﻿")
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict):
                rows.append(obj)
        return rows


register_recipe(HiFiTTSRecipe())
