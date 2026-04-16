"""FLEURS recipe: download and parse Google's 102-language speech dataset.

FLEURS (Few-shot Learning Evaluation of Universal Representations of Speech)
is available on HuggingFace. This recipe uses the ``datasets`` library to
download audio and transcriptions, then converts them into a CutSet.

Usage::

    # Download
    vkit download fleurs --root /data/fleurs --subsets en_us,zh_cn

    # In pipeline YAML
    ingest:
      source: recipe
      recipe: fleurs
      args:
        root: /data/fleurs
        subsets: [en_us]
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

import soundfile as sf

from voxkitchen.ingest.recipes import register_recipe
from voxkitchen.ingest.recipes.base import Recipe
from voxkitchen.schema.cut import Cut
from voxkitchen.schema.cutset import CutSet
from voxkitchen.schema.provenance import Provenance
from voxkitchen.schema.recording import AudioSource, Recording
from voxkitchen.schema.supervision import Supervision

if TYPE_CHECKING:
    from voxkitchen.pipeline.context import RunContext

logger = logging.getLogger(__name__)


class FleursRecipe(Recipe):
    name = "fleurs"

    def download(self, root: Path, subsets: list[str] | None) -> None:
        """Download FLEURS via HuggingFace datasets library."""
        from datasets import load_dataset

        languages = subsets or ["en_us"]
        root.mkdir(parents=True, exist_ok=True)
        for lang in languages:
            logger.info("downloading FLEURS language: %s", lang)
            ds = load_dataset(
                "google/fleurs",
                lang,
                cache_dir=str(root / ".cache"),
                trust_remote_code=True,
            )
            ds.save_to_disk(str(root / lang))
            logger.info("saved %s to %s", lang, root / lang)

    def prepare(self, root: Path, subsets: list[str] | None, ctx: RunContext) -> CutSet:
        """Parse downloaded FLEURS data into CutSet."""
        from datasets import load_from_disk

        languages = subsets or self._discover_languages(root)
        cuts: list[Cut] = []

        for lang in languages:
            lang_dir = root / lang
            if not lang_dir.is_dir():
                logger.warning("language dir not found: %s, skipping", lang_dir)
                continue

            ds = load_from_disk(str(lang_dir))
            for split_name in ds:
                for row in ds[split_name]:
                    cut = self._row_to_cut(row, lang, split_name, ctx)
                    if cut is not None:
                        cuts.append(cut)

        return CutSet(cuts)

    def _row_to_cut(
        self,
        row: dict[str, Any],
        lang: str,
        split: str,
        ctx: RunContext,  # type: ignore[type-arg]
    ) -> Cut | None:
        """Convert a HuggingFace dataset row to a Cut."""
        audio_info = row.get("audio", {})
        audio_path = audio_info.get("path")
        if not audio_path or not Path(audio_path).exists():
            return None

        utt_id = f"fleurs-{lang}-{row.get('id', 0)}"
        text = row.get("transcription", "") or row.get("raw_transcription", "")
        sr = audio_info.get("sampling_rate", 16000)

        try:
            info = sf.info(audio_path)
            duration = info.duration
            num_samples = info.frames
            num_channels = info.channels
            sr = info.samplerate
        except Exception:
            # Fallback: estimate from array if available
            arr = audio_info.get("array")
            if arr is not None:
                import numpy as np

                duration = float(len(np.asarray(arr))) / sr
                num_samples = len(np.asarray(arr))
            else:
                return None
            num_channels = 1

        rec = Recording(
            id=utt_id,
            sources=[AudioSource(type="file", channels=[0], source=audio_path)],
            sampling_rate=sr,
            num_samples=num_samples,
            duration=duration,
            num_channels=num_channels,
        )

        return Cut(
            id=utt_id,
            recording_id=rec.id,
            start=0.0,
            duration=duration,
            recording=rec,
            supervisions=[
                Supervision(
                    id=f"{utt_id}__text",
                    recording_id=rec.id,
                    start=0.0,
                    duration=duration,
                    text=text,
                    language=lang.split("_")[0],  # en_us → en
                )
            ],
            provenance=Provenance(
                source_cut_id=None,
                generated_by="fleurs_recipe@1",
                stage_name=ctx.stage_name,
                created_at=datetime.now(timezone.utc),
                pipeline_run_id=ctx.pipeline_run_id,
            ),
            custom={"subset": split, "fleurs_language": lang},
        )

    def _discover_languages(self, root: Path) -> list[str]:
        return sorted(p.name for p in root.iterdir() if p.is_dir() and p.name != ".cache")


register_recipe(FleursRecipe())
