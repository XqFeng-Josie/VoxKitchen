"""CommonVoice recipe: parse a local CommonVoice directory into a CutSet."""

from __future__ import annotations

import csv
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from voxkitchen.ingest.recipes import register_recipe
from voxkitchen.ingest.recipes.base import Recipe
from voxkitchen.schema.cut import Cut
from voxkitchen.schema.cutset import CutSet
from voxkitchen.schema.provenance import Provenance
from voxkitchen.schema.supervision import Supervision
from voxkitchen.utils.audio import recording_from_file

if TYPE_CHECKING:
    from voxkitchen.pipeline.context import RunContext

_GENDER_MAP: dict[str, Literal["m", "f"]] = {
    "male_masculine": "m",
    "female_feminine": "f",
}


class CommonVoiceRecipe(Recipe):
    name = "commonvoice"

    def prepare(self, root: Path, subsets: list[str] | None, ctx: RunContext) -> CutSet:
        tsv_files = self._resolve_tsvs(root, subsets)
        cuts: list[Cut] = []
        for tsv_path in tsv_files:
            subset_name = tsv_path.stem
            with tsv_path.open(encoding="utf-8", newline="") as fh:
                reader = csv.DictReader(fh, delimiter="\t")
                for row in reader:
                    audio_path = root / "clips" / row["path"]
                    if not audio_path.exists():
                        continue
                    utt_id = audio_path.stem
                    rec = recording_from_file(audio_path, recording_id=utt_id)
                    gender = _GENDER_MAP.get(row.get("gender", ""))
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
                                    text=row.get("sentence"),
                                    speaker=row.get("client_id"),
                                    language=row.get("locale") or None,
                                    gender=gender,
                                )
                            ],
                            provenance=Provenance(
                                source_cut_id=None,
                                generated_by="commonvoice_recipe@1",
                                stage_name=ctx.stage_name,
                                created_at=datetime.now(timezone.utc),
                                pipeline_run_id=ctx.pipeline_run_id,
                            ),
                            custom={"subset": subset_name},
                        )
                    )
        return CutSet(cuts)

    def _resolve_tsvs(self, root: Path, subsets: list[str] | None) -> list[Path]:
        if subsets:
            return [root / f"{s}.tsv" for s in subsets]
        return sorted(root.glob("*.tsv"))


register_recipe(CommonVoiceRecipe())
