"""Pack parquet operator: write CutSet metadata to a Parquet file.

Writes a flat Parquet table (no audio bytes) to
``<output_dir>/metadata.parquet``.

Column layout
-------------
Core columns (always present):
  id, recording_id, audio_path, start, duration

Supervision columns:
  text, speaker, language
    — first non-None value across all supervisions (backward-compatible shortcut
      for the common single-ASR-step case).
  supervisions
    — full JSON array of every Supervision on the cut, including ``custom``
      sub-fields (emotion, audio_event, word_alignments, …). Use this column
      when multiple ASR operators have run or when per-supervision metadata
      (emotion, audio_event) must be preserved.

Metrics columns:
  metrics_<key>  — one column per entry in cut.metrics (snr, cer, …)

Custom columns:
  custom_<key>   — one column per entry in cut.custom (scalar values are stored
                   as-is; complex objects are JSON-serialised strings).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, ClassVar

from voxkitchen.operators.base import Operator, OperatorConfig
from voxkitchen.operators.registry import register_operator
from voxkitchen.schema.cutset import CutSet


class PackParquetConfig(OperatorConfig):
    """Configuration for pack_parquet."""

    output_dir: str | None = None


@register_operator
class PackParquetOperator(Operator):
    """Export CutSet as Apache Parquet with audio file references."""

    name = "pack_parquet"
    config_cls = PackParquetConfig
    device = "cpu"
    produces_audio = False
    reads_audio_bytes = False
    required_extras: ClassVar[list[str]] = ["pack"]

    def process(self, cuts: CutSet) -> CutSet:
        assert isinstance(self.config, PackParquetConfig)
        import pyarrow as pa
        import pyarrow.parquet as pq

        rows = []
        for cut in cuts:
            # ------------------------------------------------------------------
            # Backward-compat flat supervision columns (first non-None wins)
            # ------------------------------------------------------------------
            text = next((s.text for s in cut.supervisions if s.text), None)
            speaker = next((s.speaker for s in cut.supervisions if s.speaker), None)
            language = next((s.language for s in cut.supervisions if s.language), None)

            # ------------------------------------------------------------------
            # Full supervision list — preserves all ASR outputs + per-supervision
            # metadata (emotion, audio_event, word_alignments, …)
            # ------------------------------------------------------------------
            supervisions_json = json.dumps(
                [_supervision_to_dict(s) for s in cut.supervisions],
                ensure_ascii=False,
            )

            # ------------------------------------------------------------------
            # cut.custom — flatten scalar values, JSON-serialise the rest
            # ------------------------------------------------------------------
            custom_cols: dict[str, Any] = {
                f"custom_{k}": v
                if isinstance(v, (str, int, float, bool)) or v is None
                else json.dumps(v, ensure_ascii=False)
                for k, v in cut.custom.items()
            }

            rows.append(
                {
                    "id": cut.id,
                    "recording_id": cut.recording_id,
                    "audio_path": (cut.recording.sources[0].source if cut.recording else None),
                    "start": cut.start,
                    "duration": cut.duration,
                    "text": text,
                    "speaker": speaker,
                    "language": language,
                    "supervisions": supervisions_json,
                    **{f"metrics_{k}": v for k, v in cut.metrics.items()},
                    **custom_cols,
                }
            )

        table = pa.Table.from_pylist(rows)
        out_dir = Path(self.config.output_dir or str(self.ctx.stage_dir / "parquet_output"))
        out_dir.mkdir(parents=True, exist_ok=True)
        pq.write_table(table, out_dir / "metadata.parquet")

        return CutSet(list(cuts))


def _supervision_to_dict(sup: Any) -> dict[str, Any]:
    """Serialise a Supervision to a plain dict suitable for JSON embedding."""
    return {
        "id": sup.id,
        "start": sup.start,
        "duration": sup.duration,
        "text": sup.text,
        "language": sup.language,
        "speaker": sup.speaker,
        "gender": sup.gender,
        "custom": sup.custom,
    }
