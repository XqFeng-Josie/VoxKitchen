"""Pack parquet operator: write CutSet metadata to a Parquet file.

Writes a flat Parquet table (no audio bytes) to
``<output_dir>/metadata.parquet``.  Metrics are flattened as
``metrics_<key>`` columns.
"""

from __future__ import annotations

from pathlib import Path
from typing import ClassVar

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
            rows.append(
                {
                    "id": cut.id,
                    "recording_id": cut.recording_id,
                    "audio_path": (cut.recording.sources[0].source if cut.recording else None),
                    "start": cut.start,
                    "duration": cut.duration,
                    "text": next((s.text for s in cut.supervisions if s.text), None),
                    "speaker": next((s.speaker for s in cut.supervisions if s.speaker), None),
                    "language": next((s.language for s in cut.supervisions if s.language), None),
                    **{f"metrics_{k}": v for k, v in cut.metrics.items()},
                }
            )

        table = pa.Table.from_pylist(rows)
        out_dir = Path(self.config.output_dir or str(self.ctx.stage_dir / "parquet_output"))
        out_dir.mkdir(parents=True, exist_ok=True)
        pq.write_table(table, out_dir / "metadata.parquet")

        return CutSet(list(cuts))
