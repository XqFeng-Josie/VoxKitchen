"""Pack WebDataset operator: write audio + metadata to tar shards.

Creates WebDataset-compatible tar files at ``<output_dir>/shard-NNNNN.tar``.
Each sample contains:
  ``<key>.audio.wav``       - raw WAV bytes (if the file exists on disk)
  ``<key>.metadata.json``   - JSON-encoded cut metadata
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import ClassVar

from voxkitchen.operators.base import Operator, OperatorConfig
from voxkitchen.operators.registry import register_operator
from voxkitchen.schema.cutset import CutSet


class PackWebDatasetConfig(OperatorConfig):
    """Configuration for pack_webdataset."""

    output_dir: str | None = None
    shard_size: int = 1000


@register_operator
class PackWebDatasetOperator(Operator):
    name = "pack_webdataset"
    config_cls = PackWebDatasetConfig
    device = "cpu"
    produces_audio = True
    reads_audio_bytes = True
    required_extras: ClassVar[list[str]] = ["pack"]

    def process(self, cuts: CutSet) -> CutSet:
        assert isinstance(self.config, PackWebDatasetConfig)
        import webdataset as wds

        out_dir = Path(self.config.output_dir or str(self.ctx.stage_dir / "wds_output"))
        out_dir.mkdir(parents=True, exist_ok=True)
        pattern = str(out_dir / "shard-%05d.tar")

        with wds.ShardWriter(pattern, maxcount=self.config.shard_size) as sink:
            for cut in cuts:
                audio_path = cut.recording.sources[0].source if cut.recording else None
                sample: dict[str, object] = {"__key__": cut.id}
                if audio_path and Path(audio_path).exists():
                    with open(audio_path, "rb") as f:
                        sample["audio.wav"] = f.read()
                sample["metadata.json"] = json.dumps(
                    {
                        "id": cut.id,
                        "duration": cut.duration,
                        "text": next((s.text for s in cut.supervisions if s.text), None),
                    }
                ).encode()
                sink.write(sample)

        return CutSet(list(cuts))
