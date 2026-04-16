"""Pack HuggingFace operator: write a HuggingFace Dataset with an audio column.

Saves the dataset to disk at ``<output_dir>``.  Each sample has fields:
  id, audio, text, speaker, duration.
"""

from __future__ import annotations

from pathlib import Path
from typing import ClassVar

from voxkitchen.operators.base import Operator, OperatorConfig
from voxkitchen.operators.registry import register_operator
from voxkitchen.schema.cutset import CutSet


class PackHuggingFaceConfig(OperatorConfig):
    """Configuration for pack_huggingface."""

    output_dir: str | None = None


@register_operator
class PackHuggingFaceOperator(Operator):
    """Export CutSet as a HuggingFace Dataset with audio column."""

    name = "pack_huggingface"
    config_cls = PackHuggingFaceConfig
    device = "cpu"
    produces_audio = True
    reads_audio_bytes = True
    required_extras: ClassVar[list[str]] = ["pack"]

    def process(self, cuts: CutSet) -> CutSet:
        assert isinstance(self.config, PackHuggingFaceConfig)
        from datasets import Audio, Dataset

        records = []
        for cut in cuts:
            audio_path = cut.recording.sources[0].source if cut.recording else None
            records.append(
                {
                    "id": cut.id,
                    "audio": audio_path,
                    "text": next((s.text for s in cut.supervisions if s.text), None),
                    "speaker": next((s.speaker for s in cut.supervisions if s.speaker), None),
                    "duration": cut.duration,
                }
            )

        ds = Dataset.from_list(records)
        ds = ds.cast_column("audio", Audio())
        out_dir = Path(self.config.output_dir or str(self.ctx.stage_dir / "hf_output"))
        ds.save_to_disk(str(out_dir))

        return CutSet(list(cuts))
