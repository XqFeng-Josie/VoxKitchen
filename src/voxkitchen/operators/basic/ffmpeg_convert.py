"""Format conversion operator using ffmpeg."""

from __future__ import annotations

import ffmpeg

from voxkitchen.operators.base import Operator, OperatorConfig
from voxkitchen.operators.registry import register_operator
from voxkitchen.schema.cut import Cut
from voxkitchen.schema.cutset import CutSet
from voxkitchen.schema.provenance import Provenance
from voxkitchen.utils.audio import recording_from_file
from voxkitchen.utils.time import now_utc


class FfmpegConvertConfig(OperatorConfig):
    target_format: str = "wav"


@register_operator
class FfmpegConvertOperator(Operator):
    name = "ffmpeg_convert"
    config_cls = FfmpegConvertConfig
    device = "cpu"
    produces_audio = True
    reads_audio_bytes = True

    def process(self, cuts: CutSet) -> CutSet:
        assert isinstance(self.config, FfmpegConvertConfig)
        derived_dir = self.ctx.stage_dir / "derived"
        derived_dir.mkdir(parents=True, exist_ok=True)

        out_cuts: list[Cut] = []
        for cut in cuts:
            if cut.recording is None:
                raise ValueError(f"cut {cut.id!r} has no recording")
            src = cut.recording.sources[0].source
            fmt = self.config.target_format
            out_path = derived_dir / f"{cut.id}.{fmt}"

            (ffmpeg.input(src).output(str(out_path)).overwrite_output().run(quiet=True))

            new_rec = recording_from_file(out_path, recording_id=f"{cut.recording.id}_{fmt}")
            out_cuts.append(
                Cut(
                    id=f"{cut.id}__{fmt}",
                    recording_id=new_rec.id,
                    start=0.0,
                    duration=new_rec.duration,
                    recording=new_rec,
                    supervisions=cut.supervisions,
                    metrics=cut.metrics,
                    provenance=Provenance(
                        source_cut_id=cut.id,
                        generated_by=f"ffmpeg_convert@{fmt}",
                        stage_name=self.ctx.stage_name,
                        created_at=now_utc(),
                        pipeline_run_id=self.ctx.pipeline_run_id,
                    ),
                    custom=cut.custom,
                )
            )
        return CutSet(out_cuts)
