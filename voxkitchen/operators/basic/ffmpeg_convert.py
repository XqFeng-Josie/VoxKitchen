"""Format conversion operator using ffmpeg."""

from __future__ import annotations

import re

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
    clean_names: bool = True  # simplify output filenames to {origin}_{idx}.{fmt}


@register_operator
class FfmpegConvertOperator(Operator):
    """Convert audio format using ffmpeg (e.g. opus to wav, flac to mp3)."""

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
        for idx, cut in enumerate(cuts):
            if cut.recording is None:
                raise ValueError(f"cut {cut.id!r} has no recording")
            src = cut.recording.sources[0].source
            fmt = self.config.target_format

            # Build a clean output name when clean_names is enabled:
            #   demo1__wav__svad3 → demo1_3.wav
            if self.config.clean_names:
                out_name = self._clean_name(cut, idx)
            else:
                out_name = cut.id
            out_path = derived_dir / f"{out_name}.{fmt}"

            inp = ffmpeg.input(src, ss=cut.start, t=cut.duration)
            inp.output(str(out_path)).overwrite_output().run(quiet=True)

            new_rec = recording_from_file(out_path, recording_id=out_name)
            # Track where this segment lives in the *original* source, not
            # in whatever intermediate recording we're slicing right now.
            # Chaining offsets (parent_origin_start + cut.start) keeps the
            # value correct through to_wav → vad → extract and through
            # deeper chains (e.g. trim → resample → vad → extract).
            custom = dict(cut.custom) if cut.custom else {}
            parent_origin_start = float(custom.get("origin_start", 0.0))
            custom["origin_start"] = round(parent_origin_start + cut.start, 3)
            custom["origin_end"] = round(parent_origin_start + cut.start + cut.duration, 3)
            out_cuts.append(
                Cut(
                    id=out_name,
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
                    custom=custom,
                )
            )
        return CutSet(out_cuts)

    @staticmethod
    def _clean_name(cut: Cut, idx: int) -> str:
        """Derive a clean filename: ``{origin}_{idx}``.

        Strips all operator-appended ``__suffix`` parts from the cut id
        to recover the original source name, then appends the index.
        """
        base = cut.id
        # Strip operator suffixes: __wav, __svad3, __rs16000, etc.
        base = re.sub(r"(__[a-z]+\d*)+$", "", base)
        base = re.sub(r"^rec-", "", base)
        return f"{base}_{idx}" if base else f"cut_{idx}"
