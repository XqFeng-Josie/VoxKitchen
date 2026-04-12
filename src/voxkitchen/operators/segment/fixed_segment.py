"""FixedSegment operator: split a Cut into fixed-length chunks."""

from __future__ import annotations

from voxkitchen.operators.base import Operator, OperatorConfig
from voxkitchen.operators.registry import register_operator
from voxkitchen.schema.cut import Cut
from voxkitchen.schema.cutset import CutSet
from voxkitchen.schema.provenance import Provenance
from voxkitchen.utils.time import now_utc


class FixedSegmentConfig(OperatorConfig):
    segment_duration: float = 10.0  # seconds per chunk
    min_remaining: float = 0.5  # drop final chunk if shorter than this


@register_operator
class FixedSegmentOperator(Operator):
    """Split each input Cut into fixed-length child Cuts.

    This is a 1-to-many operator: one Cut in, N Cuts out. Each child shares
    the parent's ``recording`` and ``recording_id`` — no new audio is written.
    The child's ``start`` is offset within the parent's audio file, so playback
    of ``child.recording`` from ``child.start`` for ``child.duration`` seconds
    yields the correct audio slice.
    """

    name = "fixed_segment"
    config_cls = FixedSegmentConfig
    device = "cpu"
    produces_audio = False
    reads_audio_bytes = False

    def process(self, cuts: CutSet) -> CutSet:
        assert isinstance(self.config, FixedSegmentConfig)
        seg_dur = self.config.segment_duration
        min_rem = self.config.min_remaining
        generated_by = f"fixed_segment@{seg_dur}s"

        out: list[Cut] = []
        for cut in cuts:
            t = 0.0
            idx = 0
            while t < cut.duration:
                remaining = cut.duration - t
                chunk = min(seg_dur, remaining)
                if chunk < min_rem:
                    break
                out.append(
                    Cut(
                        id=f"{cut.id}__seg{idx}",
                        recording_id=cut.recording_id,
                        start=cut.start + t,
                        duration=chunk,
                        recording=cut.recording,
                        supervisions=[],
                        metrics={},
                        provenance=Provenance(
                            source_cut_id=cut.id,
                            generated_by=generated_by,
                            stage_name=getattr(getattr(self, "ctx", None), "stage_name", "unknown"),
                            created_at=now_utc(),
                            pipeline_run_id=getattr(
                                getattr(self, "ctx", None), "pipeline_run_id", "unknown"
                            ),
                        ),
                        custom=cut.custom,
                    )
                )
                t += seg_dur
                idx += 1
        return CutSet(out)
