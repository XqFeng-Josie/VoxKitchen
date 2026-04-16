"""Volume perturbation operator: apply random gain to audio.

Applies a random gain (in dB) uniformly sampled from [min_gain_db, max_gain_db].
The gain is deterministic per cut (seeded from cut ID hash) so reruns produce
the same result. Output is clipped to [-1.0, 1.0].
"""

from __future__ import annotations

import hashlib

import numpy as np

from voxkitchen.operators.base import Operator, OperatorConfig
from voxkitchen.operators.registry import register_operator
from voxkitchen.schema.cut import Cut
from voxkitchen.schema.cutset import CutSet
from voxkitchen.schema.provenance import Provenance
from voxkitchen.utils.audio import load_audio_for_cut, recording_from_file, save_audio
from voxkitchen.utils.time import now_utc


class VolumePerturbConfig(OperatorConfig):
    min_gain_db: float = -6.0
    max_gain_db: float = 6.0


@register_operator
class VolumePerturbOperator(Operator):
    """Apply random volume gain within a dB range."""

    name = "volume_perturb"
    config_cls = VolumePerturbConfig
    device = "cpu"
    produces_audio = True
    reads_audio_bytes = True

    def process(self, cuts: CutSet) -> CutSet:
        assert isinstance(self.config, VolumePerturbConfig)
        derived_dir = self.ctx.stage_dir / "derived"
        derived_dir.mkdir(parents=True, exist_ok=True)

        out_cuts: list[Cut] = []
        for cut in cuts:
            audio, sr = load_audio_for_cut(cut)

            # Deterministic gain from cut ID
            seed = int(hashlib.sha256(cut.id.encode()).hexdigest()[:8], 16)
            rng = np.random.RandomState(seed)
            gain_db = round(rng.uniform(self.config.min_gain_db, self.config.max_gain_db), 1)

            gain_linear = 10.0 ** (gain_db / 20.0)
            perturbed = np.clip(audio * gain_linear, -1.0, 1.0).astype(np.float32)

            tag = f"vol{gain_db:+.1f}dB".replace(".", "p").replace("+", "pos").replace("-", "neg")
            out_path = derived_dir / f"{cut.id}__{tag}.wav"
            save_audio(out_path, perturbed, sr)
            new_rec = recording_from_file(out_path, recording_id=f"{cut.recording_id}_{tag}")

            custom = dict(cut.custom) if cut.custom else {}
            custom["volume_gain_db"] = gain_db

            out_cuts.append(
                Cut(
                    id=f"{cut.id}__{tag}",
                    recording_id=new_rec.id,
                    start=0.0,
                    duration=new_rec.duration,
                    recording=new_rec,
                    supervisions=cut.supervisions,
                    metrics=cut.metrics,
                    provenance=Provenance(
                        source_cut_id=cut.id,
                        generated_by=f"volume_perturb@{gain_db:+.1f}dB",
                        stage_name=self.ctx.stage_name,
                        created_at=now_utc(),
                        pipeline_run_id=self.ctx.pipeline_run_id,
                    ),
                    custom=custom,
                )
            )
        return CutSet(out_cuts)
