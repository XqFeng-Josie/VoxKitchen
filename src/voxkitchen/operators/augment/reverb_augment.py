"""Reverb augmentation operator: convolve speech with Room Impulse Responses.

Picks a random RIR file from ``rir_dir``, convolves with the input audio
using FFT-based convolution, and trims to the original signal length.
Peak-normalizes the output to prevent clipping.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import fftconvolve

from voxkitchen.operators.base import Operator, OperatorConfig
from voxkitchen.operators.registry import register_operator
from voxkitchen.schema.cut import Cut
from voxkitchen.schema.cutset import CutSet
from voxkitchen.schema.provenance import Provenance
from voxkitchen.utils.audio import (
    AUDIO_EXTENSIONS,
    load_audio_for_cut,
    recording_from_file,
    save_audio,
)
from voxkitchen.utils.time import now_utc


class ReverbAugmentConfig(OperatorConfig):
    rir_dir: str
    normalize: bool = True


@register_operator
class ReverbAugmentOperator(Operator):
    name = "reverb_augment"
    config_cls = ReverbAugmentConfig
    device = "cpu"
    produces_audio = True
    reads_audio_bytes = True

    _rir_files: list[Path]

    def setup(self) -> None:
        assert isinstance(self.config, ReverbAugmentConfig)
        rir_root = Path(self.config.rir_dir)
        self._rir_files = sorted(
            p
            for p in rir_root.rglob("*")
            if p.suffix.lower() in AUDIO_EXTENSIONS and p.is_file()
        )
        if not self._rir_files:
            raise FileNotFoundError(f"no audio files found in rir_dir: {rir_root}")

    def process(self, cuts: CutSet) -> CutSet:
        assert isinstance(self.config, ReverbAugmentConfig)
        derived_dir = self.ctx.stage_dir / "derived"
        derived_dir.mkdir(parents=True, exist_ok=True)

        out_cuts: list[Cut] = []
        for cut in cuts:
            audio, sr = load_audio_for_cut(cut)

            # Deterministic RNG from cut ID
            seed = int(hashlib.sha256(cut.id.encode()).hexdigest()[:8], 16)
            rng = np.random.RandomState(seed)

            # Pick RIR file
            rir_idx = rng.randint(0, len(self._rir_files))
            rir_path = self._rir_files[rir_idx]

            # Load and prepare RIR
            rir, rir_sr = sf.read(str(rir_path), dtype="float32")
            if rir.ndim == 2:
                rir = rir.mean(axis=1).astype(np.float32)
            if rir_sr != sr:
                from scipy.signal import resample as scipy_resample

                new_len = int(len(rir) * sr / rir_sr)
                rir = scipy_resample(rir, new_len).astype(np.float32)

            # Convolve and trim to original length
            convolved = fftconvolve(audio, rir, mode="full")[: len(audio)].astype(np.float32)

            # Normalize to prevent clipping
            if self.config.normalize:
                peak = np.abs(convolved).max()
                if peak > 1.0:
                    convolved = convolved / (peak + 1e-8)

            tag = "reverb"
            out_path = derived_dir / f"{cut.id}__{tag}.wav"
            save_audio(out_path, convolved, sr)
            new_rec = recording_from_file(out_path, recording_id=f"{cut.recording_id}_{tag}")

            custom = dict(cut.custom) if cut.custom else {}
            custom["rir_file"] = rir_path.name

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
                        generated_by=f"reverb_augment@{rir_path.name}",
                        stage_name=self.ctx.stage_name,
                        created_at=now_utc(),
                        pipeline_run_id=self.ctx.pipeline_run_id,
                    ),
                    custom=custom,
                )
            )
        return CutSet(out_cuts)
