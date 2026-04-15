"""Noise augmentation operator: mix signal with background noise at random SNR.

Picks a random noise file from ``noise_dir``, trims or loops it to match
the signal length, and mixes at a random SNR from ``snr_range``.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import soundfile as sf

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


class NoiseAugmentConfig(OperatorConfig):
    noise_dir: str
    snr_range: list[float] = [5.0, 20.0]


@register_operator
class NoiseAugmentOperator(Operator):
    name = "noise_augment"
    config_cls = NoiseAugmentConfig
    device = "cpu"
    produces_audio = True
    reads_audio_bytes = True

    _noise_files: list[Path]

    def setup(self) -> None:
        assert isinstance(self.config, NoiseAugmentConfig)
        noise_root = Path(self.config.noise_dir)
        self._noise_files = sorted(
            p
            for p in noise_root.rglob("*")
            if p.suffix.lower() in AUDIO_EXTENSIONS and p.is_file()
        )
        if not self._noise_files:
            raise FileNotFoundError(f"no audio files found in noise_dir: {noise_root}")

    def process(self, cuts: CutSet) -> CutSet:
        assert isinstance(self.config, NoiseAugmentConfig)
        derived_dir = self.ctx.stage_dir / "derived"
        derived_dir.mkdir(parents=True, exist_ok=True)
        snr_lo, snr_hi = self.config.snr_range[0], self.config.snr_range[1]

        out_cuts: list[Cut] = []
        for cut in cuts:
            audio, sr = load_audio_for_cut(cut)

            # Deterministic RNG from cut ID
            seed = int(hashlib.sha256(cut.id.encode()).hexdigest()[:8], 16)
            rng = np.random.RandomState(seed)

            # Pick noise file and SNR
            noise_idx = rng.randint(0, len(self._noise_files))
            snr_db = round(float(rng.uniform(snr_lo, snr_hi)), 1)

            noise = self._load_noise(self._noise_files[noise_idx], sr, len(audio), rng)
            mixed = self._mix_at_snr(audio, noise, snr_db)

            tag = f"noise_snr{snr_db:.0f}dB"
            out_path = derived_dir / f"{cut.id}__{tag}.wav"
            save_audio(out_path, mixed, sr)
            new_rec = recording_from_file(out_path, recording_id=f"{cut.recording_id}_{tag}")

            custom = dict(cut.custom) if cut.custom else {}
            custom["noise_snr_db"] = snr_db
            custom["noise_file"] = self._noise_files[noise_idx].name

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
                        generated_by=f"noise_augment@{snr_db}dB",
                        stage_name=self.ctx.stage_name,
                        created_at=now_utc(),
                        pipeline_run_id=self.ctx.pipeline_run_id,
                    ),
                    custom=custom,
                )
            )
        return CutSet(out_cuts)

    @staticmethod
    def _load_noise(
        noise_path: Path, target_sr: int, target_len: int, rng: np.random.RandomState
    ) -> np.ndarray:  # type: ignore[type-arg]
        """Load noise, resample if needed, trim or loop to target_len."""
        noise, noise_sr = sf.read(str(noise_path), dtype="float32")
        # Convert stereo to mono
        if noise.ndim == 2:
            noise = noise.mean(axis=1).astype(np.float32)

        # Resample if needed
        if noise_sr != target_sr:
            from scipy.signal import resample as scipy_resample

            new_len = int(len(noise) * target_sr / noise_sr)
            noise = scipy_resample(noise, new_len).astype(np.float32)

        # Trim or loop
        if len(noise) >= target_len:
            max_start = len(noise) - target_len
            start = rng.randint(0, max(1, max_start + 1))
            noise = noise[start : start + target_len]
        else:
            repeats = (target_len // len(noise)) + 1
            noise = np.tile(noise, repeats)[:target_len]

        return noise

    @staticmethod
    def _mix_at_snr(
        signal: np.ndarray, noise: np.ndarray, snr_db: float  # type: ignore[type-arg]
    ) -> np.ndarray:  # type: ignore[type-arg]
        """Mix signal and noise at the specified SNR (dB)."""
        sig_power = float(np.mean(signal**2))
        noise_power = float(np.mean(noise**2))
        if noise_power < 1e-10:
            return signal.copy()
        target_noise_power = sig_power / (10.0 ** (snr_db / 10.0))
        scale = np.sqrt(target_noise_power / noise_power)
        mixed = signal + noise * scale
        return np.clip(mixed, -1.0, 1.0).astype(np.float32)
