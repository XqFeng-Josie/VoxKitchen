"""BandwidthEstimate operator: detect effective bandwidth to catch upsampled audio."""

from __future__ import annotations

from voxkitchen.operators.base import Operator, OperatorConfig
from voxkitchen.operators.registry import register_operator
from voxkitchen.schema.cut import Cut
from voxkitchen.schema.cutset import CutSet
from voxkitchen.utils.audio import load_audio_for_cut


class BandwidthEstimateConfig(OperatorConfig):
    nfft: int = 512
    hop: int = 256


@register_operator
class BandwidthEstimateOperator(Operator):
    """Estimate effective audio bandwidth and store in metrics.

    Detects files that were upsampled from a lower sample rate — e.g., an
    8 kHz telephone recording saved as 48 kHz WAV will show
    ``bandwidth_khz ≈ 4.0``.

    Computes STFT, measures mean power per frequency bin, then finds the
    frequency where energy drops sharply (ratio method). Writes:
      - ``metrics["bandwidth_khz"]``: effective bandwidth in kHz
    """

    name = "bandwidth_estimate"
    config_cls = BandwidthEstimateConfig
    device = "cpu"
    produces_audio = False
    reads_audio_bytes = True

    def setup(self) -> None:
        import torch

        self._torch = torch

    def process(self, cuts: CutSet) -> CutSet:
        return CutSet(self._process_cut(cut) for cut in cuts)

    def _process_cut(self, cut: Cut) -> Cut:
        torch = self._torch
        audio, sr = load_audio_for_cut(cut)

        if audio.ndim == 2:
            audio = audio[:, 0]

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        tensor = torch.from_numpy(audio).unsqueeze(0).to(device).float()

        assert isinstance(self.config, BandwidthEstimateConfig)
        n_fft = int(self.config.nfft / 16000 * sr)
        hop_length = int(self.config.hop / 16000 * sr)

        spec = torch.stft(
            tensor,
            n_fft=n_fft,
            hop_length=hop_length,
            window=torch.hann_window(n_fft, device=device),
            onesided=True,
            return_complex=True,
        )

        freq = torch.fft.rfftfreq(n_fft, d=1.0 / sr).to(device)
        power = spec.real.pow(2) + spec.imag.pow(2)
        mean_power = power.mean(dim=2).squeeze(0)

        # Compute ratio of cumulative energy: front bins vs. rear bins.
        # A sharp spike indicates the boundary where real content ends.
        n = mean_power.shape[0]
        prefix = torch.cumsum(mean_power, dim=0)[:-1]
        suffix = mean_power.sum() - prefix
        idx = torch.arange(1, n, dtype=torch.float32, device=device)
        ratios = ((n - idx) * prefix) / (idx * suffix + 1e-12)
        cutoff = (torch.log(ratios + 1e-12) > 12).to(torch.int).argmax()

        if cutoff > 0:
            bw_khz = float(round(freq[cutoff].item() * 2 / 1000, 2))
        else:
            bw_khz = float(round(sr / 1000, 2))

        return cut.model_copy(update={"metrics": {**cut.metrics, "bandwidth_khz": bw_khz}})
