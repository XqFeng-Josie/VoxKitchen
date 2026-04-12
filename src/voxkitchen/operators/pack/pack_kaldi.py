"""Pack kaldi operator: write Kaldi-style text files from a CutSet.

Writes three files to the output directory:
  wav.scp   - utterance-id  audio-path
  text      - utterance-id  transcript
  utt2spk   - utterance-id  speaker-id
"""

from __future__ import annotations

from pathlib import Path

from voxkitchen.operators.base import Operator, OperatorConfig
from voxkitchen.operators.registry import register_operator
from voxkitchen.schema.cutset import CutSet


class PackKaldiConfig(OperatorConfig):
    """Configuration for pack_kaldi."""

    output_dir: str | None = None


@register_operator
class PackKaldiOperator(Operator):
    name = "pack_kaldi"
    config_cls = PackKaldiConfig
    device = "cpu"
    produces_audio = False
    reads_audio_bytes = False

    def process(self, cuts: CutSet) -> CutSet:
        assert isinstance(self.config, PackKaldiConfig)
        out = Path(self.config.output_dir or str(self.ctx.stage_dir / "kaldi_output"))
        out.mkdir(parents=True, exist_ok=True)

        with (
            open(out / "wav.scp", "w") as wav_scp,
            open(out / "text", "w") as text_f,
            open(out / "utt2spk", "w") as u2s,
        ):
            for cut in cuts:
                audio_path = cut.recording.sources[0].source if cut.recording else "MISSING"
                wav_scp.write(f"{cut.id} {audio_path}\n")
                transcript = next((s.text for s in cut.supervisions if s.text), "")
                text_f.write(f"{cut.id} {transcript}\n")
                speaker = next((s.speaker for s in cut.supervisions if s.speaker), "unknown")
                u2s.write(f"{cut.id} {speaker}\n")

        return CutSet(list(cuts))
