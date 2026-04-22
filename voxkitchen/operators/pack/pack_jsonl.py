"""Pack JSONL operator: write a flat, human-readable JSONL manifest.

Each line is a JSON object with fields:
  id, origin_id, start, end, duration, sample_rate,
  text, gender, speaker, language   — first non-None across supervisions
  supervisions                       — JSON array of all supervisions
  <metric>                           — one field per cut.metrics entry

``text`` / ``language`` / ``speaker`` are the first non-None values across all
supervisions (backward-compatible shortcut for single-ASR pipelines).
``supervisions`` is a JSON-encoded array containing every Supervision on the
cut — use it to access per-model results when multiple ASR or LangID operators
have run.
"""

from __future__ import annotations

import json
from pathlib import Path

from voxkitchen.operators.base import Operator, OperatorConfig
from voxkitchen.operators.registry import register_operator
from voxkitchen.schema.cut import Cut
from voxkitchen.schema.cutset import CutSet


class PackJsonlConfig(OperatorConfig):
    output_path: str | None = None  # defaults to <stage_dir>/manifest.jsonl


def _derive_origin_id(cut: Cut) -> str:
    """Extract the original source filename from the cut id.

    Operator suffixes follow the ``__<op><idx>`` convention. We strip
    them all to recover the ingest-time name.

    Example: ``demo1__wav__svad3__wav`` → ``demo1``
    """
    import re

    base = cut.id
    # Strip all operator-appended suffixes: __wav, __svad3, __rs16000, __lufs23, etc.
    base = re.sub(r"(__[a-z]+\d*)+$", "", base)
    # Also strip leading "rec-" prefix if present
    base = re.sub(r"^rec-", "", base)
    return base or cut.id


def _make_id(cut: Cut) -> str:
    """Use the audio filename (without extension) as id.

    This ensures the JSONL id matches the file on disk exactly.
    Falls back to the cut id if no recording is available.
    """
    if cut.recording and cut.recording.sources:
        from pathlib import PurePosixPath

        return PurePosixPath(cut.recording.sources[0].source).stem
    return cut.id


_GENDER_MAP = {"m": "male", "f": "female", "o": "unknown"}


def _flatten_cut(cut: Cut) -> dict[str, object]:
    """Extract a flat dict from a Cut for JSONL output."""
    text = next((s.text for s in cut.supervisions if s.text), "")
    raw_gender = next((s.gender for s in cut.supervisions if s.gender), "")
    gender = _GENDER_MAP.get(raw_gender, raw_gender or "unknown")
    speaker = next((s.speaker for s in cut.supervisions if s.speaker), "")
    language = next((s.language for s in cut.supervisions if s.language), "")
    sample_rate = cut.recording.sampling_rate if cut.recording else 0

    custom = cut.custom or {}
    origin_start = custom.get("origin_start", round(cut.start, 3))
    origin_end = custom.get("origin_end", round(cut.start + cut.duration, 3))

    row: dict[str, object] = {
        "id": _make_id(cut),
        "origin_id": _derive_origin_id(cut),
        "start": origin_start,
        "end": origin_end,
        "duration": round(cut.duration, 3),
        "sample_rate": sample_rate,
        "text": text,
        "gender": gender,
        "speaker": speaker,
        "language": language,
        "supervisions": [
            {
                "id": s.id,
                "text": s.text,
                "language": s.language,
                "speaker": s.speaker,
                "gender": s.gender,
                "custom": s.custom,
            }
            for s in cut.supervisions
        ],
    }
    for k, v in cut.metrics.items():
        row[k] = round(v, 3) if isinstance(v, float) else v
    return row


@register_operator
class PackJsonlOperator(Operator):
    """Write a flat JSONL manifest — one JSON object per line.

    Fields: id, origin_id, start, end, duration, sample_rate,
    text, snr, gender (male/female/unknown), speaker, language.

    ``start``/``end`` are the VAD segment boundaries in the original
    recording.  ``origin_id`` traces back to the source filename.
    """

    name = "pack_jsonl"
    config_cls = PackJsonlConfig
    device = "cpu"
    produces_audio = False
    reads_audio_bytes = False

    def process(self, cuts: CutSet) -> CutSet:
        assert isinstance(self.config, PackJsonlConfig)
        out_path = Path(self.config.output_path or str(self.ctx.stage_dir / "manifest.jsonl"))
        out_path.parent.mkdir(parents=True, exist_ok=True)

        with open(out_path, "w", encoding="utf-8") as f:
            for cut in cuts:
                row = _flatten_cut(cut)
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

        return CutSet(list(cuts))
