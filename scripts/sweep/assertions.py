"""Per-operator assertion functions for the operator sweep.

Each assertion has signature ``(work_dir: Path, run_log: str) -> tuple[bool, str]``.
The bool is the verdict; the str is a one-line message for the report.
Assertions must NOT raise — wrap any errors and return ``(False, "...")``.

The ASSERTIONS dispatch dict at the bottom maps op-name → assertion.
Operators without an entry fall through to ``default_smoke_assertion`` (cuts > 0).
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path

from voxkitchen.schema.cutset import CutSet

_logger = logging.getLogger(__name__)

# ---- Helpers ----


def _read_final_cuts(work_dir: Path) -> list:
    """Locate and load the final pack stage's manifest. Returns [] on missing.

    Deserialization errors (schema mismatch, malformed cut record, etc.) are
    logged at WARNING level and converted to ``[]`` so the assertion contract
    (must-not-raise) is preserved. The log entry lets a developer see the
    real cause without re-running the pipeline.
    """
    pack_dirs = sorted(work_dir.glob("*_pack"))
    if not pack_dirs:
        return []
    manifest = pack_dirs[-1] / "cuts.jsonl.gz"
    if not manifest.exists():
        return []
    try:
        return list(CutSet.from_jsonl_gz(manifest))
    except Exception as exc:
        _logger.warning("_read_final_cuts(%s): %s: %s", manifest, type(exc).__name__, exc)
        return []


# ---- Core assertions ----


def default_smoke_assertion(work_dir: Path, _log: str) -> tuple[bool, str]:
    """Fallback: final stage produced ≥1 cut without crash."""
    cuts = _read_final_cuts(work_dir)
    return (len(cuts) > 0, f"{len(cuts)} cuts out")


def assert_resample_target_sr(work_dir: Path, _log: str) -> tuple[bool, str]:
    """Output cuts have sampling_rate == 16000 (the default target in sweep yamls)."""
    cuts = _read_final_cuts(work_dir)
    if not cuts:
        return False, "no cuts produced"
    bad = [c.id for c in cuts if c.recording and c.recording.sampling_rate != 16000]
    if bad:
        return False, f"{len(bad)} cuts with sr ≠ 16000"
    return True, f"{len(cuts)} cuts at 16000 Hz"


def assert_vad_segments(work_dir: Path, _log: str) -> tuple[bool, str]:
    """VAD on a 60s file → ≥1 segment, max segment ≤30s."""
    cuts = _read_final_cuts(work_dir)
    if len(cuts) < 1:
        return False, "expected ≥1 VAD segment"
    longest = max(c.duration for c in cuts)
    if longest > 30:
        return False, f"segment > 30s ({longest:.1f}s)"
    return True, f"{len(cuts)} VAD segments, max {longest:.1f}s"


def assert_asr_nonempty(work_dir: Path, _log: str) -> tuple[bool, str]:
    """ASR → at least one supervision with non-empty text."""
    cuts = _read_final_cuts(work_dir)
    texts = [s.text for c in cuts for s in c.supervisions if s.text]
    if not texts:
        return False, "no transcripts produced"
    words = sum(len(t.split()) for t in texts)
    return True, f"{len(texts)} transcripts, {words} words"


def assert_tts_produces_audio(work_dir: Path, _log: str) -> tuple[bool, str]:
    """TTS → at least one cut with audio > 0.5s."""
    cuts = _read_final_cuts(work_dir)
    audios = [c for c in cuts if c.recording and c.duration > 0.5]
    if not audios:
        return False, "no audio produced (or all < 0.5s)"
    total = sum(c.duration for c in cuts)
    return True, f"{len(audios)} cuts, total {total:.1f}s"


def assert_normalize_text_strips(work_dir: Path, _log: str) -> tuple[bool, str]:
    """Output text must not contain SenseVoice <|...|> tags or doubled spaces."""
    cuts = _read_final_cuts(work_dir)
    bad = [
        s.text
        for c in cuts
        for s in c.supervisions
        if s.text and ("<|" in s.text or "  " in s.text)
    ]
    if bad:
        return False, f"{len(bad)} texts still contain markup or double-space"
    return True, "all texts normalized"


def assert_diarize_speakers(min_speakers: int = 2) -> Callable[[Path, str], tuple[bool, str]]:
    """Closure: assert at least min_speakers distinct speaker labels."""

    def check(work_dir: Path, _log: str) -> tuple[bool, str]:
        cuts = _read_final_cuts(work_dir)
        speakers = {s.speaker for c in cuts for s in c.supervisions if s.speaker}
        if len(speakers) < min_speakers:
            return False, f"only {len(speakers)} distinct speakers (need ≥{min_speakers})"
        return True, f"{len(speakers)} distinct speakers"

    return check


def assert_speaker_embed_dim(work_dir: Path, _log: str) -> tuple[bool, str]:
    """speaker_embed writes custom.speaker_embedding (a list) with consistent dim."""
    cuts = _read_final_cuts(work_dir)
    with_emb = [c for c in cuts if c.custom.get("speaker_embedding")]
    if not with_emb:
        return False, "no embeddings written"
    dims = {len(c.custom["speaker_embedding"]) for c in with_emb}
    if len(dims) != 1:
        return False, f"inconsistent embedding dims: {dims}"
    return True, f"{len(with_emb)} embeddings, dim={dims.pop()}"


def assert_speaker_similarity(work_dir: Path, _log: str) -> tuple[bool, str]:
    """speaker_similarity writes a metric in [-1, 1]."""
    cuts = _read_final_cuts(work_dir)
    sims = [c.metrics.get("speaker_similarity") for c in cuts if "speaker_similarity" in c.metrics]
    if not sims:
        return False, "no similarity metric written"
    if any(s < -1.0 or s > 1.0 for s in sims):
        return False, "similarity out of [-1, 1]"
    return True, f"{len(sims)} similarities in valid range"


def assert_metric_written(metric_name: str) -> Callable[[Path, str], tuple[bool, str]]:
    """Factory: assert at least one cut has metrics[metric_name]."""

    def check(work_dir: Path, _log: str) -> tuple[bool, str]:
        cuts = _read_final_cuts(work_dir)
        with_m = [c for c in cuts if metric_name in c.metrics]
        if not with_m:
            return False, f"no '{metric_name}' written"
        return True, f"{len(with_m)} cuts with {metric_name}"

    return check


def assert_speed_perturb_produces_cuts(work_dir: Path, _log: str) -> tuple[bool, str]:
    """speed_perturb fans out each cut into multiple speed variants.

    Without a reference input duration we can't validate the actual scaling
    ratios from work_dir alone, so this assertion only confirms the operator
    produced cuts. Tighter validation belongs in tests/unit/operators/augment/.
    """
    cuts = _read_final_cuts(work_dir)
    if not cuts:
        return False, "no cuts"
    durations = [c.duration for c in cuts]
    return True, f"{len(cuts)} cuts, mean dur {sum(durations) / len(cuts):.1f}s"


def assert_pack_file_exists(
    stage_name: str = "01_pack",
) -> Callable[[Path, str], tuple[bool, str]]:
    """Factory: assert the named pack stage's cuts.jsonl.gz exists + non-empty."""

    def check(work_dir: Path, _log: str) -> tuple[bool, str]:
        path = work_dir / stage_name / "cuts.jsonl.gz"
        if not path.exists():
            return False, f"{stage_name}/{path.name} not written"
        size = path.stat().st_size
        if size < 100:
            return False, f"{stage_name}/{path.name} suspiciously small: {size} bytes"
        return True, f"{stage_name}/{path.name} written: {size} bytes"

    return check


# ---- Dispatch table ----
#
# Populated incrementally as Tasks 5-10 add per-op pipelines. Anything not
# in this dict falls through to default_smoke_assertion (cuts > 0), which
# is sufficient for pure-transform ops.

ASSERTIONS: dict[str, Callable[[Path, str], tuple[bool, str]]] = {
    # Audio
    "resample": assert_resample_target_sr,
    "ffmpeg_convert": default_smoke_assertion,
    "channel_merge": default_smoke_assertion,
    # loudness_normalize normalises audio in-place but does NOT write a
    # loudness_lufs metric entry; smoke assertion (cuts > 0) is correct here.
    "loudness_normalize": default_smoke_assertion,
    "identity": default_smoke_assertion,
    # Segmentation
    "silero_vad": assert_vad_segments,
    "webrtc_vad": assert_vad_segments,
    "fixed_segment": assert_vad_segments,
    "silence_split": assert_vad_segments,
}
