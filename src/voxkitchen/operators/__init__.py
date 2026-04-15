"""VoxKitchen operators: transformations from CutSet to CutSet."""

from voxkitchen.operators.base import Operator, OperatorConfig
from voxkitchen.operators.basic import channel_merge as _basic_channel_merge  # noqa: F401

# --- basic (core deps) ---
from voxkitchen.operators.basic import ffmpeg_convert as _basic_ffmpeg  # noqa: F401
from voxkitchen.operators.basic import loudness_normalize as _basic_loudness  # noqa: F401
from voxkitchen.operators.basic import resample as _basic_resample  # noqa: F401

# Register all built-in operators by importing them. Every built-in module
# must be imported here so that ``get_operator(...)`` can find it at runtime.
# Optional-dependency operators are wrapped in try/except so that a missing
# pip extra never prevents the core package from loading.
from voxkitchen.operators.noop import identity as _noop_identity  # noqa: F401

# --- segment (optional: webrtcvad, librosa) ---
from voxkitchen.operators.segment import fixed_segment as _seg_fixed  # noqa: F401

try:
    from voxkitchen.operators.segment import webrtc_vad as _seg_webrtc  # noqa: F401
except ImportError:
    pass  # webrtcvad not installed
try:
    from voxkitchen.operators.segment import silence_split as _seg_silence  # noqa: F401
except ImportError:
    pass  # librosa not installed
try:
    from voxkitchen.operators.segment import silero_vad as _seg_silero  # noqa: F401
except ImportError:
    pass  # torch not installed

# --- quality (core + optional) ---
from voxkitchen.operators.quality import clipping_detect as _qual_clip  # noqa: F401
from voxkitchen.operators.quality import duration_filter as _qual_duration  # noqa: F401
from voxkitchen.operators.quality import quality_score_filter as _qual_filter  # noqa: F401
from voxkitchen.operators.quality import snr_estimate as _qual_snr  # noqa: F401

try:
    from voxkitchen.operators.quality import bandwidth_estimate as _qual_bw  # noqa: F401
except ImportError:
    pass  # torch not installed
try:
    from voxkitchen.operators.quality import pitch_stats as _qual_pitch  # noqa: F401
except ImportError:
    pass  # pyworld not installed
try:
    from voxkitchen.operators.quality import dnsmos_score as _qual_dnsmos  # noqa: F401
except ImportError:
    pass  # speechmos not installed
try:
    from voxkitchen.operators.quality import utmos_score as _qual_utmos  # noqa: F401
except ImportError:
    pass  # speechmos not installed
try:
    from voxkitchen.operators.quality import audio_fingerprint_dedup as _qual_dedup  # noqa: F401
except ImportError:
    pass  # simhash / librosa not installed

# --- annotate (optional: faster-whisper, whisperx, funasr, pyannote, speechbrain, etc.) ---
try:
    from voxkitchen.operators.annotate import faster_whisper_asr as _annotate_fwasr  # noqa: F401
except ImportError:
    pass  # faster-whisper not installed
try:
    from voxkitchen.operators.annotate import whisperx_asr as _annotate_whisperx  # noqa: F401
except ImportError:
    pass  # whisperx not installed
try:
    from voxkitchen.operators.annotate import paraformer_asr as _annotate_paraformer  # noqa: F401
    from voxkitchen.operators.annotate import sensevoice_asr as _annotate_sensevoice  # noqa: F401
except ImportError:
    pass  # funasr not installed
try:
    from voxkitchen.operators.annotate import whisper_openai_asr as _annotate_owasr  # noqa: F401
except ImportError:
    pass  # openai-whisper not installed
try:
    from voxkitchen.operators.annotate import whisper_langid as _annotate_langid_w  # noqa: F401
except ImportError:
    pass  # openai-whisper / faster-whisper not installed
try:
    from voxkitchen.operators.annotate import wenet_asr as _annotate_wenet  # noqa: F401
except ImportError:
    pass  # wenet not installed
try:
    from voxkitchen.operators.annotate import pyannote_diarize as _annotate_diar  # noqa: F401
except ImportError:
    pass  # pyannote.audio not installed
try:
    from voxkitchen.operators.annotate import speechbrain_langid as _annotate_langid  # noqa: F401
except ImportError:
    pass  # speechbrain not installed

from voxkitchen.operators.annotate import gender_classify as _annotate_gender  # noqa: F401

# --- augment (optional: torch/torchaudio for speed_perturb) ---
try:
    from voxkitchen.operators.augment import speed_perturb as _aug_speed  # noqa: F401
except ImportError:
    pass  # torch not installed
from voxkitchen.operators.augment import noise_augment as _aug_noise  # noqa: F401
from voxkitchen.operators.augment import volume_perturb as _aug_volume  # noqa: F401

# --- pack (optional: datasets, webdataset, pyarrow) ---
from voxkitchen.operators.pack import pack_manifest as _pack_manifest  # noqa: F401

try:
    from voxkitchen.operators.pack import pack_huggingface as _pack_huggingface  # noqa: F401
except ImportError:
    pass  # datasets not installed
try:
    from voxkitchen.operators.pack import pack_webdataset as _pack_webdataset  # noqa: F401
except ImportError:
    pass  # webdataset not installed
try:
    from voxkitchen.operators.pack import pack_parquet as _pack_parquet  # noqa: F401
except ImportError:
    pass  # pyarrow not installed

from voxkitchen.operators.pack import pack_jsonl as _pack_jsonl  # noqa: F401
from voxkitchen.operators.pack import pack_kaldi as _pack_kaldi  # noqa: F401
from voxkitchen.operators.registry import (
    MissingExtrasError,
    UnknownOperatorError,
    get_operator,
    list_operators,
    register_operator,
)

__all__ = [
    "MissingExtrasError",
    "Operator",
    "OperatorConfig",
    "UnknownOperatorError",
    "get_operator",
    "list_operators",
    "register_operator",
]
