"""VoxKitchen operators: transformations from CutSet to CutSet."""

from voxkitchen.operators.base import Operator, OperatorConfig

# Register all built-in operators by importing them. Every built-in module
# must be imported here so that ``get_operator(...)`` can find it at runtime.
from voxkitchen.operators.basic import channel_merge as _basic_channel_merge  # noqa: F401
from voxkitchen.operators.basic import ffmpeg_convert as _basic_ffmpeg  # noqa: F401
from voxkitchen.operators.basic import loudness_normalize as _basic_loudness  # noqa: F401
from voxkitchen.operators.basic import resample as _basic_resample  # noqa: F401
from voxkitchen.operators.noop import identity as _noop_identity  # noqa: F401
from voxkitchen.operators.pack import pack_manifest as _pack_manifest  # noqa: F401
from voxkitchen.operators.quality import audio_fingerprint_dedup as _qual_dedup  # noqa: F401
from voxkitchen.operators.quality import duration_filter as _qual_duration  # noqa: F401
from voxkitchen.operators.quality import quality_score_filter as _qual_filter  # noqa: F401
from voxkitchen.operators.quality import snr_estimate as _qual_snr  # noqa: F401
from voxkitchen.operators.registry import (
    MissingExtrasError,
    UnknownOperatorError,
    get_operator,
    list_operators,
    register_operator,
)
from voxkitchen.operators.segment import fixed_segment as _seg_fixed  # noqa: F401
from voxkitchen.operators.segment import silence_split as _seg_silence  # noqa: F401
from voxkitchen.operators.segment import webrtc_vad as _seg_webrtc  # noqa: F401

__all__ = [
    "MissingExtrasError",
    "Operator",
    "OperatorConfig",
    "UnknownOperatorError",
    "get_operator",
    "list_operators",
    "register_operator",
]
