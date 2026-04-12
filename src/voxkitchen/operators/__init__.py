"""VoxKitchen operators: transformations from CutSet to CutSet."""

from voxkitchen.operators.base import Operator, OperatorConfig

# Register all built-in operators by importing them. Every built-in module
# must be imported here so that ``get_operator(...)`` can find it at runtime.
from voxkitchen.operators.basic import channel_merge as _basic_channel_merge  # noqa: F401
from voxkitchen.operators.basic import ffmpeg_convert as _basic_ffmpeg  # noqa: F401
from voxkitchen.operators.basic import resample as _basic_resample  # noqa: F401
from voxkitchen.operators.noop import identity as _noop_identity  # noqa: F401
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
