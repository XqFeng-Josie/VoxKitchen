"""Map operator names to their canonical Docker image tag.

Uses ``voxkitchen.cli.doctor.EXPECTED_OPERATORS`` as the source of truth:
the smallest image group that contains a given op is its canonical image.

The slim image's operator group is named `core` internally; this resolver
returns the user-facing tag `slim` instead.
"""

from __future__ import annotations


class UnknownOperatorError(Exception):
    """Raised when an op name is not registered in voxkitchen at all."""


# Order matters: smallest/cheapest image first. The resolver picks the first
# match. `core` → `slim` is the only group-vs-tag rename.
_GROUP_TO_TAG: list[tuple[str, str]] = [
    ("core", "slim"),
    ("asr", "asr"),
    ("diarize", "diarize"),
    ("tts", "tts"),
    ("fish-speech", "fish-speech"),
]


def image_for_op(op_name: str) -> str:
    """Return the canonical image tag that hosts ``op_name``.

    Raises ``UnknownOperatorError`` if the op is not registered in voxkitchen.
    Falls back to ``"latest"`` for registered ops not in any image group spec
    (the union image always contains everything that's been built).
    """
    # Lazy import — voxkitchen.cli.doctor pulls in typer; we want this resolver
    # to stay importable from any context.
    from voxkitchen.cli.doctor import EXPECTED_OPERATORS
    from voxkitchen.operators.registry import list_operators

    registered = set(list_operators())
    if op_name not in registered:
        raise UnknownOperatorError(f"operator {op_name!r} is not registered in voxkitchen")

    # _GROUP_TO_TAG is smallest-first, so the first group that claims the op is
    # its canonical (cheapest) image.
    for group, tag in _GROUP_TO_TAG:
        if op_name in EXPECTED_OPERATORS.get(group, set()):
            return tag

    # Registered but no image group claims it — must be in latest (union).
    return "latest"
