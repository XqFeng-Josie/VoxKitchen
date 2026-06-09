"""Pre-flight: does a pipeline's operators fit the chosen Docker image?

Two layered checks, run when the user passes an explicit ``--tag``:

A. yaml-vs-contract (in-source, instant): every stage op must be in
   EXPECTED_OPERATORS for the image's operator group. Catches
   "wrong image for this op" before any container launches.

B. yaml-vs-pulled-image (queries the local image, when present): every
   stage op must be a key in the image's /opt/voxkitchen/op_env_map.json.
   Catches published-image lag — an op the source contract claims is in
   the image but the actual pulled image lacks (built before the op was
   added). Skipped with a note when the image isn't pulled locally.
   Also skips ops already flagged by Check A — an op that belongs in a
   different image entirely should not get a second, contradictory
   "rebuild this image" error from Check B.
"""

from __future__ import annotations

import dataclasses
from typing import Any


@dataclasses.dataclass
class ImagePreflightResult:
    errors: list[str] = dataclasses.field(default_factory=list)
    warnings: list[str] = dataclasses.field(default_factory=list)
    notes: list[str] = dataclasses.field(
        default_factory=list
    )  # informational (e.g. Check B skipped)

    @property
    def ok(self) -> bool:
        return not self.errors


def canonical_image_for_op(op_name: str) -> str:
    """Return the smallest image tag whose operator group contains op_name.

    Mirrors the sweep's image_for_op: walk groups core→asr→diarize→tts→
    fish-speech, return the tag (core→slim) of the first group containing
    op_name. Returns 'latest' if no group claims it (union-only op).

    Intentionally returns 'latest' for unknown ops rather than raising —
    unlike scripts/sweep/image_resolver.image_for_op which raises on
    unregistered ops.  A pre-flight check should never hard-fail on an op
    it doesn't recognise; the user can still attempt a run with --tag latest.
    """
    from voxkitchen.cli.doctor import EXPECTED_OPERATORS

    # (group_name, image_tag) in smallest-first order. core→slim is the rename.
    order = [
        ("core", "slim"),
        ("asr", "asr"),
        ("diarize", "diarize"),
        ("tts", "tts"),
        ("fish-speech", "fish-speech"),
    ]
    for group, tag in order:
        if op_name in EXPECTED_OPERATORS.get(group, set()):
            return tag
    return "latest"


def check_image_preflight(
    spec: Any, tag: str, *, image_name: str | None = None
) -> ImagePreflightResult:
    """Run Check A (always) + Check B (when the image is local) for ``spec``
    against image ``tag``. ``spec`` is a PipelineSpec with ``.stages`` each
    having ``.name`` and ``.op``.
    """
    result = ImagePreflightResult()
    stage_ops = [(s.name, s.op) for s in spec.stages]

    flagged_by_a = _check_a(stage_ops, tag, result)
    _check_b(stage_ops, tag, result, image_name=image_name, skip_ops=flagged_by_a)
    return result


def _check_a(stage_ops: list[tuple[str, str]], tag: str, result: ImagePreflightResult) -> set[str]:
    """In-source: stage ops must be in EXPECTED_OPERATORS for the tag's group.

    Returns the set of op names that were flagged as errors, so Check B can
    skip them (Check B's "rebuild the image" advice would be misleading for
    an op that simply belongs in a different image entirely).
    """
    from voxkitchen.cli.doctor import _IMAGE_TAG_TO_GROUP, EXPECTED_OPERATORS

    flagged: set[str] = set()

    group = _IMAGE_TAG_TO_GROUP.get(tag, tag)
    if group not in EXPECTED_OPERATORS:
        # `latest` (union — everything expected) or an unknown tag. Skip A.
        if tag == "latest":
            result.notes.append("Check A skipped: 'latest' is the union image (all ops expected).")
        else:
            result.notes.append(f"Check A skipped: no operator group known for tag {tag!r}.")
        return flagged

    # Union in core ops for non-core groups whose images bundle a core env.
    # asr/diarize/tts are already pre-unioned in doctor.py.  fish-speech is
    # the sole group defined without the union, but its image dispatches core
    # ops to a bundled core env (confirmed empirically: op_env_map.json
    # inside the fish-speech image contains resample, silero_vad, etc.).
    # Applying the union defensively here costs nothing for the other groups
    # and prevents false-positive errors on core+fish_speech mixed pipelines.
    expected = EXPECTED_OPERATORS[group]
    if group != "core":
        expected = expected | EXPECTED_OPERATORS["core"]

    for stage_name, op in stage_ops:
        if op not in expected:
            belongs = canonical_image_for_op(op)
            result.errors.append(
                f"stage {stage_name!r}: operator {op!r} is not in image {tag!r} "
                f"(group {group!r}). It belongs in image {belongs!r} — "
                f"use --tag {belongs} or --tag latest, or remove the stage."
            )
            flagged.add(op)

    return flagged


def _check_b(
    stage_ops: list[tuple[str, str]],
    tag: str,
    result: ImagePreflightResult,
    *,
    image_name: str | None,
    skip_ops: set[str] | None = None,
) -> None:
    """Query the local image's op_env_map.json; stage ops must be keys in it.

    ``skip_ops`` is the set of op names already flagged by Check A.  Ops in
    that set are skipped here — a "rebuild the image" suggestion would be
    misleading for an op that simply belongs in a different image entirely.
    """
    import json
    import subprocess

    from voxkitchen.cli.docker_cmd import DEFAULT_IMAGE

    _skip = skip_ops or set()
    image = image_name or f"{DEFAULT_IMAGE}:{tag}"

    # Is the image local?
    try:
        inspect = subprocess.run(
            ["docker", "image", "inspect", image],
            capture_output=True,
            text=True,
            timeout=10,
        )
    except subprocess.TimeoutExpired:
        result.notes.append(f"Check B skipped: `docker image inspect {image}` timed out (10s).")
        return
    if inspect.returncode != 0:
        result.notes.append(
            f"Check B skipped: image {image!r} not pulled — "
            f"published-image verification unavailable (pull it to enable)."
        )
        return

    # Read op_env_map.json from inside the image.
    try:
        cat = subprocess.run(
            [
                "docker",
                "run",
                "--rm",
                "--entrypoint",
                "cat",
                image,
                "/opt/voxkitchen/op_env_map.json",
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
    except subprocess.TimeoutExpired:
        result.notes.append(
            f"Check B skipped: reading op_env_map.json from {image} timed out (30s)."
        )
        return
    if cat.returncode != 0:
        result.notes.append(
            f"Check B skipped: could not read /opt/voxkitchen/op_env_map.json "
            f"from image {image!r} (dev image, or path moved)."
        )
        return

    try:
        op_env_map = json.loads(cat.stdout)
    except json.JSONDecodeError:
        result.notes.append(
            f"Check B skipped: op_env_map.json in image {image!r} was not valid JSON."
        )
        return

    available = set(op_env_map.keys())
    for stage_name, op in stage_ops:
        if op in _skip:
            continue
        if op not in available:
            result.errors.append(
                f"stage {stage_name!r}: operator {op!r} is NOT in the pulled "
                f"image {tag!r}'s op_env_map.json — the image predates this "
                f"operator. Rebuild the image from current source "
                f"(vkit docker build --tag {tag}), or use --tag latest."
            )
