"""Map operator names to Python environments and detect the current env.

The pipeline runner uses this to decide whether a stage can run in-process
or must be dispatched to a subprocess in a different env.

**Design constraints**

1. ``resolve_env(op_name)`` must work in the parent (core) env even when
   the operator class itself cannot be imported (its deps live in another
   env). So we never ``import`` the operator here — we consult a static
   map instead.

2. Lookup order:

   a. ``$VKIT_OP_ENV_MAP`` env var — test/override hook
   b. ``/opt/voxkitchen/op_env_map.json`` — the map generated at Docker
      image build time by merging per-env schema dumps
   c. In-process fallback derived from registered operators'
      ``required_extras`` plus :data:`EXTRA_TO_ENV`. This is the dev-mode
      path: ``pip install -e .`` users have one env only, everything
      resolves to it, and the subprocess path is never taken.

3. The mapping is **deterministic and declarative**. Every extras group in
   ``pyproject.toml`` must appear in :data:`EXTRA_TO_ENV` — adding a new
   group without classifying it is a build-time error caught by
   :mod:`voxkitchen.runtime.dump_schemas`.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from functools import lru_cache
from pathlib import Path
from typing import Final

logger = logging.getLogger(__name__)

# Which env each extras group belongs to. Must stay in sync with the extras
# installed in each env in the Dockerfile.
#
# Rule of thumb: if installing two extras together creates a resolver
# conflict or pins fight over torch/numpy/transformers, they belong in
# different envs. See docs/architecture/multi-env.md §"Why".
EXTRA_TO_ENV: Final[dict[str, str]] = {
    # core env — CPU-friendly, no heavy ML stacks
    "audio": "core",
    "segment": "core",
    "quality": "core",
    "pack": "core",
    "pitch": "core",
    "dnsmos": "core",
    "classify": "core",
    "enhance": "core",
    "codec": "core",
    "speaker": "core",
    "viz": "core",
    "viz-panel": "core",
    # asr env — faster-whisper + funasr + align (torch 2.4, numpy<2)
    "asr": "asr",
    "whisper": "asr",
    "funasr": "asr",
    "align": "asr",
    "wenet": "asr",
    # diarize env — pyannote 3.x, split out so users who only need
    # speaker diarization don't have to pull the full ASR stack.
    "diarize": "diarize",
    # tts env — modelscope + chattts + kokoro
    "tts-kokoro": "tts",
    "tts-chattts": "tts",
    "tts-cosyvoice": "tts",
    # fish-speech has its own env: upstream pins torch 2.8 + numpy 2.1
    # which can't co-exist with the tts env's torch 2.4 stack.
    "tts-fish-speech": "fish-speech",
    # gender intentionally unmapped. inaSpeechSegmenter drags tensorflow[and-cuda]
    # + onnxruntime-gpu which conflict with both asr and tts stacks. The
    # gender_classify operator still works via method=f0 (no deps) or
    # method=speechbrain (uses 'classify' extras → core).
}

KNOWN_ENVS: Final[frozenset[str]] = frozenset(
    {"core", "asr", "diarize", "tts", "fish-speech"}
)

# Root under which each venv lives. Tests may monkeypatch this to point
# at a temporary layout; production images always use the Docker path.
ENVS_DIR: Path = Path("/opt/voxkitchen/envs")

# Default op→env map path inside the Docker image. Absence is not an error —
# it just means we're in dev mode.
DEFAULT_OP_ENV_MAP_PATH: Final[Path] = Path("/opt/voxkitchen/op_env_map.json")


class EnvResolutionError(RuntimeError):
    """Raised when an operator cannot be mapped to exactly one env."""


@lru_cache(maxsize=1)
def _load_op_env_map() -> dict[str, str] | None:
    """Read op_env_map.json from $VKIT_OP_ENV_MAP or the default path.

    Returns ``None`` if neither path exists; callers then fall back to the
    registry-derived map.
    """
    override = os.environ.get("VKIT_OP_ENV_MAP")
    paths: list[Path] = []
    if override:
        paths.append(Path(override))
    paths.append(DEFAULT_OP_ENV_MAP_PATH)

    for p in paths:
        if p.is_file():
            try:
                data = json.loads(p.read_text(encoding="utf-8"))
            except Exception as exc:  # noqa: BLE001 — malformed map is advisory
                logger.warning("failed to parse %s: %s — falling back to registry", p, exc)
                continue
            if not isinstance(data, dict):
                logger.warning("%s is not a JSON object — ignoring", p)
                continue
            return {str(k): str(v) for k, v in data.items()}
    return None


@lru_cache(maxsize=1)
def _derive_from_registry() -> dict[str, str]:
    """Build the op→env map from the in-process operator registry.

    Used in dev mode (local ``pip install -e .``) where there's no prebuilt
    map. An operator with no extras maps to ``"core"``. An operator whose
    extras span multiple envs raises at lookup time; this should be caught
    in :mod:`dump_schemas` at image build time.
    """
    # Import lazily to avoid circular imports: env_resolver is imported
    # during pipeline setup; operators pull in the registry.
    from voxkitchen.operators.registry import list_operators

    # Importing the package registers whatever this env can load.
    import voxkitchen.operators  # noqa: F401  (side effect)

    mapping: dict[str, str] = {}
    for name in list_operators():
        from voxkitchen.operators.registry import get_operator

        op_cls = get_operator(name)
        extras = list(op_cls.required_extras)
        if not extras:
            mapping[name] = "core"
            continue
        envs = {EXTRA_TO_ENV[e] for e in extras if e in EXTRA_TO_ENV}
        unknown = [e for e in extras if e not in EXTRA_TO_ENV]
        if unknown:
            raise EnvResolutionError(
                f"operator {name!r} requires unknown extras {unknown!r}; "
                f"add them to EXTRA_TO_ENV in env_resolver.py"
            )
        if len(envs) > 1:
            raise EnvResolutionError(
                f"operator {name!r} spans envs {sorted(envs)}; split its extras "
                f"or move them into a single env"
            )
        mapping[name] = envs.pop()
    return mapping


def resolve_env(op_name: str) -> str:
    """Return the env name that should run operator ``op_name``.

    Raises :class:`EnvResolutionError` if the op is unknown in both the
    prebuilt map and the local registry.
    """
    prebuilt = _load_op_env_map()
    if prebuilt is not None and op_name in prebuilt:
        return prebuilt[op_name]
    derived = _derive_from_registry()
    if op_name in derived:
        return derived[op_name]
    raise EnvResolutionError(
        f"unknown operator {op_name!r} — not in op_env_map.json and not registered"
    )


def current_env() -> str:
    """Return the env name the current Python process is running in.

    Lookup order:

    1. ``$VKIT_ENV`` (set by the Docker venv activation) — trusted if non-empty
    2. Parse ``sys.executable``: ``/opt/voxkitchen/envs/<name>/bin/python``
    3. Default ``"core"`` — dev mode, everything is one env

    ``$VKIT_ENV`` is trusted even if the name is unfamiliar; wrong names
    are caught at dispatch time by :func:`env_python`'s existence check,
    not here. This keeps the function testable with synthetic env names.
    """
    from_env = os.environ.get("VKIT_ENV", "").strip()
    if from_env:
        return from_env

    exe = Path(sys.executable).resolve()
    for part in exe.parts:
        if part in KNOWN_ENVS and "/envs/" in exe.as_posix():
            return part
    return "core"


def env_python(env_name: str) -> Path:
    """Return the path to ``python`` inside env ``env_name``.

    Used by the executor when spawning a subprocess into another env.
    The env name does not have to be in :data:`KNOWN_ENVS` — tests
    commonly spin up synthetic envs (``"sandbox"`` etc.) so we only
    validate the resulting path exists at spawn time, not the name.
    """
    return ENVS_DIR / env_name / "bin" / "python"


def reset_caches() -> None:
    """Drop cached JSON reads. For tests."""
    _load_op_env_map.cache_clear()
    _derive_from_registry.cache_clear()
