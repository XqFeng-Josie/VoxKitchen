"""Dispatch a stage to a subprocess in another env.

This is the bridge from the parent (core) env's pipeline runner to the
:mod:`voxkitchen.runtime.stage_runner` living in another env. Everything
operator-specific crosses the boundary via disk files; this module only
knows how to launch the subprocess and surface failures.

The dispatcher is intentionally function-shaped rather than class-shaped —
there is no state to carry, and keeping it flat avoids the temptation to
smuggle pickle-unsafe state across the env boundary.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING, Any

from voxkitchen.runtime.env_resolver import env_python

if TYPE_CHECKING:
    from voxkitchen.pipeline.context import RunContext

logger = logging.getLogger(__name__)


class StageDispatchFailure(RuntimeError):
    """Raised when the subprocess stage runner exits with a non-zero code.

    The subprocess's stderr is inherited by the parent, so the actual
    traceback is already visible in the user's terminal; this exception
    just gives the runner a typed error to propagate upward.
    """

    def __init__(self, op_name: str, target_env: str, returncode: int) -> None:
        self.op_name = op_name
        self.target_env = target_env
        self.returncode = returncode
        super().__init__(
            f"stage {op_name!r} in env {target_env!r} exited with "
            f"code {returncode} (stderr was streamed above)"
        )


def dispatch_stage_to_env(
    *,
    target_env: str,
    op_name: str,
    config_args: dict[str, Any],
    input_path: Path,
    output_path: Path,
    ctx: RunContext,
) -> None:
    """Spawn ``voxkitchen.runtime.stage_runner`` in ``target_env``.

    Blocks until the subprocess exits. Output and error artifacts are
    written by the subprocess itself:

    - ``output_path`` — the transformed cuts manifest
    - ``output_path.parent / _errors.jsonl`` — per-cut errors
    - ``output_path.parent / _stats.json`` — timing and throughput

    stderr is inherited (not captured), so live progress bars from ASR/TTS
    operators stream to the user's terminal as they would in-process.

    Raises :class:`StageDispatchFailure` on non-zero exit.
    """
    python = env_python(target_env)
    if not python.exists():
        # Fast-fail with a useful message: the env was supposed to be built
        # at image time and isn't — almost certainly a Dockerfile / layout bug.
        raise StageDispatchFailure(
            op_name=op_name,
            target_env=target_env,
            returncode=-1,
        ) from FileNotFoundError(f"python interpreter not found at {python}")

    ctx_dict = asdict(ctx)
    # RunContext.work_dir is a Path; asdict leaves it as-is. JSON serialize
    # with default=str so Path (and any future non-primitive fields) turn
    # into strings. The subprocess's _deserialize_ctx() turns them back.
    ctx_json = json.dumps(ctx_dict, default=str)

    cmd = [
        str(python),
        "-m",
        "voxkitchen.runtime.stage_runner",
        "--op",
        op_name,
        "--config-json",
        json.dumps(config_args, default=str),
        "--input",
        str(input_path),
        "--output",
        str(output_path),
        "--ctx-json",
        ctx_json,
    ]

    # VKIT_ENV tells the subprocess (and its env_resolver.current_env())
    # which env it is running in. Set PYTHONUNBUFFERED so live logs and
    # progress bars don't buffer invisibly when stderr is a pipe.
    env = os.environ.copy()
    env["VKIT_ENV"] = target_env
    env.setdefault("PYTHONUNBUFFERED", "1")

    logger.info("dispatch stage %s → env %s", op_name, target_env)
    result = subprocess.run(
        cmd,
        env=env,
        stdin=subprocess.DEVNULL,
        stdout=sys.stdout,
        stderr=sys.stderr,
        check=False,
    )
    if result.returncode != 0:
        raise StageDispatchFailure(
            op_name=op_name,
            target_env=target_env,
            returncode=result.returncode,
        )
