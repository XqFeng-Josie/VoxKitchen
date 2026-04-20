"""Build-time: merge per-env schema dumps into the unified runtime maps.

Reads N files produced by :mod:`voxkitchen.runtime.dump_schemas`, one per
env, and writes two files:

- ``op_schemas.json`` — ``{op_name: operator-info-dict}`` consumed by the
  parent env's ``vkit validate`` and by the executor to look up device /
  required_extras without importing the operator.
- ``op_env_map.json`` — ``{op_name: canonical_env_name}`` consumed by
  :mod:`voxkitchen.runtime.env_resolver` to decide dispatch.

Why "canonical" env: in the multi-env Docker layout, derived envs
inherit their base env's extras (asr-env ``FROM`` core-env, tts-env
``FROM`` asr-env, ...). That means every operator registered in core
is ALSO registered in asr, tts, and fish-speech — their imports succeed
there too. Picking "whichever env dump mentioned it first" would produce
an inconsistent map. Instead we compute the canonical env for each op
from its ``required_extras`` via
:data:`voxkitchen.runtime.env_resolver.EXTRA_TO_ENV`, which is the
single source of truth.

Usage from the Dockerfile::

    /opt/voxkitchen/envs/core/bin/python -m voxkitchen.runtime.merge_schemas \\
        /opt/voxkitchen/schemas_core.json \\
        /opt/voxkitchen/schemas_asr.json \\
        /opt/voxkitchen/schemas_tts.json \\
        /opt/voxkitchen/schemas_fish-speech.json \\
        --schemas-out /opt/voxkitchen/op_schemas.json \\
        --env-map-out /opt/voxkitchen/op_env_map.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class SchemaMergeError(RuntimeError):
    """Something inconsistent across env dumps."""


def _canonical_env(op_name: str, required_extras: list[str]) -> str:
    """Compute the env an operator belongs to from its extras.

    - No extras → "core" (the base env; any derived env also has it).
    - All extras mapping to the same env → that env.
    - Extras spanning multiple envs → SchemaMergeError (architecture smell).
    - Unknown extras → SchemaMergeError (missing from EXTRA_TO_ENV).
    """
    from voxkitchen.runtime.env_resolver import EXTRA_TO_ENV

    if not required_extras:
        return "core"

    unknown = [e for e in required_extras if e not in EXTRA_TO_ENV]
    if unknown:
        raise SchemaMergeError(
            f"operator {op_name!r} requires unknown extras {unknown!r}; "
            "add them to EXTRA_TO_ENV in voxkitchen/runtime/env_resolver.py"
        )
    envs = {EXTRA_TO_ENV[e] for e in required_extras}
    if len(envs) > 1:
        raise SchemaMergeError(
            f"operator {op_name!r} spans envs {sorted(envs)} via extras "
            f"{required_extras!r}; split its extras so they all live in one env"
        )
    return envs.pop()


def merge(
    dump_files: list[Path],
    schemas_out: Path,
    env_map_out: Path,
) -> tuple[int, int]:
    """Merge ``dump_files`` into the two output files.

    Returns ``(n_operators, n_envs)``.

    Same operator appearing in multiple dumps is EXPECTED (env inheritance).
    We keep the first dump's schema; a later dump with a different
    ``required_extras`` field for the same operator signals a genuine bug
    (the code in different envs disagrees) and raises.
    """
    schemas: dict[str, dict[str, Any]] = {}
    envs_seen: set[str] = set()

    for f in dump_files:
        data = json.loads(f.read_text(encoding="utf-8"))
        env_name = data["env"]
        envs_seen.add(env_name)
        for op_name, info in data["operators"].items():
            if op_name in schemas:
                prior = schemas[op_name]["required_extras"]
                curr = info["required_extras"]
                if sorted(prior) != sorted(curr):
                    raise SchemaMergeError(
                        f"operator {op_name!r} reports different required_extras in "
                        f"different envs — {prior!r} vs {curr!r}. This means the "
                        "code diverged across envs; fix before merging."
                    )
                # Identical schema registered in multiple envs — normal,
                # just keep the first.
                continue
            schemas[op_name] = info

    # Compute canonical env per op from its extras.
    env_map: dict[str, str] = {
        op_name: _canonical_env(op_name, info.get("required_extras", []))
        for op_name, info in schemas.items()
    }

    schemas_out.parent.mkdir(parents=True, exist_ok=True)
    schemas_out.write_text(
        json.dumps(schemas, indent=2, ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )
    env_map_out.parent.mkdir(parents=True, exist_ok=True)
    env_map_out.write_text(
        json.dumps(env_map, indent=2, ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )
    return len(schemas), len(envs_seen)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dumps", nargs="+", type=Path, help="Per-env schema dump JSON files.")
    parser.add_argument(
        "--schemas-out",
        type=Path,
        required=True,
        help="Output path for the merged op_schemas.json.",
    )
    parser.add_argument(
        "--env-map-out",
        type=Path,
        required=True,
        help="Output path for the op_env_map.json.",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
    n_ops, n_envs = merge(args.dumps, args.schemas_out, args.env_map_out)
    logger.info(
        "merged %d operators from %d env(s) → %s, %s",
        n_ops,
        n_envs,
        args.schemas_out,
        args.env_map_out,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
