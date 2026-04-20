"""Build-time: dump this env's operator schemas to JSON.

Each Docker env runs this once during image build. The per-env dumps are
then combined by :mod:`voxkitchen.runtime.merge_schemas` into the unified
``op_schemas.json`` and ``op_env_map.json`` that the parent (core) env
consults at runtime.

Output JSON structure::

    {
      "env": "asr",
      "operators": {
        "paraformer_asr": {
          "config_schema": { ... pydantic JSON schema ... },
          "required_extras": ["funasr"],
          "device": "gpu",
          "module": "voxkitchen.operators.annotate.paraformer_asr",
          "doc": "Transcribe Chinese audio using FunASR's Paraformer-large model."
        },
        ...
      }
    }

Usage from the Dockerfile::

    /opt/voxkitchen/envs/asr/bin/python -m voxkitchen.runtime.dump_schemas \\
        --env asr --out /tmp/schemas_asr.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def dump_current_env(env_name: str, out_path: Path) -> int:
    """Walk the in-process operator registry and write its schemas to ``out_path``.

    Returns the number of operators dumped.
    """
    # Importing the package populates the registry with every operator
    # whose optional deps are installed in this env. Unavailable operators
    # fail their try/except import and simply don't register.
    import voxkitchen.operators  # noqa: F401  (side effect: registration)
    from voxkitchen.operators.registry import get_operator, list_operators

    operators: dict[str, dict[str, Any]] = {}
    for name in sorted(list_operators()):
        op_cls = get_operator(name)
        try:
            schema = op_cls.config_cls.model_json_schema()
        except Exception as exc:
            logger.warning("could not derive JSON schema for %s: %s", name, exc)
            schema = {"error": repr(exc)}
        doc = (op_cls.__doc__ or "").strip().split("\n", 1)[0]
        operators[name] = {
            "config_schema": schema,
            "required_extras": list(op_cls.required_extras),
            "device": str(op_cls.device),
            "module": op_cls.__module__,
            "doc": doc,
        }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps({"env": env_name, "operators": operators}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return len(operators)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--env",
        required=True,
        help="Name of the env being dumped (core|asr|tts). Recorded in the output.",
    )
    parser.add_argument("--out", required=True, type=Path, help="Output JSON path.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
    n = dump_current_env(args.env, args.out)
    logger.info("dumped %d operators for env=%s → %s", n, args.env, args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
