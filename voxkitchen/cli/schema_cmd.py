"""``vkit schema export`` — emit a JSON Schema for ``pipeline.yaml`` files.

The exported document tells YAML language servers (the VS Code "YAML" extension,
Neovim's ``yamlls``, JetBrains' built-in JSON Schema support, etc.) about the
structure of a VoxKitchen pipeline so users get autocompletion, hover docs, and
inline error squiggles while editing.

The schema is derived from two sources:

1. ``PipelineSpec.model_json_schema()`` gives the top-level shape — `version`,
   `name`, `work_dir`, `ingest`, `stages`, plus all the ingest spec details.
2. The operator registry — every registered operator's name is added to a
   ``$defs.StageSpec.properties.op.enum`` so editors offer autocomplete on
   the registered operator names.

Per-operator ``args`` validation (i.e. validating the per-stage args against
each operator's specific config schema) is intentionally **not** wired up
here yet: the minimal schema already delivers most of the DX value
(structural completion + op-name validation) and keeps the generated file
small. A future iteration can layer per-op ``if/then`` discriminator rules
on top without changing how callers use this command.

The output is meant to be committed under ``docs/schemas/pipeline.schema.json``
so it can be served via raw.githubusercontent.com and referenced from
``# yaml-language-server: $schema=...`` comments at the top of pipeline
YAMLs. ``vkit init`` writes that comment automatically.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import typer
from rich import print as rprint

schema_app = typer.Typer(
    name="schema",
    help="Generate JSON Schemas for VoxKitchen YAML.",
    no_args_is_help=True,
)


SCHEMA_URI = "https://json-schema.org/draft/2020-12/schema"
SCHEMA_TITLE = "VoxKitchen pipeline"


def build_pipeline_schema() -> dict[str, Any]:
    """Return the merged JSON Schema for ``pipeline.yaml`` files.

    The Pydantic base schema gets two surgical enhancements:

    - ``stages[].op`` is constrained to the set of registered operator names.
      Editors offer those as autocomplete candidates and flag typos.
    - The document gains a top-level ``$schema`` URI and a ``title`` so YAML
      language servers display a meaningful tooltip.

    Generating the schema in the current Python process means the result
    reflects whatever operators are importable here. In source-tree dev that's
    most of the operators (the core cluster); inside the published Docker
    images it's the full set for the image's env. To produce a complete
    schema for the public site, run this command inside the ``:latest``
    image where every env's operators are registered.
    """
    # Local import keeps `vkit schema --help` cheap when only listing commands.
    from voxkitchen.operators.registry import list_operators
    from voxkitchen.pipeline.spec import PipelineSpec

    schema = PipelineSpec.model_json_schema()
    schema["$schema"] = SCHEMA_URI
    schema["title"] = SCHEMA_TITLE
    schema["description"] = (
        "Schema for VoxKitchen pipeline YAML files. See "
        "https://github.com/XqFeng-Josie/VoxKitchen for the project."
    )

    op_names = sorted(list_operators())
    stage_spec = schema.get("$defs", {}).get("StageSpec")
    if stage_spec is not None:
        op_prop = stage_spec.get("properties", {}).get("op")
        if isinstance(op_prop, dict):
            op_prop["enum"] = op_names
            op_prop["description"] = (
                f"Name of one of the {len(op_names)} registered operators. "
                "Run `vkit operators` for a categorized list, or "
                "`vkit operators search <kw>` to find one by keyword."
            )

    return schema


@schema_app.command("export")
def export_cmd(
    out: Path = typer.Option(
        Path("pipeline.schema.json"),
        "--out",
        "-o",
        help="Where to write the generated schema JSON.",
    ),
    indent: int = typer.Option(2, "--indent", help="JSON indentation (0 = compact)."),
) -> None:
    """Generate the pipeline JSON Schema and write it to a file."""
    schema = build_pipeline_schema()
    out.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(schema, indent=indent if indent > 0 else None, ensure_ascii=False) + "\n"
    out.write_text(text, encoding="utf-8")

    op_count = len(
        schema.get("$defs", {})
        .get("StageSpec", {})
        .get("properties", {})
        .get("op", {})
        .get("enum", [])
    )
    rprint(f"[green]wrote[/green] {out}  ([dim]{op_count} operators, {len(text)} bytes[/dim])")
