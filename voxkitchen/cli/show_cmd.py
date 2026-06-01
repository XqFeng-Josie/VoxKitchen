"""Implementation of ``vkit show <pipeline.yaml>``.

Pretty-prints a pipeline YAML as an operator chain with each stage's
declarative field contract (``reads`` / ``writes`` / ``optional_reads`` /
``clears``) inline — the same information the static pre-flight uses, surfaced
so users can *see* what a pipeline does without scanning raw YAML.

Two data sources, both already used by ``vkit validate``:

1. **Fast path** — when the operator is importable in this env, read the
   contract straight off the class (``op_cls.reads`` etc.). Default for
   dev / local runs.
2. **Fallback** — when the operator lives in another env (multi-env Docker
   layout), read the merged ``op_schemas.json``. Same path
   ``vkit validate`` uses.

Pure presentation: no validation, no execution, no I/O beyond reading the
YAML and (optionally) the schemas file. Safe to run anywhere.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import typer
from rich import print as rprint
from rich.console import Console
from rich.panel import Panel

from voxkitchen.operators.registry import UnknownOperatorError, get_operator
from voxkitchen.pipeline.loader import PipelineLoadError, load_pipeline_spec
from voxkitchen.runtime.schemas import load_op_schemas

console = Console()


def _contract_for(
    op_name: str, schemas: dict[str, dict[str, Any]] | None
) -> tuple[dict[str, list[str]], bool]:
    """Return ``({reads, writes, optional_reads, clears}, found)``.

    ``found`` distinguishes two cases the user otherwise can't tell apart:

    - ``True``  : the operator is known (registry or schemas), the contract
                  is whatever it declares — *empty lists are intentional*
                  (filter operators declare no writes by design).
    - ``False`` : the operator is unknown in both sources — the show
                  command shows a footnote so the user knows the
                  visualisation is incomplete for this stage.
    """
    empty: dict[str, list[str]] = {"reads": [], "writes": [], "optional_reads": [], "clears": []}
    try:
        op_cls = get_operator(op_name)
    except UnknownOperatorError:
        op_cls = None

    if op_cls is not None:
        return {
            "reads": list(getattr(op_cls, "reads", [])),
            "writes": list(getattr(op_cls, "writes", [])),
            "optional_reads": list(getattr(op_cls, "optional_reads", [])),
            "clears": list(getattr(op_cls, "clears", [])),
        }, True

    if schemas is None:
        return empty, False
    info = schemas.get(op_name)
    if not isinstance(info, dict):
        return empty, False
    return {
        "reads": list(info.get("reads", [])),
        "writes": list(info.get("writes", [])),
        "optional_reads": list(info.get("optional_reads", [])),
        "clears": list(info.get("clears", [])),
    }, True


def _device_for(op_name: str, schemas: dict[str, dict[str, Any]] | None) -> str:
    """Return the operator's runtime device tag (``cpu`` / ``gpu`` / ``?``)."""
    try:
        op_cls = get_operator(op_name)
        return str(getattr(op_cls, "device", "?"))
    except UnknownOperatorError:
        pass
    if schemas is not None and op_name in schemas:
        return str(schemas[op_name].get("device", "?"))
    return "?"


def _format_args(args: dict[str, Any]) -> str:
    """One-line ``key=value`` rendering for a small args dict (≤4 keys).

    Larger args are summarised as ``"<N args>"`` so the visualisation stays
    readable. Empty args render as a dim ``""`` so the layout doesn't shift.
    """
    if not args:
        return "[dim](no args)[/dim]"
    if len(args) > 4:
        return f"[dim]<{len(args)} args>[/dim]"
    pairs = []
    for k, v in args.items():
        if isinstance(v, (dict, list)):
            pairs.append(f"{k}=<{type(v).__name__}>")
        else:
            pairs.append(f"{k}={v!r}")
    return ", ".join(pairs)


def _contract_lines(contract: dict[str, list[str]]) -> list[str]:
    """Render the four contract lists as up to four ``key: a, b, c`` rows.

    Empty lists are dropped — a stage with no declared reads still prints
    the writes row, no dangling ``reads: (none)`` filler.
    """
    rows: list[str] = []
    if contract["reads"]:
        rows.append(f"  [cyan]reads:[/cyan]          {', '.join(contract['reads'])}")
    if contract["optional_reads"]:
        rows.append(f"  [dim]optional reads:[/dim] {', '.join(contract['optional_reads'])}")
    if contract["writes"]:
        rows.append(f"  [green]writes:[/green]         {', '.join(contract['writes'])}")
    if contract["clears"]:
        rows.append(f"  [yellow]clears:[/yellow]         {', '.join(contract['clears'])}")
    return rows


def show_command(pipeline: Path) -> None:
    """Render a pipeline YAML as a visual stage list with contract info."""
    try:
        spec = load_pipeline_spec(pipeline)
    except PipelineLoadError as exc:
        rprint(f"[red]error loading pipeline:[/red] {exc}")
        raise typer.Exit(code=1) from exc

    schemas = load_op_schemas()

    header_lines: list[str] = []
    header_lines.append(f"[bold]name:[/bold]        {spec.name}")
    if spec.description:
        header_lines.append(f"[bold]description:[/bold] {spec.description}")
    header_lines.append(f"[bold]work_dir:[/bold]    {spec.work_dir}")
    header_lines.append("")
    header_lines.append(
        f"[bold]ingest[/bold] [magenta]\\[{spec.ingest.source}][/magenta]   "
        f"{_format_args(spec.ingest.args)}"
    )
    console.print(Panel("\n".join(header_lines), title="[bold]pipeline[/bold]"))

    n = len(spec.stages)
    console.print(f"\n[bold]Stages ({n})[/bold]\n")

    for i, stage in enumerate(spec.stages):
        contract, found = _contract_for(stage.op, schemas)
        device = _device_for(stage.op, schemas)
        device_marker = "[red]gpu[/red]" if device == "gpu" else f"[dim]{device}[/dim]"

        # One-line stage header: "01  resample  (op: resample)  cpu"
        console.print(
            f"  [bold cyan]{i + 1:02d}[/bold cyan]  "
            f"[bold]{stage.name}[/bold]  "
            f"[dim](op: {stage.op})[/dim]  {device_marker}"
        )
        # Args inline as a dim sub-row when present.
        if stage.args:
            console.print(f"      [dim]args:[/dim] {_format_args(stage.args)}")
        # Contract lines (each already coloured + indented).
        for row in _contract_lines(contract):
            console.print(f"    {row}")

        if not found:
            # Operator unknown in registry AND op_schemas.json — flag it so
            # the user knows this stage's contract info is missing rather
            # than empty.
            console.print(
                f"    [dim]contract: not available in this env "
                f"(operator {stage.op!r} not importable here)[/dim]"
            )
        elif not any(contract.values()):
            # Operator IS known but declares no static contract (filters /
            # config-driven inputs). Soft note so the visualisation reads
            # as intentional, not broken.
            console.print(
                "    [dim](no static contract — typically a filter / config-driven op)[/dim]"
            )

        # Empty line between stages keeps the chain readable but doesn't
        # trail at the end.
        if i < n - 1:
            console.print()

    # Footer with the natural next commands. Keeps the discoverability
    # loop tight — show → validate → run — without nagging.
    console.print(
        "\n[dim]next: `vkit validate "
        f"{pipeline}` to confirm the chain is statically sound, "
        f"then `vkit run {pipeline}` to execute.[/dim]"
    )
