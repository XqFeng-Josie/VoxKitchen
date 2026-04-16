"""vkit ingest: build a CutSet from a data source."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import typer
from rich import print as rprint

from voxkitchen.ingest import get_ingest_source
from voxkitchen.pipeline.context import RunContext
from voxkitchen.schema.io import SCHEMA_VERSION, HeaderRecord
from voxkitchen.utils.run_id import generate_run_id


def ingest_command(
    source: str,
    out: Path,
    root: str | None = None,
    path: str | None = None,
    recipe: str | None = None,
    recursive: bool = True,
    subsets: str | None = None,
) -> None:
    args: dict[str, object] = {}
    if source == "dir":
        if not root:
            rprint("[red]--root required for source=dir[/red]")
            raise typer.Exit(code=1)
        args = {"root": root, "recursive": recursive}
    elif source == "manifest":
        if not path:
            rprint("[red]--path required for source=manifest[/red]")
            raise typer.Exit(code=1)
        args = {"path": path}
    elif source == "recipe":
        if not recipe or not root:
            rprint("[red]--recipe and --root required for source=recipe[/red]")
            raise typer.Exit(code=1)
        args = {"recipe": recipe, "root": root}
        if subsets:
            args["subsets"] = [s.strip() for s in subsets.split(",")]
    else:
        rprint(f"[red]unknown source: {source}[/red]")
        raise typer.Exit(code=1)

    source_cls = get_ingest_source(source)
    config = source_cls.config_cls.model_validate(args)
    run_id = generate_run_id()

    ctx = RunContext(
        work_dir=out.parent,
        pipeline_run_id=run_id,
        stage_index=0,
        stage_name="ingest",
        num_gpus=0,
        num_cpu_workers=1,
        gc_mode="aggressive",
        device="cpu",
    )

    source_obj = source_cls(config, ctx)
    cuts = source_obj.run()

    header = HeaderRecord(
        schema_version=SCHEMA_VERSION,
        created_at=datetime.now(tz=timezone.utc),
        pipeline_run_id=run_id,
        stage_name="ingest",
    )
    cuts.to_jsonl_gz(out, header)
    rprint(f"[green]wrote {len(cuts)} cuts to {out}[/green]")
