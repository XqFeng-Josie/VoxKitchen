"""vkit ingest: build a CutSet from a data source."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import typer
from rich import print as rprint

from voxkitchen.cli.hints import warn_if_unmanaged_runtime
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
            rprint("[red]error:[/red] --root required for source=dir")
            raise typer.Exit(code=1)
        args = {"root": root, "recursive": recursive}
    elif source == "manifest":
        if not path:
            rprint("[red]error:[/red] --path required for source=manifest")
            raise typer.Exit(code=1)
        args = {"path": path}
    elif source == "recipe":
        if not recipe or not root:
            rprint("[red]error:[/red] --recipe and --root required for source=recipe")
            raise typer.Exit(code=1)
        # Recipe ingest pulls dependencies (e.g. `datasets` for FLEURS) that
        # ship in Docker images, not in the lightweight PyPI launcher. Warn
        # host users. `dir` and `manifest` need no extra deps and stay quiet.
        warn_if_unmanaged_runtime(
            command="ingest --source recipe",
            recommended="vkit docker download <recipe>",
        )
        args = {"recipe": recipe, "root": root}
        if subsets:
            args["subsets"] = [s.strip() for s in subsets.split(",")]
    else:
        rprint(f"[red]error:[/red] unknown source: {source}")
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
    try:
        cuts = source_obj.run()
    except KeyError as exc:
        # `get_recipe()` raises KeyError("recipe 'X' not found. Available: [...]")
        # for unknown recipe names. Strip the surrounding quotes and re-render
        # so the user sees one error line + the available list, not a full
        # Python traceback panel (peer commands like `vkit validate` already
        # render lookup failures this way).
        msg = exc.args[0] if exc.args else str(exc)
        rprint(f"[red]error:[/red] {msg}")
        raise typer.Exit(code=1) from exc
    except FileNotFoundError as exc:
        # Covers all three sources: manifest (missing --path file), dir
        # (missing root), and recipe (missing subset dir / archive).
        # All sources raise self-descriptive FileNotFoundError messages,
        # so we just surface them with the standard error prefix.
        rprint(f"[red]error:[/red] {exc}")
        raise typer.Exit(code=1) from exc

    header = HeaderRecord(
        schema_version=SCHEMA_VERSION,
        created_at=datetime.now(tz=timezone.utc),
        pipeline_run_id=run_id,
        stage_name="ingest",
    )
    cuts.to_jsonl_gz(out, header)
    rprint(f"[green]wrote {len(cuts)} cuts to {out}[/green]")
