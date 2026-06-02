"""Real implementation of `vkit run`."""

from __future__ import annotations

import json
from pathlib import Path

import typer
from rich import print as rprint
from rich.table import Table

from voxkitchen.cli.hints import warn_if_unmanaged_runtime
from voxkitchen.pipeline.checkpoint import stage_dir_name
from voxkitchen.pipeline.loader import PipelineLoadError, load_pipeline_spec
from voxkitchen.pipeline.runner import StageFailedError, run_pipeline
from voxkitchen.pipeline.spec import PipelineSpec


def _print_dry_run(
    spec: PipelineSpec, *, pipeline_path: Path | None = None, preflight: bool = True
) -> bool:
    """Validate the pipeline and print a stage summary. Returns True if valid."""
    from voxkitchen.cli.hints import format_recommended_image_hint, recommend_docker_tag
    from voxkitchen.cli.validate import validate_stage_args
    from voxkitchen.runtime.schemas import load_op_schemas

    rprint(f"\n[bold]{spec.name}[/bold]  (dry-run)")
    rprint(f"  work_dir: {spec.work_dir}")
    rprint(f"  gc_mode:  {spec.gc_mode}")
    rprint(f"  gpus: {spec.num_gpus} (requested)  cpu_workers: {spec.num_cpu_workers or 'auto'}")
    rprint(f"  ingest:   source={spec.ingest.source}", end="")
    if spec.ingest.recipe:
        rprint(f"  recipe={spec.ingest.recipe}", end="")
    rprint()

    for warning in _dry_run_warnings(spec):
        rprint(f"  [yellow]warning:[/yellow] {warning}")

    schemas = load_op_schemas()
    errors: list[str] = []
    schema_checked: list[str] = []
    validation_results = []
    t = Table(title="Stages")
    t.add_column("#", justify="right")
    t.add_column("Name")
    t.add_column("Operator")
    t.add_column("Device")
    t.add_column("Args")
    has_gpu_pref = False
    for i, stage in enumerate(spec.stages):
        result = validate_stage_args(stage.op, stage.args, schemas)
        validation_results.append(result)
        if result.error is not None:
            errors.append(f"stage {i} ({stage.name}): {result.error}")
        if result.validator == "schema" and result.device != "?":
            schema_checked.append(stage.op)
        if result.device == "gpu":
            has_gpu_pref = True
        args_str = ", ".join(f"{k}={v}" for k, v in stage.args.items()) if stage.args else "-"
        t.add_row(str(i), stage.name, stage.op, result.device, args_str)
    rprint(t)

    if has_gpu_pref:
        rprint(
            "[dim]Device shows the preferred executor. gpu stages run on GPU "
            "when one is available and fall back to CPU otherwise.[/dim]"
        )

    if schema_checked:
        checked = ", ".join(sorted(set(schema_checked)))
        rprint(f"[dim]Checked via op_schemas.json: {checked}[/dim]")

    tag = recommend_docker_tag([r.required_extras for r in validation_results])
    pipeline_hint = str(pipeline_path) if pipeline_path is not None else "<yaml>"
    rprint(f"[dim]{format_recommended_image_hint(tag, pipeline_hint)}[/dim]")

    if preflight:
        from typing import cast

        from voxkitchen.pipeline.preflight import make_contract_lookup, preflight_spec

        pf = preflight_spec(
            spec, contract_lookup=make_contract_lookup(cast("dict[str, object] | None", schemas))
        )
        for w in pf.warnings:
            rprint(f"  [yellow]warning:[/yellow] {w}")
        for e in pf.errors:
            errors.append(e)

    if errors:
        for err in errors:
            rprint(f"  [red]error:[/red] {err}")
        rprint(
            "[dim]Tip: run `vkit validate <yaml>` for schema-only checks, "
            "or `vkit docker doctor` to inspect the Docker image.[/dim]"
        )
        rprint(f"\n[red]validation failed ({len(errors)} error(s))[/red]")
        return False
    rprint("[green]validation passed[/green]")
    return True


def _dry_run_warnings(spec: PipelineSpec) -> list[str]:
    """Non-blocking checks that catch common first-run mistakes."""
    args = spec.ingest.args
    warnings: list[str] = []

    if spec.ingest.source == "dir":
        root = args.get("root")
        if isinstance(root, str):
            root_path = Path(root)
            if not root_path.is_dir():
                warnings.append(
                    f"ingest root {root!r} was not found from this shell; "
                    "create it, edit pipeline.yaml, or run with Docker mounts."
                )
            elif not _contains_audio_files(root_path, recursive=bool(args.get("recursive", True))):
                warnings.append(
                    f"ingest root {root!r} contains no supported audio files; "
                    "put audio under data/ or update ingest.args.root."
                )
    elif spec.ingest.source == "manifest":
        path = args.get("path")
        if isinstance(path, str) and not Path(path).is_file():
            warnings.append(
                f"manifest {path!r} was not found from this shell; "
                "the real run will fail unless that path exists in the runtime env."
            )
    elif spec.ingest.source == "recipe":
        root = args.get("root")
        if isinstance(root, str) and not Path(root).exists():
            warnings.append(
                f"recipe root {root!r} was not found from this shell; "
                "download the dataset first or adjust ingest.args.root."
            )

    return warnings


def _contains_audio_files(root: Path, *, recursive: bool) -> bool:
    """Return quickly after finding the first supported audio file."""
    from voxkitchen.utils.audio import AUDIO_EXTENSIONS

    paths = root.rglob("*") if recursive else root.iterdir()
    return any(p.is_file() and p.suffix.lower() in AUDIO_EXTENSIONS for p in paths)


def run_command(
    pipeline: Path,
    num_gpus: int | None = None,
    num_workers: int | None = None,
    work_dir: str | None = None,
    resume_from: str | None = None,
    stop_at: str | None = None,
    dry_run: bool = False,
    keep_intermediates: bool = False,
    no_preflight: bool = False,
) -> None:
    """Execute a pipeline."""
    import logging

    warn_if_unmanaged_runtime(command="run", recommended="vkit docker run <yaml>")

    logging.basicConfig(
        format="%(asctime)s  %(message)s",
        datefmt="%H:%M:%S",
        level=logging.INFO,
    )

    try:
        spec = load_pipeline_spec(pipeline)
    except PipelineLoadError as exc:
        rprint(f"[red]error loading pipeline:[/red] {exc}")
        raise typer.Exit(code=1) from exc

    # CLI flag overrides
    if num_gpus is not None:
        spec = spec.model_copy(update={"num_gpus": num_gpus})
    if num_workers is not None:
        spec = spec.model_copy(update={"num_cpu_workers": num_workers})
    if work_dir is not None:
        spec = spec.model_copy(update={"work_dir": work_dir})

    if dry_run:
        valid = _print_dry_run(spec, pipeline_path=pipeline, preflight=not no_preflight)
        raise typer.Exit(code=0 if valid else 1)

    # Fail fast on broken stage chains BEFORE spinning up models / the executor.
    if not no_preflight:
        from typing import cast

        from voxkitchen.pipeline.preflight import make_contract_lookup, preflight_spec
        from voxkitchen.runtime.schemas import load_op_schemas

        schemas = cast("dict[str, object] | None", load_op_schemas())
        pf = preflight_spec(spec, contract_lookup=make_contract_lookup(schemas))
        for warning in pf.warnings:
            rprint(f"[yellow]warning:[/yellow] {warning}")
        if pf.errors:
            for err in pf.errors:
                rprint(f"[red]error:[/red] {err}")
            rprint(
                "[dim]Pre-flight found broken stage chains; fix them or pass "
                "--no-preflight to run anyway.[/dim]"
            )
            raise typer.Exit(code=1)

    try:
        run_pipeline(
            spec,
            stop_at=stop_at,
            resume_from=resume_from,
            keep_intermediates=keep_intermediates,
        )
    except StageFailedError as exc:
        # Exit code 1 = runtime failure (file missing, model error, bad data).
        # Code 2 across the rest of the CLI is reserved for "invocation is
        # malformed" (unknown flag, missing docker, unknown category).
        rprint(f"[red]stage failed:[/red] {exc}")
        raise typer.Exit(code=1) from exc

    _print_completion(spec, stop_at=stop_at)


def _read_stage_stats(stage_dir: Path) -> dict[str, object] | None:
    """Read cuts_in/cuts_out from a stage's _stats.json. Returns None on missing/unreadable.

    Values may be int (counts) or float (timings, e.g. wall_time_seconds,
    throughput_cuts_per_sec).  Callers should use .get() with a typed default.
    """
    stats_path = stage_dir / "_stats.json"
    if not stats_path.exists():
        return None
    try:
        from typing import cast

        return cast("dict[str, object]", json.loads(stats_path.read_text()))
    except (OSError, json.JSONDecodeError):
        return None


def _print_completion(spec: PipelineSpec, *, stop_at: str | None) -> None:
    """Print the useful next paths after a successful run."""
    work_dir = Path(spec.work_dir)
    stage_names = [s.name for s in spec.stages]
    final_stage = stop_at or stage_names[-1]
    final_idx = stage_names.index(final_stage)
    final_stage_dir = work_dir / stage_dir_name(final_idx, final_stage)
    final_manifest = final_stage_dir / "cuts.jsonl.gz"
    report = work_dir / "report.html"

    # Detect the 100%-drop case: ingest produced cuts but the final stage
    # output zero. Almost always means an operator silently swallowed every
    # cut (try/except in process()) — surface it before the green banner
    # so users don't get a misleading "complete" with no data.
    first_stage_dir = work_dir / stage_dir_name(0, stage_names[0])
    first_stats = _read_stage_stats(first_stage_dir)
    final_stats = _read_stage_stats(final_stage_dir)
    if first_stats is not None and final_stats is not None:
        cuts_in_raw = first_stats.get("cuts_in", 0)
        cuts_out_raw = final_stats.get("cuts_out", 0)
        if (
            isinstance(cuts_in_raw, int)
            and isinstance(cuts_out_raw, int)
            and cuts_in_raw > 0
            and cuts_out_raw == 0
        ):
            rprint(
                f"[yellow]⚠ warning:[/yellow] all {cuts_in_raw} input cuts were "
                f"dropped — likely an operator silently failed. Check "
                f"`vkit inspect errors {work_dir}` and per-stage `_stats.json`."
            )

    if stop_at:
        rprint(f"[green]stopped after stage[/green] {stop_at}")
    else:
        rprint("[green]pipeline complete[/green]")

    rprint(f"  work_dir: {work_dir}")
    if final_manifest.exists():
        rprint(f"  final cuts: {final_manifest}")
    if report.exists():
        rprint(f"  report: {report}")
    hf_output = _pack_huggingface_output_dir(spec, max_stage_idx=final_idx)
    if hf_output is not None:
        rprint(f"  HuggingFace dataset: {hf_output}")

    rprint("\n[bold]Next steps[/bold]")
    rprint(f"  vkit inspect run {work_dir}")
    if final_manifest.exists():
        rprint(f"  vkit inspect cuts {final_manifest}")
    if hf_output is not None:
        rprint(
            "  load HF audio arrays with `torchcodec` installed, or use "
            "`datasets.Audio(decode=False)` for metadata/bytes-only reads"
        )


def _pack_huggingface_output_dir(spec: PipelineSpec, *, max_stage_idx: int) -> Path | None:
    """Return the output directory for the last completed pack_huggingface stage."""
    output_dir: Path | None = None
    for idx, stage in enumerate(spec.stages[: max_stage_idx + 1]):
        if stage.op != "pack_huggingface":
            continue
        configured = stage.args.get("output_dir")
        if isinstance(configured, str) and configured.strip():
            output_dir = Path(configured)
        else:
            output_dir = Path(spec.work_dir) / stage_dir_name(idx, stage.name) / "hf_output"
    return output_dir
