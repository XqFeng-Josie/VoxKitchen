"""Rich terminal rendering for vkit inspect subcommands."""

from __future__ import annotations

import json
from pathlib import Path

from rich.console import Console
from rich.table import Table

from voxkitchen.schema.cut import Cut
from voxkitchen.schema.cutset import CutSet
from voxkitchen.viz.stats import compute_cutset_stats

console = Console()


def render_cuts_stats(manifest_path: Path) -> None:
    """Print CutSet statistics to the terminal."""
    cuts = CutSet.from_jsonl_gz(manifest_path)
    stats = compute_cutset_stats(cuts)

    console.print(f"\n[bold]CutSet: {manifest_path.name}[/bold]")
    console.print(f"  Cuts: {stats['count']}")
    console.print(f"  Total duration: {stats['total_duration_s']:.1f}s")

    if stats["duration_stats"]:
        t = Table(title="Duration (seconds)")
        for col in ["min", "mean", "p50", "p95", "max"]:
            t.add_column(col, justify="right")
        ds = stats["duration_stats"]
        t.add_row(*[f"{ds[c]:.3f}" for c in ["min", "mean", "p50", "p95", "max"]])
        console.print(t)

    if stats["languages"]:
        console.print(f"  Languages: {stats['languages']}")
    if stats["speaker_count"]:
        console.print(f"  Speakers: {stats['speaker_count']}")
    if stats["metrics_summary"]:
        mt = Table(title="Metrics")
        mt.add_column("metric")
        for col in ["min", "mean", "p50", "p95", "max"]:
            mt.add_column(col, justify="right")
        for k, v in stats["metrics_summary"].items():
            mt.add_row(k, *[f"{v.get(c, 0):.3f}" for c in ["min", "mean", "p50", "p95", "max"]])
        console.print(mt)


def render_run_summary(work_dir: Path) -> None:
    """Print pipeline run summary: stages, status, cut counts."""
    if not work_dir.exists():
        console.print(f"[red]work_dir does not exist: {work_dir}[/red]")
        return
    stage_dirs = sorted(d for d in work_dir.iterdir() if d.is_dir() and d.name[:2].isdigit())
    console.print(f"\n[bold]Pipeline run: {work_dir.name}[/bold]")
    for sd in stage_dirs:
        success = (sd / "_SUCCESS").exists()
        manifest = sd / "cuts.jsonl.gz"
        if manifest.exists() and manifest.stat().st_size > 0:
            try:
                count: int | str = len(CutSet.from_jsonl_gz(manifest))
            except Exception:
                count = "?"
        else:
            count = "?"
        status = "[green]OK[/green]" if success else "[red]INCOMPLETE[/red]"
        console.print(f"  {sd.name}: {status} ({count} cuts)")


def render_trace(cut_id: str, work_dir: Path) -> None:
    """Walk the provenance chain for *cut_id* across all stage manifests."""
    if not work_dir.exists():
        console.print(f"[red]work_dir does not exist: {work_dir}[/red]")
        return

    # Build lookup: cut_id → Cut across all stages (latest stage wins)
    stage_dirs = sorted(d for d in work_dir.iterdir() if d.is_dir() and d.name[:2].isdigit())
    all_cuts: dict[str, tuple[str, Cut]] = {}
    for sd in stage_dirs:
        manifest = sd / "cuts.jsonl.gz"
        if manifest.exists():
            try:
                for c in CutSet.from_jsonl_gz(manifest):
                    all_cuts[c.id] = (sd.name, c)
            except Exception:
                continue

    if cut_id not in all_cuts:
        console.print(f"[red]cut {cut_id!r} not found in any stage[/red]")
        return

    # Walk the chain
    console.print(f"\n[bold]Provenance trace for {cut_id}[/bold]\n")
    current_id: str | None = cut_id
    depth = 0
    while current_id and current_id in all_cuts:
        stage_name, cut = all_cuts[current_id]
        indent = "  " * depth
        if depth > 0:
            console.print(f"{indent}↑")
        console.print(f"{indent}[bold]{cut.id}[/bold]  (stage: {stage_name})")
        console.print(f"{indent}  duration: {cut.duration:.3f}s  start: {cut.start:.3f}s")
        if cut.provenance:
            p = cut.provenance
            console.print(f"{indent}  generated_by: {p.generated_by}")
            console.print(f"{indent}  created_at:   {p.created_at}")
            current_id = p.source_cut_id
        else:
            current_id = None
        depth += 1
        if depth > 50:
            console.print(f"{indent}  [yellow]... (chain truncated at 50)[/yellow]")
            break

    if current_id and current_id not in all_cuts:
        indent = "  " * depth
        console.print(f"{indent}↑")
        console.print(f"{indent}[dim]{current_id}[/dim]  (source — not in pipeline stages)")


def render_errors(work_dir: Path) -> None:
    """Print _errors.jsonl from each stage (if any)."""
    if not work_dir.exists():
        console.print(f"[red]work_dir does not exist: {work_dir}[/red]")
        return
    stage_dirs = sorted(d for d in work_dir.iterdir() if d.is_dir() and d.name[:2].isdigit())
    found = False
    for sd in stage_dirs:
        err_file = sd / "_errors.jsonl"
        if err_file.exists():
            found = True
            console.print(f"\n[bold red]{sd.name} errors:[/bold red]")
            for line in err_file.read_text().strip().split("\n"):
                if line.strip():
                    err = json.loads(line)
                    console.print(f"  cut={err.get('cut_id', '?')} error={err.get('error', '?')}")
    if not found:
        console.print("[green]No errors found.[/green]")
