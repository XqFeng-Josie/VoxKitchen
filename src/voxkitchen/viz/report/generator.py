"""Generate a self-contained HTML report from pipeline results."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from voxkitchen.schema.cutset import CutSet
from voxkitchen.viz.stats import compute_cutset_stats


def generate_report(
    work_dir: Path,
    output_path: Path | None = None,
    pipeline_name: str = "",
    run_id: str = "",
) -> Path:
    """Generate report.html from a pipeline work_dir."""
    import jinja2

    stage_dirs = sorted(d for d in work_dir.iterdir() if d.is_dir() and d.name[:2].isdigit())
    stages_info = []
    final_cuts = CutSet([])

    for sd in stage_dirs:
        success = (sd / "_SUCCESS").exists()
        manifest = sd / "cuts.jsonl.gz"
        if manifest.exists():
            cs = CutSet.from_jsonl_gz(manifest)
            stages_info.append(
                {"name": sd.name, "status": "OK" if success else "INCOMPLETE", "count": len(cs)}
            )
            if success:
                final_cuts = cs
        else:
            stages_info.append({"name": sd.name, "status": "MISSING", "count": 0})

    stats = compute_cutset_stats(final_cuts)
    duration_chart = _make_duration_chart(final_cuts)

    template_dir = Path(__file__).parent / "templates"
    env = jinja2.Environment(loader=jinja2.FileSystemLoader(str(template_dir)), autoescape=True)
    template = env.get_template("report.html.j2")

    html = template.render(
        pipeline_name=pipeline_name or work_dir.name,
        run_id=run_id,
        generated_at=datetime.now(tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        stats=stats,
        duration_chart=duration_chart,
        stages=stages_info,
    )

    out = output_path or (work_dir / "report.html")
    out.write_text(html, encoding="utf-8")
    return out


def _make_duration_chart(cuts: CutSet) -> str:
    """Return a Plotly histogram as an HTML div string, or a fallback message."""
    try:
        import plotly.graph_objects as go
        import plotly.io as pio

        durations = [c.duration for c in cuts]
        if not durations:
            return ""
        fig = go.Figure(go.Histogram(x=durations, nbinsx=30))
        fig.update_layout(
            xaxis_title="Duration (s)",
            yaxis_title="Count",
            height=300,
            margin=dict(l=40, r=20, t=20, b=40),
        )
        return str(pio.to_html(fig, full_html=False, include_plotlyjs=True))
    except ImportError:
        return "<p><i>Plotly not installed — chart unavailable.</i></p>"
