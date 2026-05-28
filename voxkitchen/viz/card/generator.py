"""Generate a standalone, shareable HTML dataset card from a CutSet."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from voxkitchen.schema.cutset import CutSet
from voxkitchen.viz import charts
from voxkitchen.viz.stats import compute_cutset_stats


def generate_dataset_card(
    cuts: CutSet,
    output_path: Path,
    *,
    title: str = "",
    description: str = "",
) -> Path:
    """Render an HTML dataset card to ``output_path`` and return it."""
    import jinja2

    stats = compute_cutset_stats(cuts)
    metric_charts = [(k, charts.metric_histogram(cuts, k)) for k in stats["metric_keys"]]
    samples = []
    for c in list(cuts)[:10]:
        sup = c.supervisions[0] if c.supervisions else None
        samples.append(
            {
                "text": ((sup.text if sup else "") or "")[:120],
                "speaker": (sup.speaker if sup else "") or "",
                "language": (sup.language if sup else "") or "",
            }
        )

    template_dir = Path(__file__).parent / "templates"
    env = jinja2.Environment(loader=jinja2.FileSystemLoader(str(template_dir)), autoescape=True)
    template = env.get_template("card.html.j2")
    html = template.render(
        title=title or "Dataset Card",
        description=description,
        generated_at=datetime.now(tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        stats=stats,
        plotly_script=charts.plotly_script(),
        duration_chart=charts.duration_histogram(cuts),
        metric_charts=metric_charts,
        language_chart=charts.category_bar(stats["languages"], "Language"),
        gender_chart=charts.category_bar(stats["genders"], "Gender"),
        samples=samples,
    )
    output_path.write_text(html, encoding="utf-8")
    return output_path
