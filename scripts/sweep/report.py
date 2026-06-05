"""Render the operator sweep results as a Markdown report.

Format (per design spec):

    # Operator sweep — YYYY-MM-DD HH:MM
    **Verdict:** A/B PASS, C SKIP, D FAIL · TOTAL_TIME total
    ## Failures
    - op (image) — message
    ## Skips
    - op (image) — message
    ## Full results
    | Op | Image | Result | Wall | Note |
"""

from __future__ import annotations

import datetime as _dt
from pathlib import Path


def write_report(*, records: list, path: Path) -> None:
    """Render records to a Markdown file."""
    path.parent.mkdir(parents=True, exist_ok=True)

    total = len(records)
    passes = sum(1 for r in records if r.verdict == "PASS")
    fails = [r for r in records if r.verdict == "FAIL"]
    skips = [r for r in records if r.verdict == "SKIP"]
    total_wall = sum(r.wall_seconds for r in records)

    now = _dt.datetime.now().strftime("%Y-%m-%d %H:%M")
    lines: list[str] = []
    lines.append(f"# Operator sweep — {now}\n")
    lines.append(
        f"**Verdict:** {passes}/{total} PASS, {len(skips)} SKIP, {len(fails)} FAIL"
        f" · {_format_seconds(total_wall)} total\n"
    )

    if fails:
        lines.append("\n## Failures\n")
        for r in fails:
            lines.append(f"- `{r.op}` ({r.image}) — {r.message}\n")
    if skips:
        lines.append("\n## Skips\n")
        for r in skips:
            lines.append(f"- `{r.op}` ({r.image}) — {r.message}\n")

    lines.append("\n## Full results\n")
    lines.append("| Op | Image | Result | Wall | Note |\n")
    lines.append("|---|---|---|---|---|\n")
    glyph = {"PASS": "✅ PASS", "FAIL": "❌ FAIL", "SKIP": "⏭ SKIP"}
    for r in records:
        lines.append(
            f"| `{r.op}` | {r.image} | {glyph.get(r.verdict, r.verdict)} | "
            f"{r.wall_seconds:.1f}s | {r.message} |\n"
        )

    path.write_text("".join(lines), encoding="utf-8")


def _format_seconds(seconds: float) -> str:
    """Format seconds as 'Hh Mm Ss' or 'Mm Ss' or 'Ss'."""
    h, rem = divmod(int(seconds), 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h {m}m {s}s"
    if m:
        return f"{m}m {s}s"
    return f"{s}s"
