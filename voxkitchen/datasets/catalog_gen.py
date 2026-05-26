"""Render the dataset catalog (catalog.yaml) into committed Markdown.

Run as: ``python -m voxkitchen.datasets.catalog_gen [--out docs/datasets] [--check]``.
Rendering is pure and deterministic (stable sort, no timestamps) so the
"docs up to date" test never flaps.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path

from voxkitchen.datasets.catalog import DatasetEntry

_HEADER = (
    "<!-- AUTO-GENERATED from voxkitchen/datasets/catalog.yaml — do not edit; "
    "run python -m voxkitchen.datasets.catalog_gen -->\n"
)

_DISCLAIMER = (
    "> VoxKitchen only curates and links publicly available dataset *information* "
    "— it does not host or redistribute data. You are responsible for each "
    "dataset's license and for obtaining the data. The recommendations below are "
    "guidance to help you decide.\n"
)


@dataclass
class RecipeDownloadInfo:
    source: str
    size_range: str
    subsets: list[str] = field(default_factory=list)


def _sorted(entries: list[DatasetEntry]) -> list[DatasetEntry]:
    return sorted(entries, key=lambda e: e.id)


def render_all(
    entries: list[DatasetEntry],
    download_info: dict[str, RecipeDownloadInfo],
) -> dict[str, str]:
    """Return {repo-relative-doc-path: markdown} for the index + per-entry pages."""
    out: dict[str, str] = {"datasets/index.md": _render_index(entries)}
    for e in _sorted(entries):
        out[f"datasets/{e.id}.md"] = _render_page(e, download_info.get(e.id))
    return out


def _render_index(entries: list[DatasetEntry]) -> str:
    lines = [_HEADER, "# Dataset Catalog\n", _DISCLAIMER, ""]
    lines += [
        "| Dataset | Task | Languages | Hours | License | Access |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for e in _sorted(entries):
        hours = f"{e.hours:g}" if e.hours is not None else "—"
        access = "recipe" if e.recipe else "manual"
        lines.append(
            f"| [{e.name}](./{e.id}.md) | {', '.join(e.task)} | "
            f"{', '.join(e.languages)} | {hours} | {e.license} | {access} |"
        )
    lines.append("")
    lines.append("## Browse by task\n")
    all_tasks = sorted({t for e in entries for t in e.task})
    for task in all_tasks:
        lines.append(f"### {task}\n")
        for e in _sorted(entries):
            if task in e.task:
                lines.append(f"- [{e.name}](./{e.id}.md) — {e.summary}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def _render_page(e: DatasetEntry, dl: RecipeDownloadInfo | None) -> str:
    lines = [_HEADER, f"# {e.name}\n", e.summary, ""]
    lines.append(f"- **Task:** {', '.join(e.task)}")
    lines.append(f"- **Languages:** {', '.join(e.languages)}")
    if e.hours is not None:
        lines.append(f"- **Hours:** {e.hours:g}")
    if e.domain:
        lines.append(f"- **Domain:** {e.domain}")
    lines.append(f"- **License:** {e.license}")
    lines.append(f"- **Homepage:** [{e.homepage}]({e.homepage})")
    if e.paper:
        lines.append(f"- **Paper:** [{e.paper}]({e.paper})")
    lines.append("")
    lines.append("## Recommendation\n")
    lines.append(e.recommendation)
    lines.append("")
    lines.append("## Getting the data\n")
    if e.recipe and dl is not None:
        lines.append(
            f"Downloadable via VoxKitchen (`{e.recipe}`, source: {dl.source}, "
            f"size: {dl.size_range}):\n"
        )
        lines.append("```bash")
        lines.append(f"vkit docker download --tag slim {e.recipe} --root ./data/{e.id}")
        lines.append("```")
        if dl.subsets:
            lines.append(f"\nSubsets: {', '.join(dl.subsets)}.")
    else:
        lines.append(f"Obtain from the [dataset homepage]({e.homepage}).")
    if e.notes:
        lines.append(f"\n{e.notes}")
    lines.append("")
    if e.recommended_pipeline:
        lines.append("## Suggested processing\n")
        # Plain code reference, not a relative link: the pipeline lives outside
        # the docs/ tree, so a clickable ../../ link would break mkdocs --strict.
        lines.append(
            f"A recommended VoxKitchen pipeline ships in the repository at "
            f"`{e.recommended_pipeline}` — run it with `vkit docker run`."
        )
    return "\n".join(lines).rstrip() + "\n"


def build_download_info(entries: list[DatasetEntry]) -> dict[str, RecipeDownloadInfo]:
    """For recipe-backed entries, pull source/size/subsets from the recipe registry."""
    from voxkitchen.datasets.recipe_meta import download_source_label, format_size_range
    from voxkitchen.ingest.recipes import get_recipe

    info: dict[str, RecipeDownloadInfo] = {}
    for e in entries:
        if not e.recipe:
            continue
        recipe = get_recipe(e.recipe)
        # FLEURS-style recipes use a custom download() with no download_urls.
        source = (
            download_source_label(recipe.download_urls) if recipe.download_urls else "HuggingFace"
        )
        info[e.id] = RecipeDownloadInfo(
            source=source,
            size_range=format_size_range(recipe.download_sizes),
            subsets=sorted(recipe.download_urls.keys()),
        )
    return info


def write_docs(
    entries: list[DatasetEntry],
    download_info: dict[str, RecipeDownloadInfo],
    out_dir: Path,
) -> None:
    """Render and write all docs under out_dir (filenames only, no 'datasets/' prefix)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    for rel, md in render_all(entries, download_info).items():
        (out_dir / Path(rel).name).write_text(md, encoding="utf-8")


def check_docs(
    entries: list[DatasetEntry],
    download_info: dict[str, RecipeDownloadInfo],
    out_dir: Path,
) -> list[str]:
    """Return filenames whose on-disk content differs from a fresh render (sorted)."""
    drift: list[str] = []
    for rel, md in render_all(entries, download_info).items():
        target = out_dir / Path(rel).name
        if not target.is_file() or target.read_text(encoding="utf-8") != md:
            drift.append(Path(rel).name)
    return sorted(drift)


def main() -> int:
    from voxkitchen.datasets.catalog import load_catalog

    parser = argparse.ArgumentParser(description="Render the dataset catalog.")
    parser.add_argument("--out", type=Path, default=Path("docs/datasets"))
    parser.add_argument(
        "--check",
        action="store_true",
        help="Exit non-zero if committed docs differ from catalog.yaml.",
    )
    args = parser.parse_args()

    entries = load_catalog()
    info = build_download_info(entries)
    if args.check:
        drift = check_docs(entries, info, args.out)
        if drift:
            print("dataset docs are stale; run python -m voxkitchen.datasets.catalog_gen")
            print("  drifted:", ", ".join(drift))
            return 1
        print("dataset docs up to date")
        return 0
    write_docs(entries, info, args.out)
    print(f"wrote {len(entries) + 1} files to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
