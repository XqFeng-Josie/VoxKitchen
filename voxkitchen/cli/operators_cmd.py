"""Implementation of `vkit operators` subcommands."""

from __future__ import annotations

import re
import textwrap
from typing import Any

import typer
from rich.console import Console
from rich.table import Table

from voxkitchen.cli.hints import docker_tag_for_extras

console = Console()

operators_app = typer.Typer(
    name="operators",
    help="List and inspect available operators.",
    invoke_without_command=True,
)

# Display-friendly category names and order. The keys are the directory names
# under voxkitchen/operators/ — module path is the source of truth for which
# category an operator belongs to (see _get_category).
_CATEGORY_LABELS = {
    "basic": "Audio",
    "segment": "Segmentation",
    "augment": "Augmentation",
    "annotate": "Annotation",
    "quality": "Quality",
    "synthesize": "Synthesis",
    "pack": "Pack",
    "noop": "Utility",
}

_CATEGORY_ORDER = list(_CATEGORY_LABELS.keys())


def _get_category(op_cls: type) -> str:
    """Extract category from module path: voxkitchen.operators.<category>.<name>."""
    parts = op_cls.__module__.split(".")
    return parts[2] if len(parts) > 2 else "other"


def _docker_runtime_hint(required_extras: list[str]) -> str:
    if not required_extras:
        return "vkit docker run --tag slim <yaml>"

    tags = {tag for extra in required_extras if (tag := docker_tag_for_extras(extra)) is not None}
    if len(tags) == 1:
        return f"vkit docker run --tag {sorted(tags)[0]} <yaml>"
    return "vkit docker run --tag latest <yaml>"


@operators_app.callback()
def list_all(
    ctx: typer.Context,
    category: str | None = typer.Option(
        None,
        "--category",
        "-c",
        help=(f"Show only operators in a single category. One of: {', '.join(_CATEGORY_LABELS)}."),
    ),
) -> None:
    """List all registered operators with a brief description."""
    if ctx.invoked_subcommand is not None:
        return
    _render_table(category=category)


@operators_app.command()
def search(
    keyword: str = typer.Argument(
        ..., help="Substring matched (case-insensitively) against name and description."
    ),
) -> None:
    """Find operators by keyword in name or one-line description.

    Example: ``vkit operators search noise`` lists `noise_augment`,
    `snr_estimate`, `speech_enhance`, etc.
    """
    _render_table(keyword=keyword)


def _render_table(*, category: str | None = None, keyword: str | None = None) -> None:
    """Group registered operators by category and print a Rich table.

    Filters (``category`` and ``keyword``) compose. An empty result set exits
    with code 1 so shell scripts can branch on "no matches" without parsing
    text. Interactive callers see a styled message either way.
    """
    from voxkitchen.operators.registry import get_operator, list_operators

    if category is not None and category not in _CATEGORY_LABELS:
        console.print(
            f"[red]error:[/red] unknown category {category!r}. "
            f"Available: {', '.join(_CATEGORY_LABELS)}."
        )
        raise typer.Exit(code=2)

    needle = keyword.lower() if keyword else None

    groups: dict[str, list[tuple[str, type]]] = {}
    total = 0
    for name in list_operators():
        op_cls = get_operator(name)
        cat = _get_category(op_cls)
        if category is not None and cat != category:
            continue
        if needle is not None:
            # Match the same one-line summary the table displays in the
            # Description column. Matching the full docstring would surface
            # operators whose match text is hidden — confusing for users who
            # can't tell why a result was returned.
            summary = (op_cls.__doc__ or "").strip().split("\n", 1)[0].lower()
            if needle not in name.lower() and needle not in summary:
                continue
        groups.setdefault(cat, []).append((name, op_cls))
        total += 1

    if total == 0:
        if keyword and category:
            msg = f"no operators in category {category!r} match {keyword!r}"
        elif keyword:
            msg = f"no operators match {keyword!r}"
        else:
            msg = f"no operators in category {category!r}"
        console.print(f"[yellow]{msg}[/yellow]")
        raise typer.Exit(code=1)

    if keyword:
        title = f"Operators matching {keyword!r} ({total})"
    elif category:
        title = f"Operators in '{_CATEGORY_LABELS[category]}' ({total})"
    else:
        title = f"Available operators ({total})"

    t = Table(title=title)
    t.add_column("Category", style="bold")
    t.add_column("Name")
    t.add_column("Device")
    t.add_column("Description")

    sorted_cats = sorted(
        groups.keys(), key=lambda c: _CATEGORY_ORDER.index(c) if c in _CATEGORY_ORDER else 99
    )
    for cat in sorted_cats:
        label = _CATEGORY_LABELS.get(cat, cat.title())
        ops = groups[cat]
        for i, (name, op_cls) in enumerate(ops):
            doc = (op_cls.__doc__ or "").strip().split("\n")[0]
            cat_col = label if i == 0 else ""
            t.add_row(cat_col, name, op_cls.device, doc)
        if cat != sorted_cats[-1]:
            t.add_row("", "", "", "")

    console.print(t)
    console.print()
    console.print(
        "[dim]Use[/dim] [bold]vkit operators show <name>[/bold] "
        "[dim]to see config fields and YAML example.[/dim]"
    )
    console.print()


@operators_app.command(name="show")
def show(name: str = typer.Argument(..., help="Operator name.")) -> None:
    """Show detailed info for an operator: description, device, config fields."""
    from voxkitchen.operators.registry import get_operator

    try:
        op_cls = get_operator(name)
    except Exception as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1) from exc

    # Header
    cat = _get_category(op_cls)
    label = _CATEGORY_LABELS.get(cat, cat.title())
    console.print(f"\n[bold]{op_cls.name}[/bold]  (device: {op_cls.device}, category: {label})")
    console.print(f"  runtime: {_docker_runtime_hint(list(op_cls.required_extras))}")
    console.print()

    # Docstring — highlight warnings
    if op_cls.__doc__:
        doc = op_cls.__doc__.strip()
        # Extract .. warning:: blocks and render in yellow
        warning_match = re.search(r"\.\. warning::\s*\n\s+(.+?)(?:\n\n|\Z)", doc, re.DOTALL)
        if warning_match:
            doc = doc[: warning_match.start()].rstrip()
            warning = textwrap.dedent(warning_match.group(1)).strip()
            console.print(doc)
            console.print(f"\n  [yellow]Warning: {warning}[/yellow]")
        else:
            console.print(doc)
        console.print()

    # Config fields
    cfg_cls = op_cls.config_cls
    fields: dict[str, Any] = cfg_cls.model_fields
    if not fields:
        console.print("[dim]No config parameters.[/dim]")
        return

    t = Table(title="Config")
    t.add_column("Field")
    t.add_column("Type")
    t.add_column("Default")
    t.add_column("Description")

    for field_name, info in fields.items():
        type_str = _format_type(info.annotation)
        default = info.default if info.default is not info.default.__class__ else "required"
        # PydanticUndefined check
        try:
            default_str = "required" if repr(default) == "PydanticUndefined" else repr(default)
        except Exception:
            default_str = "required"
        desc = info.description or ""
        t.add_row(field_name, type_str, default_str, desc)
    console.print(t)

    # YAML example
    console.print("\n[bold]YAML example:[/bold]")
    console.print(f"  - name: my_{op_cls.name}")
    console.print(f"    op: {op_cls.name}")
    if fields:
        console.print("    args:")
        for field_name, info in fields.items():
            try:
                d = info.default
                if repr(d) == "PydanticUndefined":
                    d = f"<{_format_type(info.annotation)}>"
                console.print(f"      {field_name}: {d}")
            except Exception:
                console.print(f"      {field_name}: ...")
    console.print()


def _format_type(annotation: Any) -> str:
    """Turn a type annotation into a readable string."""
    if annotation is None:
        return "Any"
    origin = getattr(annotation, "__origin__", None)
    if origin is not None:
        args = getattr(annotation, "__args__", ())
        args_str = ", ".join(_format_type(a) for a in args)
        origin_name = getattr(origin, "__name__", str(origin))
        return f"{origin_name}[{args_str}]" if args_str else origin_name
    name: str | None = getattr(annotation, "__name__", None)
    if name:
        return name
    return str(annotation).replace("typing.", "")
