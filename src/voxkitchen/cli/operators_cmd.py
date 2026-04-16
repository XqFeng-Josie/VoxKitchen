"""Implementation of `vkit operators` subcommands."""

from __future__ import annotations

import re
from typing import Any

import typer
from rich.console import Console
from rich.table import Table

console = Console()

operators_app = typer.Typer(
    name="operators",
    help="List and inspect available operators.",
    invoke_without_command=True,
)

# Display-friendly category names and order
_CATEGORY_LABELS = {
    "basic": "Audio",
    "segment": "Segmentation",
    "augment": "Augmentation",
    "annotate": "Annotation",
    "quality": "Quality",
    "pack": "Pack",
    "noop": "Utility",
}

_CATEGORY_ORDER = list(_CATEGORY_LABELS.keys())


def _get_category(op_cls: type) -> str:
    """Extract category from module path: voxkitchen.operators.<category>.<name>."""
    parts = op_cls.__module__.split(".")
    return parts[2] if len(parts) > 2 else "other"


@operators_app.callback()
def list_all(ctx: typer.Context) -> None:
    """List all registered operators with a brief description."""
    if ctx.invoked_subcommand is not None:
        return
    from voxkitchen.operators.registry import get_operator, list_operators

    names = list_operators()

    # Group by category
    groups: dict[str, list[tuple[str, type]]] = {}
    for name in names:
        op_cls = get_operator(name)
        cat = _get_category(op_cls)
        groups.setdefault(cat, []).append((name, op_cls))

    t = Table(title=f"Available operators ({len(names)})")
    t.add_column("Category", style="bold")
    t.add_column("Name")
    t.add_column("Device")
    t.add_column("Description")

    # Print in category order
    sorted_cats = sorted(groups.keys(), key=lambda c: _CATEGORY_ORDER.index(c) if c in _CATEGORY_ORDER else 99)
    for cat in sorted_cats:
        label = _CATEGORY_LABELS.get(cat, cat.title())
        ops = groups[cat]
        for i, (name, op_cls) in enumerate(ops):
            doc = (op_cls.__doc__ or "").strip().split("\n")[0]
            cat_col = label if i == 0 else ""
            t.add_row(cat_col, name, op_cls.device, doc)
        # Add separator between categories (except last)
        if cat != sorted_cats[-1]:
            t.add_row("", "", "", "")

    console.print(t)
    console.print()
    console.print("[dim]Use[/dim] [bold]vkit operators show <name>[/bold] [dim]to see config fields and YAML example.[/dim]")
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
    if op_cls.required_extras:
        extras = ",".join(op_cls.required_extras)
        console.print(f"  install: pip install voxkitchen[{extras}]")
    else:
        console.print("  install: pip install voxkitchen")
    console.print()

    # Docstring — highlight warnings
    if op_cls.__doc__:
        doc = op_cls.__doc__.strip()
        # Extract .. warning:: blocks and render in yellow
        warning_match = re.search(r"\.\. warning::\s*\n\s+(.+?)(?:\n\n|\Z)", doc, re.DOTALL)
        if warning_match:
            doc = doc[: warning_match.start()].rstrip()
            console.print(doc)
            console.print(f"\n  [yellow]Warning: {warning_match.group(1).strip()}[/yellow]")
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
