"""Implementation of `vkit operators` subcommands."""

from __future__ import annotations

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


@operators_app.callback()
def list_all(ctx: typer.Context) -> None:
    """List all registered operators with a brief description."""
    if ctx.invoked_subcommand is not None:
        return
    from voxkitchen.operators.registry import get_operator, list_operators

    names = list_operators()
    t = Table(title=f"Available operators ({len(names)})")
    t.add_column("Name")
    t.add_column("Device")
    t.add_column("Description")

    for name in names:
        op_cls = get_operator(name)
        doc = (op_cls.__doc__ or "").strip().split("\n")[0]
        t.add_row(name, op_cls.device, doc)
    console.print(t)


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
    console.print(f"\n[bold]{op_cls.name}[/bold]  (device: {op_cls.device})")
    if op_cls.required_extras:
        extras = ",".join(op_cls.required_extras)
        console.print(f"  install: pip install voxkitchen[{extras}]")
    console.print()

    # Docstring
    if op_cls.__doc__:
        console.print(op_cls.__doc__.strip())
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
