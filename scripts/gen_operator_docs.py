#!/usr/bin/env python3
"""Generate operator reference documentation from the operator registry.

Usage:
    python scripts/gen_operator_docs.py > docs/reference/operators.md
    # or
    python scripts/gen_operator_docs.py --output docs/reference/operators.md
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Display-friendly category names and order
CATEGORY_LABELS = {
    "basic": "Audio Processing",
    "segment": "Segmentation",
    "augment": "Data Augmentation",
    "annotate": "Annotation",
    "quality": "Quality & Filtering",
    "synthesize": "Synthesis",
    "pack": "Output / Packing",
    "noop": "Utility",
}

CATEGORY_ORDER = list(CATEGORY_LABELS.keys())


def get_category(op_cls: type) -> str:
    parts = op_cls.__module__.split(".")
    return parts[2] if len(parts) > 2 else "other"


def format_type(annotation: Any) -> str:
    if annotation is None:
        return "Any"
    origin = getattr(annotation, "__origin__", None)
    if origin is not None:
        args = getattr(annotation, "__args__", ())
        args_str = ", ".join(format_type(a) for a in args)
        origin_name = getattr(origin, "__name__", str(origin))
        return f"{origin_name}[{args_str}]" if args_str else origin_name
    name = getattr(annotation, "__name__", None)
    if name:
        return name
    return str(annotation).replace("typing.", "")


def format_table_cell(value: str) -> str:
    return value.replace("|", "\\|").replace("\n", "<br>")


def format_yaml_value(value: Any) -> str:
    if isinstance(value, str) and value.startswith("<") and value.endswith(">"):
        return value
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int | float):
        return str(value)
    if isinstance(value, str):
        if not value or value.lower() in {"null", "none", "true", "false", "yes", "no"}:
            return json.dumps(value)
        if any(ch in value for ch in [":", "#", "{", "}", "[", "]", ","]):
            return json.dumps(value)
        return value
    return json.dumps(value)


def docker_tag_for_required_extras(required_extras: list[str]) -> str:
    from voxkitchen.cli.hints import docker_tag_for_extras

    if not required_extras:
        return "slim"
    tags = {docker_tag_for_extras(extra) for extra in required_extras}
    tags.discard(None)
    if len(tags) == 1:
        return next(iter(tags))
    return "latest"


def format_docstring(doc: str) -> tuple[str, str]:
    """Return the one-line summary and MkDocs-friendly body text."""
    if not doc:
        return "", ""

    raw_lines = doc.splitlines()
    first_line = raw_lines[0].strip()
    body_lines = [line.rstrip() for line in raw_lines[1:]]

    lines: list[str] = []
    i = 0
    while i < len(body_lines):
        line = body_lines[i]
        stripped = line.strip()
        if stripped == ".. warning::":
            lines.append("!!! warning")
            i += 1
            while i < len(body_lines):
                warning_line = body_lines[i]
                if warning_line.strip() == "":
                    i += 1
                    break
                if not warning_line.startswith((" ", "\t")):
                    break
                lines.append(f"    {warning_line.strip()}")
                i += 1
            continue

        lines.append(stripped)
        i += 1

    return first_line, "\n".join(lines).strip()


def generate() -> str:
    from voxkitchen.operators.registry import get_operator, list_operators

    names = list_operators()
    groups: dict[str, list[tuple[str, type]]] = {}
    for name in names:
        op_cls = get_operator(name)
        cat = get_category(op_cls)
        groups.setdefault(cat, []).append((name, op_cls))

    lines: list[str] = []
    lines.append("# Operator Reference")
    lines.append("")
    lines.append(
        f"VoxKitchen ships with **{len(names)} built-in operators** across "
        f"{len(groups)} categories."
    )
    lines.append("")
    lines.append("!!! tip")
    lines.append(
        "    Run `vkit operators` to see this list in your terminal, "
        "or `vkit operators show <name>` for details."
    )
    lines.append("")

    # Table of contents
    lines.append("## Categories")
    lines.append("")
    sorted_cats = sorted(
        groups.keys(), key=lambda c: CATEGORY_ORDER.index(c) if c in CATEGORY_ORDER else 99
    )
    for cat in sorted_cats:
        label = CATEGORY_LABELS.get(cat, cat.title())
        count = len(groups[cat])
        noun = "operator" if count == 1 else "operators"
        lines.append(f"- [{label}](#{cat}) ({count} {noun})")
    lines.append("")

    # Each category
    for cat in sorted_cats:
        label = CATEGORY_LABELS.get(cat, cat.title())
        ops = groups[cat]
        lines.append(f"## {label} {{ #{cat} }}")
        lines.append("")

        for op_name, op_cls in ops:
            doc = (op_cls.__doc__ or "").strip()
            first_line, full_doc = format_docstring(doc)

            lines.append(f"### `{op_name}`")
            lines.append("")
            lines.append(f"**{first_line}**")
            lines.append("")
            if full_doc:
                lines.append(full_doc)
                lines.append("")

            # Metadata
            tag = docker_tag_for_required_extras(list(op_cls.required_extras))
            lines.append(f"- **Device:** {op_cls.device}")
            lines.append(f"- **Runtime:** `vkit docker run --tag {tag} <yaml>`")
            lines.append(f"- **Produces audio:** {'Yes' if op_cls.produces_audio else 'No'}")
            lines.append("")

            # Config table
            cfg_cls = op_cls.config_cls
            fields = cfg_cls.model_fields
            if fields:
                lines.append("| Parameter | Type | Default | Description |")
                lines.append("|-----------|------|---------|-------------|")
                for field_name, info in fields.items():
                    type_str = format_type(info.annotation)
                    try:
                        d = info.default
                        default_str = "required" if repr(d) == "PydanticUndefined" else f"`{d}`"
                    except Exception:
                        default_str = "required"
                    desc = info.description or ""
                    lines.append(
                        f"| `{format_table_cell(field_name)}` | "
                        f"{format_table_cell(type_str)} | "
                        f"{format_table_cell(default_str)} | "
                        f"{format_table_cell(desc)} |"
                    )
                lines.append("")

            # YAML example
            lines.append("```yaml")
            lines.append(f"- name: my_{op_name}")
            lines.append(f"  op: {op_name}")
            if fields:
                lines.append("  args:")
                for field_name, info in fields.items():
                    try:
                        d = info.default
                        if repr(d) == "PydanticUndefined":
                            d = f"<{format_type(info.annotation)}>"
                    except Exception:
                        d = "..."
                    lines.append(f"    {field_name}: {format_yaml_value(d)}")
            lines.append("```")
            lines.append("")
            lines.append("---")
            lines.append("")

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate operator reference docs.")
    parser.add_argument(
        "--output", "-o", type=str, default=None, help="Output file path. Defaults to stdout."
    )
    args = parser.parse_args()

    content = generate()
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(content)
        print(f"Written to {args.output}", file=sys.stderr)
    else:
        print(content)


if __name__ == "__main__":
    main()
