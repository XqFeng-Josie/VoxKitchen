#!/usr/bin/env python3
"""Generate operator reference documentation from the operator registry.

Usage:
    python scripts/gen_operator_docs.py > docs/reference/operators.md
    # or
    python scripts/gen_operator_docs.py --output docs/reference/operators.md
"""

from __future__ import annotations

import argparse
import sys
from typing import Any


# Display-friendly category names and order
CATEGORY_LABELS = {
    "basic": "Audio Processing",
    "segment": "Segmentation",
    "augment": "Data Augmentation",
    "annotate": "Annotation",
    "quality": "Quality & Filtering",
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
    lines.append(f"VoxKitchen ships with **{len(names)} built-in operators** across "
                 f"{len(groups)} categories.")
    lines.append("")
    lines.append("!!! tip")
    lines.append("    Run `vkit operators` to see this list in your terminal, "
                 "or `vkit operators show <name>` for details.")
    lines.append("")

    # Table of contents
    lines.append("## Categories")
    lines.append("")
    sorted_cats = sorted(groups.keys(),
                         key=lambda c: CATEGORY_ORDER.index(c) if c in CATEGORY_ORDER else 99)
    for cat in sorted_cats:
        label = CATEGORY_LABELS.get(cat, cat.title())
        count = len(groups[cat])
        lines.append(f"- [{label}](#{cat}) ({count} operators)")
    lines.append("")

    # Each category
    for cat in sorted_cats:
        label = CATEGORY_LABELS.get(cat, cat.title())
        ops = groups[cat]
        lines.append(f"## {label} {{ #{cat} }}")
        lines.append("")

        for op_name, op_cls in ops:
            doc = (op_cls.__doc__ or "").strip()
            first_line = doc.split("\n")[0] if doc else ""
            full_doc = "\n".join(line.strip() for line in doc.split("\n")[1:]).strip()

            lines.append(f"### `{op_name}`")
            lines.append("")
            lines.append(f"**{first_line}**")
            lines.append("")
            if full_doc:
                lines.append(full_doc)
                lines.append("")

            # Metadata
            extras = ",".join(op_cls.required_extras) if op_cls.required_extras else "core"
            lines.append(f"- **Device:** {op_cls.device}")
            lines.append(f"- **Install:** `pip install voxkitchen[{extras}]`")
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
                    lines.append(f"| `{field_name}` | {type_str} | {default_str} | {desc} |")
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
                    lines.append(f"    {field_name}: {d}")
            lines.append("```")
            lines.append("")
            lines.append("---")
            lines.append("")

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate operator reference docs.")
    parser.add_argument("--output", "-o", type=str, default=None,
                        help="Output file path. Defaults to stdout.")
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
