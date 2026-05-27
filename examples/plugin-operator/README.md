# Example VoxKitchen operator plugin

A minimal third-party operator (`word_count`) distributed via a Python entry
point. Use it as a template.

## How it works

`voxkitchen_example_plugin/operator.py` defines `WordCountOperator` (subclasses
`voxkitchen.operators.Operator`, declares a `reads`/`writes` field contract).
It is registered through the entry point in `pyproject.toml`:

    [project.entry-points."voxkitchen.operators"]
    word_count = "voxkitchen_example_plugin.operator:WordCountOperator"

Do **not** also decorate the class with `@register_operator` — the entry point
is the registration mechanism for plugins.

## Try it

    pip install -e examples/plugin-operator
    vkit operators            # 'word_count' now appears in the list
    vkit operators show word_count

Use it in a pipeline like any built-in operator: `op: word_count`.

Target the operator API: `from voxkitchen.operators import OPERATOR_API_VERSION`.
