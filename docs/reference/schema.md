# Pipeline JSON Schema

VoxKitchen ships a JSON Schema for `pipeline.yaml` so editors that speak the
[YAML Language Server protocol](https://github.com/redhat-developer/yaml-language-server)
provide autocompletion, hover documentation, and inline validation while you
type.

The schema is regenerated from `voxkitchen/pipeline/spec.py` plus the registered
operators and lives at
[`docs/schemas/pipeline.schema.json`](https://github.com/XqFeng-Josie/VoxKitchen/blob/main/docs/schemas/pipeline.schema.json).
It's served via raw.githubusercontent.com so no extra hosting is needed.

## Use the schema in your editor

Pipelines scaffolded by `vkit init` already get the right directive on
line 1:

```yaml
# yaml-language-server: $schema=https://raw.githubusercontent.com/XqFeng-Josie/VoxKitchen/main/docs/schemas/pipeline.schema.json
version: "0.1"
name: my-pipeline
...
```

For an existing pipeline, paste that line at the top. The following editors
pick it up automatically:

| Editor | Plugin |
|---|---|
| VS Code | [YAML by Red Hat](https://marketplace.visualstudio.com/items?itemName=redhat.vscode-yaml) |
| Neovim | [yamlls](https://github.com/redhat-developer/yaml-language-server) via lspconfig |
| JetBrains | Built-in JSON Schema support |
| Sublime / Helix / Zed | Any client wired to yamlls |

You'll see:

- Autocomplete on `op:` — all 51 operator names show up
- Errors for unknown `op:` values
- Errors for unknown top-level keys (`extra="forbid"` is honored)
- Hover docs for fields like `gc_mode`, `num_gpus`, etc.

## Generate the schema yourself

```bash
vkit schema export --out docs/schemas/pipeline.schema.json
```

The exported document captures whichever operators are registered in the
running Python environment. From a source checkout with the `[dev]` extras
installed, that's already 51 operators. To regenerate a fully-loaded schema
that includes operator-specific defaults for every env, run the command
inside the `:latest` Docker image:

```bash
vkit docker shell --tag latest
# inside container:
vkit schema export --out /app/docs/schemas/pipeline.schema.json
```

## Re-generating after operator changes

Whenever you add, rename, or change an operator's config fields, regenerate
the snapshot and commit it:

```bash
vkit schema export --out docs/schemas/pipeline.schema.json
git add docs/schemas/pipeline.schema.json
```

CI does not auto-regenerate this file — it's a maintainer responsibility.
A stale snapshot causes editors to flag new operators as "unknown op", which
is loud enough to catch.

## Versioning

The schema URL written by `vkit init` points at the `main` branch:

```
https://raw.githubusercontent.com/XqFeng-Josie/VoxKitchen/main/docs/schemas/pipeline.schema.json
```

Editors refetch this file periodically, so a scaffolded pipeline picks up
new operators added to `main` without any local change. The trade-off is
that changes on `main` can affect validation of pipelines users wrote
weeks earlier — adding operators is always safe, but renaming or removing
one (which we currently do not do) would silently invalidate older
pipelines.

If you need a frozen view, replace `main` with a release tag in the
`# yaml-language-server:` directive at the top of your `pipeline.yaml`:

```yaml
# yaml-language-server: $schema=https://raw.githubusercontent.com/XqFeng-Josie/VoxKitchen/v0.3.0/docs/schemas/pipeline.schema.json
```

Pin to a tag when reproducibility matters more than picking up new
operators; otherwise leave the default.
