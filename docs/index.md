# VoxKitchen

Declarative speech data processing toolkit. Write a YAML recipe, run `vkit run`, get training-ready data.

## Documentation

- [Getting Started](getting-started.md) — install, first pipeline, inspect results
- [Data Protocol](concepts/data-protocol.md) — Recording, Supervision, Cut, CutSet, Provenance

## Quick reference

```bash
vkit operators                      # list all operators
vkit operators show <name>          # config fields + YAML example
vkit run pipeline.yaml --dry-run    # validate without executing
```

## Links

- [Example pipelines](https://github.com/voxkitchen/voxkitchen/tree/main/examples/pipelines)
- [GitHub](https://github.com/voxkitchen/voxkitchen)
- [License](https://github.com/voxkitchen/voxkitchen/blob/main/LICENSE) (Apache 2.0)
