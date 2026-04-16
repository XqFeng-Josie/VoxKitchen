# Contributing to VoxKitchen

Thank you for your interest in contributing! This guide covers everything you need to get started.

## Development Setup

```bash
# Clone and install in development mode
git clone https://github.com/voxkitchen/voxkitchen.git
cd voxkitchen
conda create -n voxkitchen python=3.11 -y
conda activate voxkitchen
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install -e ".[dev]"

# Set up pre-commit hooks
pre-commit install
```

## Running Tests

```bash
# All tests (skip slow model-download tests)
pytest -v -m "not slow and not gpu"

# Specific test file
pytest tests/unit/operators/augment/test_speed_perturb.py -v

# With coverage
pytest --cov=voxkitchen --cov-report=term-missing
```

## Code Style

We use **ruff** for linting and formatting, **mypy** for type checking.

```bash
ruff check voxkitchen tests   # Lint
ruff format voxkitchen tests  # Format
mypy voxkitchen tests         # Type check
```

Pre-commit hooks run these automatically on `git commit`.

## Commit Messages

We follow [Conventional Commits](https://www.conventionalcommits.org/):

```
feat(operators): add noise_augment operator
fix: speaker_embed expects torch tensor, not numpy
docs: add TTS tutorial
chore: update CI matrix
refactor(pipeline): simplify GC logic
test: add lazy CutSet tests
```

Scopes: `operators`, `pipeline`, `schema`, `cli`, `viz`, `ingest`, `tools`.

## Adding a New Operator

1. **Create the operator file** in the appropriate category:

   ```
   voxkitchen/operators/<category>/<name>.py
   ```

   Categories: `basic/`, `segment/`, `augment/`, `annotate/`, `quality/`, `pack/`

2. **Follow the pattern** — every operator has:

   ```python
   from voxkitchen.operators.base import Operator, OperatorConfig
   from voxkitchen.operators.registry import register_operator

   class MyConfig(OperatorConfig):
       param1: str = "default"

   @register_operator
   class MyOperator(Operator):
       """One-line description shown in vkit operators."""

       name = "my_operator"
       config_cls = MyConfig
       device = "cpu"  # or "gpu"
       produces_audio = False
       reads_audio_bytes = True

       def process(self, cuts: CutSet) -> CutSet:
           ...
   ```

3. **Register** in `voxkitchen/operators/__init__.py`:

   ```python
   from voxkitchen.operators.<category> import <module> as _alias  # noqa: F401
   ```

   For optional dependencies, wrap in `try/except ImportError`.

4. **Write tests** in `tests/unit/operators/<category>/test_<name>.py`

5. **Create an example** in `examples/pipelines/` (optional but appreciated)

6. **Regenerate operator docs**:

   ```bash
   python scripts/gen_operator_docs.py -o docs/reference/operators.md
   ```

## Adding a New Recipe

1. Create `voxkitchen/ingest/recipes/<name>.py`
2. Subclass `Recipe`, implement `prepare()`, optionally override `download()`
3. Call `register_recipe(YourRecipe())` at module bottom
4. Import in `voxkitchen/ingest/recipes/__init__.py`
5. Add tests in `tests/unit/ingest/recipes/`

## Pull Request Process

1. Fork the repo and create a branch from `main`
2. Make your changes with tests
3. Ensure all tests pass: `pytest -v -m "not slow and not gpu"`
4. Commit with conventional commit messages
5. Open a PR with a clear description of what and why

## Questions?

Open a [GitHub Discussion](https://github.com/voxkitchen/voxkitchen/discussions) or file an issue.
