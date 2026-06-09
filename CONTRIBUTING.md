# Contributing to VoxKitchen

Thank you for your interest in contributing! This guide covers everything you need to get started.

## Development Setup

The local development environment is for editing code, running unit tests,
and building images. Pipeline execution should use `vkit docker run`, the
same path users get.

```bash
# Clone and install in development mode
git clone https://github.com/XqFeng-Josie/VoxKitchen.git
cd VoxKitchen
conda create -n voxkitchen python=3.11 -y
conda activate voxkitchen
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install -e ".[dev]"

# Set up commit-time and push-time hooks
pre-commit install
```

## Running Tests

```bash
# Local CI gate before push
scripts/check-ci.sh

# Fast pytest subset only
pytest -v -m "not slow and not gpu"

# Specific test file
pytest tests/unit/operators/augment/test_speed_perturb.py -v

# With coverage
pytest --cov=voxkitchen --cov-report=term-missing
```

**Test coverage layers.** Know what each layer does and does *not* cover:

- **Per-PR CI** installs only the core (`slim`) dependency cluster, so unit
  tests for heavy operators (ASR, diarization, TTS, fish-speech) `SKIP`.
- **`extras-ci`** (weekly + `workflow_dispatch`) installs the `asr` / `diarize`
  / `codec` / `tts-kokoro` extras on CPU and runs their previously-skipped
  tests. It does **not** cover fish-speech, cosyvoice, or chattts, and it runs
  against `pip`-installed extras, not the published Docker images.
- **The operator sweep** (`scripts/sweep/run.py`) is the only thing that
  exercises every operator inside its canonical published image. It is
  **manual** — run it before tagging/republishing images; nothing in CI does.

## Code Style

We use **ruff** for linting and formatting, **mypy** for type checking.

```bash
ruff check voxkitchen tests   # Lint
ruff format voxkitchen tests  # Format
ruff format --check voxkitchen tests  # CI format check
mypy voxkitchen tests         # Type check
```

Pre-commit hooks run fast style checks on `git commit`. The pre-push hook runs
`scripts/check-ci.sh`, which mirrors the fast CI gate before code leaves your
machine.

Before pushing manually, run:

```bash
scripts/check-ci.sh
```

`[dev]` installs test tooling plus the core dependencies needed by the
unit-test suite. Operator dependency clusters live in Docker images. Use
`vkit docker run --tag <image> ...`, `vkit docker doctor`, and
`vkit docker build <target>` when validating real pipeline behavior.

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

   Categories: `basic/`, `segment/`, `augment/`, `annotate/`, `quality/`, `synthesize/`, `pack/`

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

3. **Declare the field contract.** Add `reads`, `writes`, `optional_reads`,
   and `clears` ClassVars to your operator. The contract-completeness test
   fails if any of these are missing. For contracts that depend on config
   values, implement `dynamic_reads(self) -> list[str]` instead of (or in
   addition to) `reads`; access config via `self.config` inside the method.
   See [Field Contracts](docs/architecture.md#field-contracts) for the full
   token vocabulary.

   ```python
   reads: ClassVar[list[str]] = ["audio"]
   writes: ClassVar[list[str]] = ["metrics.snr"]
   optional_reads: ClassVar[list[str]] = []
   clears: ClassVar[list[str]] = []
   ```

4. **Register** in `voxkitchen/operators/__init__.py`:

   ```python
   from voxkitchen.operators.<category> import <module> as _alias  # noqa: F401
   ```

   For optional dependencies, wrap in `try/except ImportError`.

5. **Map the operator's `required_extras` to a Docker env**. If the
   operator declares any `required_extras = [...]`, every listed extras
   group must appear in `EXTRA_TO_ENV` in
   [`voxkitchen/runtime/env_resolver.py`](voxkitchen/runtime/env_resolver.py).
   The runner consults this table (plus `op_env_map.json` inside the
   Docker image) to route stages to the right env. If two extras in an
   operator map to different envs, the runner refuses to dispatch — split
   the extras or pick one env.

   **Git URL dependencies.** If your operator depends on a package that
   has no PyPI release and must be installed from git, do **not** add
   it to `pyproject.toml`'s `[project.optional-dependencies]`. PyPI
   rejects any uploaded distribution whose metadata declares a PEP 440
   direct reference (`pkg @ git+https://...`), regardless of the
   `[tool.hatch.metadata] allow-direct-references` flag — that flag
   only affects the build step, not what PyPI accepts on upload.
   Install the dependency from the Dockerfile instead, pinning to a
   specific commit so image builds stay reproducible:

   ```dockerfile
   RUN uv pip install --python /opt/voxkitchen/envs/<env>/bin/python \
           -c /app/docker/constraints/<env>.txt \
           "<pkg> @ git+https://github.com/.../<repo>.git@<sha>"
   ```

   See `wenet_asr` and `tts_fish_speech` in `docker/Dockerfile` for the
   existing pattern. The operator's `required_extras` string still needs
   an `EXTRA_TO_ENV` entry so the runner knows where to dispatch it,
   even though no matching pyproject group exists.

6. **Add to expected set** in
   [`voxkitchen/cli/doctor.py`](voxkitchen/cli/doctor.py)'s
   `EXPECTED_OPERATORS[<env>]` — this is what the `vkit doctor --expect <env>`
   smoke test at the end of each Docker build stage checks. If your
   operator is in the image, it must be in the expected set, or the
   build-time smoke test will flag it as "extra" (harmless but noisy).

7. **Write tests** in `tests/unit/operators/<category>/test_<name>.py`.

8. **Create an example** in `examples/pipelines/` (optional but appreciated).

9. **Regenerate operator docs**:

   ```bash
   python scripts/gen_operator_docs.py -o docs/reference/operators.md
   ```

10. **Regenerate the pipeline JSON Schema** so editors flag the new operator
   correctly:

   ```bash
   vkit schema export --out docs/schemas/pipeline.schema.json
   ```

   Commit the result alongside the operator change. CI does not do this
   automatically.

## Third-party operator plugins

A plugin is a separate pip package that registers operators through the
`voxkitchen.operators` entry-point group.  It is **not** a contribution to
the core repository — ship it as its own package.

| | Built-in operator | Third-party plugin |
|--|-------------------|--------------------|
| **Location** | `voxkitchen/operators/<category>/` | Separate pip package |
| **Registration** | `@register_operator` + import in `voxkitchen/operators/__init__.py` | `[project.entry-points."voxkitchen.operators"]` in `pyproject.toml` |
| **Decorator** | Required | **Do NOT use** — the entry point is the registration mechanism; decorating too would double-register |
| **Discovery** | Eager (imported at package load) | Lazy (discovered at first registry access) |

For the full authoring workflow — operator skeleton, `pyproject.toml` shape,
install/verify steps, stable API surface, and compatibility policy — see
[docs/guides/operator-plugins.md](docs/guides/operator-plugins.md).

A runnable example lives at
[`examples/plugin-operator/`](examples/plugin-operator/).

## Adding a New Docker Env

Do this only when a new operator's dependencies genuinely cannot share a
pip resolver with any existing env (e.g., it pins a different major
version of torch or numpy). One-off dep conflicts inside an existing env
are cheaper to resolve by pinning than by creating a new env.

The canonical example is `:fish-speech`: fish-speech 2.0 pins `torch==2.8`,
incompatible with the `:tts` env's `torch==2.4` (shared with ChatTTS,
CosyVoice, kokoro). Isolating it keeps the other three TTS engines on a
validated stack.

Steps (mirrors the `:diarize` env split in the same PR):

1. **Name the env and register it**:
   - Add the name to `KNOWN_ENVS` in
     [`voxkitchen/runtime/env_resolver.py`](voxkitchen/runtime/env_resolver.py).
   - Map each of its extras in `EXTRA_TO_ENV` to the new env name.

2. **Declare the expected operator set**:
   - Add `EXPECTED_OPERATORS["<env>"] = EXPECTED_OPERATORS["core"] | {...}`
     in [`voxkitchen/cli/doctor.py`](voxkitchen/cli/doctor.py).
   - Include the new env in `_available_envs()` canonical order.

3. **Add a per-env constraints file**:
   - `docker/constraints/<env>.txt` — pin whatever shared deps (torch,
     numpy, huggingface_hub, etc.) are specific to this env.

4. **Extend warmup**:
   - Add a `run_<env>()` function to `scripts/warmup_models.py` that
     pre-downloads the env's models at image build time.
   - Add the env name to the `--group` argparse choices.

5. **Add a Dockerfile stage + target**:
   ```dockerfile
   FROM core-env AS <env>-env
   RUN uv venv /opt/voxkitchen/envs/<env> --python 3.11
   RUN uv pip install --python /opt/voxkitchen/envs/<env>/bin/python \
       -c /app/docker/constraints/<env>.txt \
       -e ".[...your extras...]"
   RUN uv pip install --python /opt/voxkitchen/envs/<env>/bin/python -e . --no-deps
   RUN /opt/voxkitchen/envs/<env>/bin/python scripts/warmup_models.py --group <env>
   RUN /opt/voxkitchen/envs/<env>/bin/python -m voxkitchen.runtime.dump_schemas \
       --env <env> --out /opt/voxkitchen/schemas_<env>.json

   FROM <env>-env AS <env>
   # merge core + <env> schemas, doctor smoke test, chmod, ENTRYPOINT
   ```
   Also extend the `latest` stage to `COPY --from=<env>-env` the new env
   subtree + its schema + warmup status.

6. **Update docs**:
   - [`docs/architecture/multi-env.md`](docs/architecture/multi-env.md) —
     add the new env to the layout diagram.
   - [`docs/docker-build.md`](docs/docker-build.md) — add the new tag to
     the target matrix.
   - `README.md` — add the new tag to the short table.

7. **Verify with integration tests**:
   - The existing `test_cross_env_dispatch_happy_path` in
     `tests/integration/test_multi_env_dispatch_e2e.py` uses a synthetic
     sandbox env to verify the dispatch mechanism — no need to add env-
     specific integration tests unless the env has unique dispatch
     semantics.

## Adding a dataset to the catalog

1. Append an entry to `voxkitchen/datasets/catalog.yaml` (minimum fields:
   `id, name, task, languages, license, summary, homepage, recommendation`).
   Use `recipe: <name>` only if VoxKitchen has a download recipe for it;
   otherwise add `notes:` explaining how to obtain it. Optionally set
   `recommended_pipeline:` to an existing pipeline under `examples/pipelines/`
   or `voxkitchen/templates/pipelines/`.
2. Regenerate the docs: `python -m voxkitchen.datasets.catalog_gen`
3. Commit `catalog.yaml` and the generated `docs/datasets/` together.

Do not fabricate license or size figures — cite the official source or leave
them blank. The catalog only links and describes datasets; it never hosts or
redistributes data, and users are responsible for license compliance.

## Adding a New Recipe

1. Create `voxkitchen/ingest/recipes/<name>.py`
2. Subclass `Recipe`, implement `prepare()`, optionally override `download()`
3. Call `register_recipe(YourRecipe())` at module bottom
4. Import in `voxkitchen/ingest/recipes/__init__.py`
5. Add tests in `tests/unit/ingest/recipes/`
6. Add a matching `recipe:` entry to `voxkitchen/datasets/catalog.yaml` and
   regenerate the catalog docs (`python -m voxkitchen.datasets.catalog_gen`).

## Pull Request Process

1. Fork the repo and create a branch from `main`
2. Make your changes with tests
3. Ensure local CI passes: `scripts/check-ci.sh`
4. Commit with conventional commit messages
5. Open a PR with a clear description of what and why

## Questions?

Open an issue at https://github.com/XqFeng-Josie/VoxKitchen/issues.
