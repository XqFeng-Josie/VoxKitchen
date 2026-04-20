# Contributing to VoxKitchen

Thank you for your interest in contributing! This guide covers everything you need to get started.

## Development Setup

```bash
# Clone and install in development mode
git clone https://github.com/XqFeng-Josie/VoxKitchen.git
cd VoxKitchen
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

`[dev]` installs test tooling plus the **core-cluster** extras (the same
set CI validates, mirroring the `:slim` Docker image). To work on an
operator whose extras live outside that cluster (ASR, diarize, TTS),
add those extras on top:

```bash
pip install -e ".[dev,asr,funasr]"      # ASR cluster
pip install -e ".[dev,diarize]"          # Diarization
pip install -e ".[dev,tts-kokoro]"       # TTS
```

`pip install -e ".[all]"` is intentionally unsupported — it crosses
dep clusters and fails at the resolver. Pick one cluster or use
`vkit docker` for cross-cluster pipelines.

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

3. **Register** in `voxkitchen/operators/__init__.py`:

   ```python
   from voxkitchen.operators.<category> import <module> as _alias  # noqa: F401
   ```

   For optional dependencies, wrap in `try/except ImportError`.

4. **Map the operator's `required_extras` to a Docker env**. If the
   operator declares any `required_extras = [...]`, every listed extras
   group must appear in `EXTRA_TO_ENV` in
   [`voxkitchen/runtime/env_resolver.py`](voxkitchen/runtime/env_resolver.py).
   The runner consults this table (plus `op_env_map.json` inside the
   Docker image) to route stages to the right env. If two extras in an
   operator map to different envs, the runner refuses to dispatch — split
   the extras or pick one env.

5. **Add to expected set** in
   [`voxkitchen/cli/doctor.py`](voxkitchen/cli/doctor.py)'s
   `EXPECTED_OPERATORS[<env>]` — this is what the `vkit doctor --expect <env>`
   smoke test at the end of each Docker build stage checks. If your
   operator is in the image, it must be in the expected set, or the
   build-time smoke test will flag it as "extra" (harmless but noisy).

6. **Write tests** in `tests/unit/operators/<category>/test_<name>.py`.

7. **Create an example** in `examples/pipelines/` (optional but appreciated).

8. **Regenerate operator docs**:

   ```bash
   python scripts/gen_operator_docs.py -o docs/reference/operators.md
   ```

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
