# Multi-env architecture

Status: **design, in progress**.
Supersedes the interim 3-image Docker approach (`docker/Dockerfile.{core,asr,tts}`).

## Why

A single Python environment cannot host all 51 operators. Concrete conflicts verified
against wheel metadata:

- `pyannote-audio>=4.0` ⇒ `torch>=2.8` and `numpy>=2.1`
- `funasr + modelscope` ⇒ effectively capped at the `transformers<5` / `numpy<2` stack
- `inaSpeechSegmenter` ⇒ `tensorflow[and-cuda]` + `onnxruntime-gpu`
- `ChatTTS` / `CosyVoice` / `Fish-Speech` have transitive pins that fight with the ASR stack

Any "one image, all operators" approach either (a) silently downgrades deps with
`pip install || echo WARN` — producing an image that *looks* healthy but breaks
at runtime — or (b) refuses to resolve at all.

## Core idea

VoxKitchen already checkpoints each stage to disk as `cuts.jsonl.gz`. Stages do not
share memory. **Therefore each stage can run in its own Python interpreter** with its
own installed packages, provided we route inputs and outputs via the existing disk
checkpoints.

## Shape

One Docker image. Four Python environments inside it, created with `uv`. A thin
dispatch layer in the pipeline runner decides which env each stage runs in.

```
/opt/voxkitchen/
  envs/
    core/         # CPU torch 2.4;  audio/segment/quality/pack/pitch/dnsmos/classify/enhance/codec/speaker/viz
    asr/          # GPU torch 2.4;  asr/whisper/funasr/align (no diarize)
    diarize/      # GPU torch 2.4;  pyannote 3.x only — separate from asr so :diarize can ship small
    tts/          # GPU torch 2.4;  tts-kokoro/chattts/cosyvoice
    fish-speech/  # GPU torch 2.8;  tts-fish-speech (upstream pins torch 2.8 / numpy 2.1)
  op_schemas.json   # every op's pydantic schema, keyed by op name
  op_env_map.json   # op name → env name
  model_cache/      # shared HF / torch / modelscope cache across envs
```

Why fish-speech is its own env: its `torch==2.8.0` pin is incompatible with
ChatTTS / CosyVoice / kokoro on torch 2.4. Forcing them onto torch 2.8 would
expand the risk surface from "one broken TTS" to "all TTS broken". Isolating
fish-speech costs ~5 GB image size and one extra venv to warm; in return
every other env stays on its validated stack.

> **Current status**: fish-speech 2.0 reshaped its Python API
> (`fish_speech.inference.TTSInference` → `fish_speech.inference_engine.TTSInferenceEngine`
> with a queue-based Llama generator + DAC decoder wiring). The
> `tts_fish_speech` operator still targets the 1.x shape and is therefore
> temporarily **excluded from `EXPECTED_OPERATORS["fish-speech"]`** so the
> `latest` build passes. The env is still built, the model is still
> pre-downloaded; a follow-up PR will rewrite the operator against the
> 2.0 API. This is an operator-level lag, not an architecture issue.

The `vkit` command on `$PATH` is a shim that routes into `envs/core/bin/python`. The
core env is the parent: it loads the pipeline YAML, decides per-stage envs, and
dispatches.

## Data flow

```
  vkit run pipeline.yaml
        │
        ▼  (core env, parent process)
  load spec → validate args against op_schemas.json
        │
        ▼
  for each stage:
      target_env = resolve_env(stage.op)
      if target_env == "core":
          run in-process (existing CpuPoolExecutor / GpuPoolExecutor)
      else:
          write input cuts.jsonl.gz (already present from prior stage)
          spawn: /opt/voxkitchen/envs/<target_env>/bin/python \
                     -m voxkitchen.runtime.stage_runner \
                     --op <name> --config-json <json> \
                     --input <prev/cuts.jsonl.gz> \
                     --output <this/cuts.jsonl.gz> \
                     --ctx-json <ctx>
          wait, check exit code, surface stderr on failure
  // next stage reads <this/cuts.jsonl.gz> from disk — same as today
```

The disk-based stage boundary already exists; subprocess dispatch is additive.

## Components

### `voxkitchen/runtime/env_resolver.py`

Resolves an operator name to an env name. Does NOT import the operator class — only
reads two small JSON files so the parent (core env) can decide dispatch for operators
it cannot import.

```python
def resolve_env(op_name: str) -> str: ...
def current_env() -> str: ...  # reads $VKIT_ENV, set by each venv's bin/activate
```

Lookup order:
1. `$VKIT_OP_ENV_MAP` (override for tests) →
2. `/opt/voxkitchen/op_env_map.json` (docker) →
3. In-process fallback: walk registered operators, derive from `required_extras`

The fallback matters for local `pip install -e .` dev — there's only one env,
everything maps to it, and the subprocess path is never taken.

### `voxkitchen/runtime/stage_runner.py`

Subprocess entry point. Self-contained, runs inside any env that has the operator's
deps installed.

```
python -m voxkitchen.runtime.stage_runner \
    --op <name> \
    --config-json <json-string> \
    --input  <cuts.jsonl.gz> \
    --output <cuts.jsonl.gz> \
    --ctx-json <json-string>
```

Behavior:
1. Import `voxkitchen.operators` (populates registry with whatever this env can load)
2. Read input cuts with `CutSet.from_jsonl_gz`
3. Resolve op, validate config, pick executor (CPU pool or GPU pool)
4. Run, write output, write `_errors.jsonl` and `_stats.json`
5. Exit 0 on success, non-zero on unrecoverable failure

The parent treats this process as a black box: same pipeline, just remote.

### `voxkitchen/runtime/dump_schemas.py`

Run once per env at image build time. Walks the registered operators and emits a
JSON object `{op_name: {schema, required_extras, device}}`. Output is merged across
envs in the Dockerfile.

### `voxkitchen/runtime/merge_schemas.py`

Combines per-env dumps into the final `op_schemas.json` and `op_env_map.json`.
Detects when the same operator is registered in multiple envs (which should not
happen after this refactor — a symptom of an incorrect `EXTRA_TO_ENV` mapping).

### `voxkitchen/pipeline/executor.py` (changed)

Add a third executor:

```python
class SubprocessStageExecutor:
    """Run one stage in a different Python env via subprocess.

    Inputs and outputs cross the env boundary as jsonl.gz files on disk.
    """
    def __init__(self, target_env: str) -> None: ...

    def run(self, op_cls, config, cuts, ctx) -> CutSet:
        # op_cls is not importable in this env — it's a placeholder. We only
        # need the name. config is serialized to JSON.
        # cuts must already be on disk at ctx.input_manifest.
        ...
```

`op_cls` is awkward here — the runner currently passes the operator class. When the
parent env cannot import the class, the runner should instead pass the op name and
let `SubprocessStageExecutor` handle it. This is a small executor protocol extension.

### `voxkitchen/pipeline/runner.py` (changed)

Two changes:

1. Replace `op_cls = get_operator(stage.op)` with a try/except that falls back to
   reading `op_schemas.json` when the operator is not importable in the parent env.
2. Add env-aware executor selection:

   ```python
   target_env = resolve_env(stage.op)
   if target_env == current_env():
       executor = _make_in_process_executor(op_cls.device, ctx)
   else:
       executor = SubprocessStageExecutor(target_env)
   ```

3. `stage_runner` inside the subprocess re-runs the full in-process executor path,
   so CPU sharding and GPU pinning still work.

## Environment construction

### Tool: uv

- ~10× faster installs than pip
- Lockfile support (`uv.lock`) — committed to repo for reproducibility
- Supports PyTorch CUDA wheels correctly
- Astral Inc. maintained, stable

### Per-env constraints

`docker/constraints/{core,asr,tts}.txt` pin shared deps. Constraints are **stricter per
env** than before because we no longer need one set to cover all operators — each env
can pin to exactly what its extras agree on:

- `core`: `torch==2.4.1+cpu`, `numpy<2.0`, `transformers>=4.40,<5.0`
- `asr`: `torch==2.4.1`, `numpy<2.0`, `transformers>=4.40,<5.0`, `huggingface_hub>=0.23,<1.0`, `ctranslate2>=4.4,<5.0`
- `tts`: `torch==2.4.1`, `numpy<2.0`, `transformers>=4.40,<5.0` (modelscope may add more pins — TBD on first build)

### Lockfiles

Each env maintains its own `uv.lock`:

- `docker/uv.lock.core`
- `docker/uv.lock.asr`
- `docker/uv.lock.tts`

`docker build` uses `--frozen` to fail the build if lockfiles drift from `pyproject.toml`.

CI regenerates lockfiles on a schedule (weekly) and opens a PR — keeps them fresh
without surprising builds.

## Dockerfile

One `Dockerfile` with six BuildKit targets:

- `target=slim`:        core env only, torch-cpu, ~13 GB
- `target=asr`:         core + asr env,          ~48 GB
- `target=diarize`:     core + diarize env (pyannote only), ~32 GB
- `target=tts`:         core + tts env,          ~44 GB
- `target=fish-speech`: core + fish-speech env (isolated torch 2.8), ~38 GB
- `target=latest`:      all five envs merged,    ~103 GB

```
FROM pytorch/pytorch:2.4.1-cuda12.4-cudnn9-runtime AS base
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential python3-dev ffmpeg libsndfile1 espeak-ng sox \
    && rm -rf /var/lib/apt/lists/*

FROM base AS core-env
RUN uv venv /opt/voxkitchen/envs/core --python 3.11 && \
    uv pip install --python /opt/voxkitchen/envs/core/bin/python \
        -c docker/constraints/core.txt \
        -e ".[audio,segment,quality,pack,pitch,dnsmos,classify,enhance,codec,speaker,viz]"
# warmup + schema dump for core
RUN /opt/voxkitchen/envs/core/bin/python scripts/warmup_models.py --group core
RUN /opt/voxkitchen/envs/core/bin/python -m voxkitchen.runtime.dump_schemas \
        --env core --out /tmp/schemas_core.json

FROM core-env AS slim
# Just the vkit shim → core env, plus a minimal op_env_map.json & op_schemas.json
RUN /opt/voxkitchen/envs/core/bin/python -m voxkitchen.runtime.merge_schemas \
        /tmp/schemas_core.json --out /opt/voxkitchen/op_schemas.json
COPY docker/vkit-shim.sh /usr/local/bin/vkit
ENTRYPOINT ["vkit"]

FROM core-env AS asr-env
RUN uv venv /opt/voxkitchen/envs/asr --python 3.11 && \
    uv pip install --python /opt/voxkitchen/envs/asr/bin/python \
        -c docker/constraints/asr.txt \
        -e ".[audio,segment,quality,pack,pitch,dnsmos,classify,enhance,codec,speaker,viz,asr,whisper,funasr,align,diarize]"
RUN /opt/voxkitchen/envs/asr/bin/python scripts/warmup_models.py --group asr
RUN /opt/voxkitchen/envs/asr/bin/python -m voxkitchen.runtime.dump_schemas \
        --env asr --out /tmp/schemas_asr.json

FROM asr-env AS tts-env
RUN uv venv /opt/voxkitchen/envs/tts --python 3.11 && \
    uv pip install --python /opt/voxkitchen/envs/tts/bin/python \
        -c docker/constraints/tts.txt \
        -e ".[audio,segment,quality,pack,pitch,dnsmos,classify,enhance,codec,speaker,viz,tts-kokoro,tts-chattts,tts-cosyvoice,tts-fish-speech]"
RUN /opt/voxkitchen/envs/tts/bin/python scripts/warmup_models.py --group tts
RUN /opt/voxkitchen/envs/tts/bin/python -m voxkitchen.runtime.dump_schemas \
        --env tts --out /tmp/schemas_tts.json

FROM tts-env AS latest
RUN /opt/voxkitchen/envs/core/bin/python -m voxkitchen.runtime.merge_schemas \
        /tmp/schemas_core.json /tmp/schemas_asr.json /tmp/schemas_tts.json \
        --out /opt/voxkitchen/op_schemas.json
RUN /opt/voxkitchen/envs/core/bin/vkit doctor --expect-all
COPY docker/vkit-shim.sh /usr/local/bin/vkit
ENTRYPOINT ["vkit"]
```

Build commands:

```bash
docker build --target latest -t voxkitchen:latest .
docker build --target slim   -t voxkitchen:slim   .
```

## CLI surface

No change to existing commands. One new command:

- `vkit doctor [--env <name>]` — inspect a specific env or all envs (default)
- `vkit env list` — short listing of installed envs and their operator counts
- `vkit env exec <env> <cmd...>` — debugging helper: run a command inside an env

Validate stays the same but uses `op_schemas.json` instead of importing operators.

## Risks and mitigations

| Risk | Mitigation |
|---|---|
| Subprocess spawn latency (~0.5s × N stages) | Only crossed when the op's env ≠ current. Core-only pipelines never spawn. |
| Error propagation across subprocess boundary | `stage_runner` writes `_errors.jsonl` and `_stats.json` as today. Exit code + stderr tail passes up through `SubprocessStageExecutor.run`. |
| Cross-env pickle mismatch | We do NOT pickle across envs. Everything crosses via jsonl.gz on disk (Pydantic v2 serialization is env-agnostic). |
| Operator registration differs per env | Each env runs its own `dump_schemas.py`. Parent trusts the merged `op_env_map.json` — it doesn't need to import. |
| User adds a custom operator via plugin entry_points | The plugin's env must be declared. We extend `op_env_map.json` to accept operator→env overrides from `$VKIT_EXTRA_OP_ENV_MAP`. |
| Lockfile drift | CI regenerates weekly + `--frozen` in Dockerfile. |
| GPU memory not released between stages | Subprocess exits between stages, so this is automatic. Better than current behavior within a single process. |
| Resume across env switches | Checkpoint format is unchanged (`cuts.jsonl.gz` + `_SUCCESS` marker). Resume reads the file; it doesn't know or care which env produced it. |

## Phased rollout

1. **P1 runtime modules** — `env_resolver`, `stage_runner`, `dump_schemas`, `merge_schemas`. Self-contained, unit-testable, no executor changes yet.
2. **P2 executor wiring** — `SubprocessStageExecutor`, runner dispatch. Feature-flagged: if `VKIT_MULTI_ENV=0`, use current in-process path.
3. **P3 schema-driven validate** — `vkit validate` stops importing operators; reads `op_schemas.json`.
4. **P4 Dockerfile + uv + lockfiles** — unified Dockerfile, BuildKit targets for `latest` / `slim`.
5. **P5 doctor** — cross-env aggregation; migrate existing `doctor.py`.
6. **P6 integration** — four pipelines end-to-end (core-only, asr-only, tts-only, mixed).
7. **P7 cleanup + docs** — delete the interim 3-Dockerfile setup, rewrite README.

## What does NOT change

- Operator authoring contract: subclass `Operator`, declare `name`, `config_cls`, `required_extras`.
- YAML surface: `vkit run pipeline.yaml` still works identically.
- Checkpoint / resume semantics.
- GC / trash behavior.
- Tests: existing operator tests run in the env that has their deps. Parent-env smoke tests run in `core`.
