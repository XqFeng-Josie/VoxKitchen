# VoxKitchen v0.1 Design Spec

- **Project codename**: SpeechDatasetHub (internal)
- **Public name**: VoxKitchen
- **Python package**: `voxkitchen`
- **CLI command**: `vkit`
- **License**: Apache 2.0
- **Python**: 3.10+
- **Date**: 2026-04-11
- **Scope**: v0.1 MVP

## 1. Overview and Scope

### 1.1 Positioning

VoxKitchen is a researcher-friendly, declarative speech data processing toolkit with a unified data protocol. Its purpose is to make the path from raw audio to training-ready datasets reproducible, resumable, and inspectable.

The metaphor is a **kitchen**: `pipeline.yaml` is a recipe, operators are cooking steps, ingest recipes are ingredient prep, `pack` is plating, and `vkit viz` is tasting.

### 1.2 Target Users

- **Primary**: speech researchers working on ASR, TTS, speaker recognition, and speech LLMs.
- **Secondary**: data engineers and small teams organizing internal speech corpora.

### 1.3 In Scope for v0.1

1. A unified Lhotse-style data protocol (Recording / Supervision / Cut), implemented in-house without a hard dependency on Lhotse.
2. Project skeleton: Python package, CLI, configuration system, plugin entry points.
3. Pipeline engine: YAML declaration, stage-parallel multi-GPU runner, aggressive GC by default.
4. Built-in operators covering 5 categories: basic processing, segmentation, auto-labeling, quality filtering, and packaging.
5. Ingestion: directory scan, manifest import, and 2–3 dataset recipes (LibriSpeech, CommonVoice, AISHELL-1).
6. Visualization triad: Rich CLI, self-contained HTML report, and a transient local Gradio panel.

### 1.4 Non-Goals (v0.1)

See section 10 for the full list. Summary: no catalog UI, no crawlers, no manual annotation UI, no distributed execution, no training, no long-running server, no databases, no Windows/macOS CI, no Docker image, no multi-language docs.

### 1.5 Success Criterion

A researcher points at a LibriSpeech directory, writes ~20 lines of YAML, runs one `vkit run pipeline.yaml`, and after a reasonable time produces a HuggingFace Datasets directory containing ASR transcripts, speaker IDs, quality scores, and full provenance — plus a self-contained `report.html`. The pipeline supports `--resume-from <stage>` and bounds disk usage via aggressive GC.

---

## 2. Repository Skeleton

```
voxkitchen/
├── pyproject.toml
├── README.md
├── LICENSE                        # Apache 2.0
├── CHANGELOG.md
├── docs/
│   ├── index.md
│   ├── getting-started.md
│   ├── concepts/
│   ├── recipes/
│   ├── operators/
│   └── contributing.md
├── src/voxkitchen/
│   ├── __init__.py
│   ├── __main__.py
│   │
│   ├── schema/
│   │   ├── __init__.py
│   │   ├── recording.py
│   │   ├── supervision.py
│   │   ├── cut.py
│   │   ├── cutset.py
│   │   ├── provenance.py
│   │   └── io.py
│   │
│   ├── operators/
│   │   ├── __init__.py
│   │   ├── base.py
│   │   ├── registry.py
│   │   ├── basic/
│   │   ├── segment/
│   │   ├── annotate/
│   │   ├── quality/
│   │   └── pack/
│   │
│   ├── pipeline/
│   │   ├── __init__.py
│   │   ├── spec.py
│   │   ├── loader.py
│   │   ├── runner.py
│   │   ├── executor.py
│   │   ├── checkpoint.py
│   │   ├── gc.py
│   │   └── context.py
│   │
│   ├── ingest/
│   │   ├── __init__.py
│   │   ├── base.py
│   │   ├── dir_scan.py
│   │   ├── manifest_import.py
│   │   └── recipes/
│   │       ├── __init__.py
│   │       ├── base.py
│   │       ├── librispeech.py
│   │       ├── commonvoice.py
│   │       └── aishell.py
│   │
│   ├── viz/
│   │   ├── __init__.py
│   │   ├── cli.py
│   │   ├── report/
│   │   │   ├── generator.py
│   │   │   ├── templates/
│   │   │   └── assets/
│   │   └── panel/
│   │       └── app.py
│   │
│   ├── cli/
│   │   ├── __init__.py
│   │   ├── main.py
│   │   ├── run.py
│   │   ├── ingest.py
│   │   ├── inspect.py
│   │   ├── viz.py
│   │   ├── validate.py
│   │   └── init.py
│   │
│   ├── config/
│   │   ├── __init__.py
│   │   ├── defaults.py
│   │   └── settings.py
│   │
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── audio.py
│   │   ├── hashing.py
│   │   ├── logging.py
│   │   ├── progress.py
│   │   └── gpu.py
│   │
│   └── plugins/
│       ├── __init__.py
│       └── discovery.py
│
├── tests/
│   ├── unit/
│   ├── integration/
│   ├── gpu/
│   ├── fixtures/
│   └── conftest.py
│
├── examples/
│   ├── pipelines/
│   │   ├── minimal.yaml
│   │   ├── librispeech-asr.yaml
│   │   └── cv-diar-pack.yaml
│   └── notebooks/
│
├── recipes/                       # community recipe contributions
│   └── README.md
│
└── .github/
    ├── workflows/
    │   ├── ci.yml
    │   ├── docs.yml
    │   └── release.yml
    └── ISSUE_TEMPLATE/
```

### 2.1 Layout Decisions

- **`src/` layout** to avoid accidentally importing uninstalled code during development.
- **Strict module boundaries**: `schema` depends on nothing internal; `operators` depends on `schema + utils`; `pipeline` depends on `schema + operators + utils`; `cli` depends on all. No circular imports.
- **One file per operator** under `operators/<category>/`. Adding a new operator requires creating one new file, never editing existing code.
- **`viz/report/assets/` is fully self-contained**. No CDN references. `report.html` works on offline research machines.
- **Two recipe directories with different roles**:
  - `src/voxkitchen/ingest/recipes/`: v0.1 built-in ingest recipe code.
  - Top-level `recipes/`: example pipeline YAMLs for complete dataset processing flows.
- **Explicitly absent**: no `models/` (we do not host weights), no `api/` (no REST), no `db/` (no database).

---

## 3. Data Protocol

### 3.1 Design Principles

1. **Immutable records, chained evolution**: every stage reads a `CutSet` and produces a new `CutSet`. The original is not modified. History is tracked via Provenance.
2. **Flat JSONL.gz serialization**: one JSON per line, gzipped. Inspectable with `zcat | head` and `zcat | jq`. Friendly to `rsync`, `diff`, and streaming writers.
3. **Every field is optional except the essentials**: before ASR runs, `supervision.text` is `None`, not an error.
4. **User extensions go in `custom: dict[str, Any]`**. No arbitrary top-level fields — prevents schema fragmentation while keeping extensibility.
5. **Strong typing with Pydantic v2**. All dataclasses are Pydantic v2 BaseModel subclasses.

### 3.2 Core Types

```python
# schema/recording.py

class AudioSource(BaseModel):
    type: Literal["file", "url", "command"]
    channels: list[int]
    source: str

class Recording(BaseModel):
    id: str
    sources: list[AudioSource]
    sampling_rate: int
    num_samples: int
    duration: float
    num_channels: int
    checksum: str | None = None
    custom: dict[str, Any] = {}
```

```python
# schema/supervision.py

class Supervision(BaseModel):
    id: str
    recording_id: str
    start: float
    duration: float
    channel: int | list[int] | None = None
    text: str | None = None
    language: str | None = None
    speaker: str | None = None
    gender: Literal["m", "f", "o"] | None = None
    age_range: str | None = None
    custom: dict[str, Any] = {}
```

```python
# schema/cut.py

class Cut(BaseModel):
    id: str
    recording_id: str
    start: float
    duration: float
    channel: int | list[int] | None = None
    supervisions: list[Supervision]
    metrics: dict[str, float] = {}
    provenance: Provenance
    custom: dict[str, Any] = {}
```

```python
# schema/provenance.py

class Provenance(BaseModel):
    source_cut_id: str | None
    generated_by: str           # e.g. "silero_vad@0.4.1"
    stage_name: str             # e.g. "02_vad"
    created_at: datetime
    pipeline_run_id: str
```

### 3.3 CutSet

`CutSet` wraps a sequence of Cuts with lazy I/O and functional operations. It is not a Pydantic model; it is a thin wrapper over a JSONL.gz stream.

```python
class CutSet:
    @classmethod
    def from_jsonl_gz(cls, path: Path) -> "CutSet": ...
    def to_jsonl_gz(self, path: Path) -> None: ...

    def split(self, n: int) -> list["CutSet"]: ...   # for GPU sharding
    def filter(self, pred) -> "CutSet": ...
    def map(self, fn) -> "CutSet": ...
    def __iter__(self) -> Iterator[Cut]: ...
    def __len__(self) -> int: ...
```

### 3.4 Serialization Format

Each stage writes a `cuts.jsonl.gz` file. The first line is a header record for schema versioning:

```jsonl
{"__type__": "voxkitchen.header", "schema_version": "0.1", "created_at": "2026-04-11T10:30:00Z", "pipeline_run_id": "run-a1b2c3", "stage_name": "02_vad"}
{"__type__": "cut", "id": "...", "recording_id": "...", "start": 0.0, "duration": 3.45, ...}
{"__type__": "cut", ...}
```

### 3.5 Directory Layout for a Run

```
work_dir/
├── recordings.jsonl.gz        # all Recordings, shared across stages
├── run.yaml                   # expanded snapshot of the pipeline YAML
├── 00_ingest/
│   ├── cuts.jsonl.gz
│   └── _SUCCESS
├── 01_format_convert/
│   ├── cuts.jsonl.gz
│   ├── derived/               # new wav files, GC'd per plan
│   └── _SUCCESS
├── 02_vad/
│   ├── cuts.jsonl.gz
│   └── _SUCCESS
├── ...
└── report.html
```

Only stages that materialize new audio (format convert, resample, pack) write to `derived/`. Other operators only update supervisions/metrics and produce tiny manifests.

New Recordings created by materialization stages are **appended** to `recordings.jsonl.gz` with fresh IDs. Original Recordings are never replaced.

### 3.6 Provenance Chain

Every Cut's `provenance.source_cut_id` points at its parent Cut in the previous stage. Across stages this forms a DAG, traversable via `vkit inspect trace <cut_id>`.

Stage-level parameters (YAML args) are not duplicated per Cut — they live in `work_dir/run.yaml`, linked via `pipeline_run_id`.

### 3.7 Schema Versioning

- `schema_version` lives in every `cuts.jsonl.gz` header.
- Version mismatches trigger migration functions in `schema/migrations/` (none needed for v0.1).
- v0.1 starts at `"0.1"`.

---

## 4. Operator Abstraction

### 4.1 What an Operator Is

An Operator is a pure transformation `CutSet → CutSet`. All pipeline behavior composes from operators. Each operator defines:

- A unique `name` (referenced in YAML).
- A Pydantic `OperatorConfig` subclass declaring all parameters with defaults.
- A `process()` implementation.
- Optional `setup()` / `teardown()` for model load/release.
- A `device` declaration (`"cpu"` or `"gpu"`).
- A `produces_audio` declaration (for GC tracking).
- A `reads_audio_bytes` declaration (default `True`; used by GC planning).
- A `required_extras` declaration for dependency hinting.

### 4.2 Base Class

```python
# operators/base.py

class OperatorConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

class Operator(ABC):
    name: ClassVar[str]
    config_cls: ClassVar[type[OperatorConfig]]
    device: ClassVar[Literal["cpu", "gpu"]] = "cpu"
    produces_audio: ClassVar[bool] = False
    reads_audio_bytes: ClassVar[bool] = True
    required_extras: ClassVar[list[str]] = []

    def __init__(self, config: OperatorConfig, ctx: RunContext):
        self.config = config
        self.ctx = ctx

    def setup(self) -> None: ...

    @abstractmethod
    def process(self, cuts: CutSet) -> CutSet: ...

    def teardown(self) -> None: ...
```

`process()` takes a `CutSet` shard, not a single Cut — this allows batched execution (e.g. ASR batch_size=8).

### 4.3 Execution Modes

- **CpuOperator** (`device="cpu"`): dispatched to `CpuPoolExecutor`, which uses `multiprocessing.Pool` with `num_workers` = CLI `--num-workers` or YAML `num_cpu_workers`.
- **GpuOperator** (`device="gpu"`): dispatched to `GpuPoolExecutor`, which spawns N worker processes (N = `--num-gpus`), each binding one GPU via `CUDA_VISIBLE_DEVICES` before importing torch. In `setup()` the operator loads a private model copy.

Operator authors only write `process(cuts) -> cuts`. Sharding, process management, and GPU binding are handled by the executor.

### 4.4 Registration Mechanism

```python
# operators/registry.py

_REGISTRY: dict[str, type[Operator]] = {}

def register_operator(op_cls: type[Operator]) -> type[Operator]:
    if op_cls.name in _REGISTRY:
        raise ValueError(f"Operator '{op_cls.name}' already registered")
    _REGISTRY[op_cls.name] = op_cls
    return op_cls

def get_operator(name: str) -> type[Operator]:
    _ensure_plugins_loaded()
    if name not in _REGISTRY:
        raise UnknownOperatorError(name, suggestions=_fuzzy_match(name))
    return _REGISTRY[name]
```

Built-in operators are registered via imports from `operators/__init__.py`. Third-party operators register via `entry_points` (section 8). First access to the registry triggers a one-time plugin discovery pass.

### 4.5 Example: Silero VAD Operator

```python
# operators/segment/silero_vad.py

class SileroVadConfig(OperatorConfig):
    threshold: float = Field(0.5, ge=0.0, le=1.0)
    min_speech_duration_ms: int = 250
    min_silence_duration_ms: int = 100
    speech_pad_ms: int = 30
    model_version: str = "v4.0"

@register_operator
class SileroVadOperator(Operator):
    name = "silero_vad"
    config_cls = SileroVadConfig
    device = "gpu"
    produces_audio = False
    required_extras = []  # silero-vad is in core under Plan B

    def setup(self) -> None:
        import torch
        self.model, _ = torch.hub.load("snakers4/silero-vad", "silero_vad", trust_repo=True)
        self.model.to(self.ctx.device)

    def process(self, cuts: CutSet) -> CutSet:
        out: list[Cut] = []
        for cut in cuts:
            audio = load_audio_for_cut(cut, sr=16000)
            timestamps = self._detect_speech(audio)
            for i, (start, end) in enumerate(timestamps):
                out.append(self._make_child_cut(cut, start, end, idx=i))
        return CutSet(out)
```

### 4.6 Built-in Operator Catalog (v0.1)

In Plan B (section 9.2), most audio-processing dependencies are in core. Only `asr`, `diarize`, and `classify` exist as true optional extras; operators depending on in-core libraries have `required_extras=[]`.

| Category | `name` | Underlying lib | device | produces_audio | required_extras |
|---|---|---|---|---|---|
| **Basic** | `ffmpeg_convert` | ffmpeg-python | cpu | yes | — |
| | `resample` | torchaudio | cpu | yes | — |
| | `channel_merge` | torchaudio | cpu | yes | — |
| | `loudness_normalize` | pyloudnorm | cpu | yes | — |
| **Segment** | `silero_vad` | silero-vad | gpu | no | — |
| | `webrtc_vad` | webrtcvad | cpu | no | — |
| | `fixed_segment` | — (pure compute) | cpu | no | — |
| | `silence_split` | librosa | cpu | no | — |
| **Annotate** | `faster_whisper_asr` | faster-whisper | gpu | no | asr |
| | `whisperx_asr` | whisperx | gpu | no | asr |
| | `pyannote_diarize` | pyannote.audio | gpu | no | diarize |
| | `speechbrain_langid` | speechbrain | gpu | no | classify |
| | `speechbrain_gender` | speechbrain | gpu | no | classify |
| **Quality** | `snr_estimate` | torchaudio | cpu | no | — |
| | `duration_filter` | — | cpu | no | — |
| | `audio_fingerprint_dedup` | librosa + simhash | cpu | no | — |
| | `quality_score_filter` | — | cpu | no | — |
| **Pack** | `pack_huggingface` | datasets | cpu | yes | — |
| | `pack_webdataset` | webdataset | cpu | yes | — |
| | `pack_parquet` | pyarrow | cpu | yes | — |
| | `pack_kaldi` | — | cpu | yes | — |
| | `pack_manifest` | — | cpu | no | — |

All 22 operators ship in v0.1 with unit tests, at least one example YAML reference, and auto-generated reference docs.

---

## 5. Pipeline and Runner

### 5.1 Pipeline YAML Example

```yaml
version: "0.1"
name: librispeech-asr-pipeline
description: "Segment LibriSpeech, run ASR, pack to HF format"

work_dir: /data/work/${name}-${run_id}
num_gpus: 4
num_cpu_workers: 16
gc_mode: aggressive

ingest:
  source: recipe
  recipe: librispeech
  args:
    root: /data/librispeech
    subsets: [train-clean-100, dev-clean]

stages:
  - name: resample
    op: resample
    args: { target_sr: 16000, target_channels: 1 }

  - name: vad
    op: silero_vad
    args: { threshold: 0.5, min_speech_duration_ms: 250 }

  - name: asr
    op: faster_whisper_asr
    args: { model: large-v3, language: en, batch_size: 8, compute_type: float16 }

  - name: snr
    op: snr_estimate

  - name: filter_short
    op: duration_filter
    args: { min: 0.5, max: 30.0 }

  - name: filter_noisy
    op: quality_score_filter
    args: { conditions: ["metrics.snr > 10"] }

  - name: pack
    op: pack_huggingface
    args: { output_dir: /data/out/librispeech-asr, split_field: "custom.split" }
```

### 5.2 PipelineSpec

```python
# pipeline/spec.py

class StageSpec(BaseModel):
    name: str
    op: str
    args: dict[str, Any] = {}
    model_config = ConfigDict(extra="forbid")

class IngestSpec(BaseModel):
    source: Literal["dir", "manifest", "recipe"]
    args: dict[str, Any] = {}
    recipe: str | None = None

class PipelineSpec(BaseModel):
    version: str
    name: str
    description: str = ""
    work_dir: str
    num_gpus: int = 1
    num_cpu_workers: int | None = None
    gc_mode: Literal["aggressive", "keep"] = "aggressive"
    ingest: IngestSpec
    stages: list[StageSpec]

    @field_validator("stages")
    def unique_stage_names(cls, v): ...
```

`loader.py` handles `YAML → PipelineSpec`, expanding `${name}`, `${run_id}`, and `${env:VAR}` interpolations. `validate.py` performs deeper checks (operator existence, config instantiation, recipe existence, GC plan).

After validation, `work_dir/run.yaml` is written as a fully-expanded snapshot and becomes the authoritative configuration for provenance.

### 5.3 Runner Execution Flow

```
vkit run pipeline.yaml
  │
  ├─ load & validate        → PipelineSpec, run_id, work_dir
  ├─ detect resume          → scan work_dir for stages with _SUCCESS marker
  ├─ ingest stage           → IngestSource.run() → 00_ingest/cuts.jsonl.gz
  ├─ plan GC                → static analysis of stages to compute last_consumer map
  │
  ├─ for stage in stages:
  │    ├─ get_operator(stage.op)
  │    ├─ op_cfg = OperatorConfig.parse(stage.args)
  │    ├─ executor = GpuPoolExecutor if op.device == "gpu" else CpuPoolExecutor
  │    ├─ in_cuts = CutSet.from_jsonl_gz(prev_stage/cuts.jsonl.gz)
  │    ├─ out_cuts = executor.run(operator_cls, op_cfg, in_cuts, ctx)
  │    ├─ write stage_dir/cuts.jsonl.gz
  │    ├─ write stage_dir/_SUCCESS
  │    ├─ run GC             → move derived/ of newly-expired stages to trash
  │    └─ emit progress      → Rich live UI + structured log
  │
  └─ finalize                → write work_dir/report.html, empty trash
```

**Stage directory naming**: `work_dir/<NN>_<stage_name>/` where `NN` is the zero-padded 1-based stage index (`00_ingest`, `01_resample`, `02_vad`, …). The numeric prefix guarantees lexicographic ordering matches execution order when a user does `ls work_dir/`.

**Resume policy**: a stage is considered complete only if both `cuts.jsonl.gz` and `_SUCCESS` exist. If `cuts.jsonl.gz` exists without `_SUCCESS`, the stage is rerun.

All I/O is mediated by `RunContext`; no path hardcoding in operators.

### 5.4 Executor Implementations

```python
# pipeline/executor.py

class Executor(ABC):
    @abstractmethod
    def run(self, operator_cls: type[Operator], config: OperatorConfig,
            cuts: CutSet, ctx: RunContext) -> CutSet: ...

class CpuPoolExecutor(Executor):
    def __init__(self, num_workers: int): ...
    def run(self, operator_cls, config, cuts, ctx):
        shards = cuts.split(self.num_workers)
        with mp.get_context("spawn").Pool(
            self.num_workers,
            initializer=_cpu_init,
            initargs=(operator_cls, config, ctx),
        ) as pool:
            out_shards = list(pool.imap_unordered(_cpu_process, enumerate(shards)))
        return CutSet.merge(out_shards)

class GpuPoolExecutor(Executor):
    def __init__(self, num_gpus: int): ...
    def run(self, operator_cls, config, cuts, ctx):
        shards = cuts.split(self.num_gpus)
        processes = []
        for gpu_id, shard in enumerate(shards):
            p = mp.get_context("spawn").Process(
                target=_gpu_worker,
                args=(gpu_id, operator_cls, config, shard, ctx,
                      stage_shard_dir(ctx, gpu_id)),
            )
            p.start()
            processes.append(p)
        for p in processes:
            p.join()
            if p.exitcode != 0:
                raise StageFailedError(...)
        return CutSet.concat_from_disk(
            [stage_shard_dir(ctx, i) / "cuts.jsonl.gz" for i in range(self.num_gpus)]
        )
```

Each GPU worker writes its shard output to disk rather than returning it via IPC. This avoids serializing gigabytes of data across process boundaries and leaves partial shards for diagnostics when a worker crashes.

`CUDA_VISIBLE_DEVICES` binding happens inside `_gpu_worker` before torch is imported, so each worker always sees its GPU as `cuda:0`.

### 5.5 Garbage Collection

GC planning runs once at pipeline startup, producing a `GcPlan` that maps each `produces_audio=True` stage to its `last_consumer` — the last downstream stage with `reads_audio_bytes=True`. After a stage completes, any upstream stages whose `last_consumer` has just finished are eligible for GC.

**Trash safety net**: GC'd directories are renamed to `work_dir/derived_trash/<stage>/`, not immediately deleted. Trash is emptied only after the entire pipeline completes successfully. Crashed runs leave trash for diagnostics.

`gc_mode: keep` or `--keep-intermediates` disables GC entirely.

### 5.6 Error Handling

- **Single Cut error**: logged and skipped. Appended to `work_dir/<stage>/_errors.jsonl`. Pipeline continues.
- **Whole-shard / stage failure**: runner exits with an error. No automatic retry — operator errors are assumed to be real bugs, not transient faults.

Error record format:

```json
{"cut_id": "...", "operator": "faster_whisper_asr", "error": "...", "traceback": "..."}
```

Inspectable via `vkit inspect errors <work_dir>`.

### 5.7 Dry Run and Validation

- `vkit validate pipeline.yaml` — parse and schema-check only. No data touched. Seconds.
- `vkit run pipeline.yaml --dry-run` — full validation plus a printed summary of stages, operator params (expanded defaults), executor types, and the GC plan. No models loaded, no audio processed.

---

## 6. Command-Line Interface

### 6.1 Top-Level Commands

The CLI exposes **6 commands**:

```
vkit init        Scaffold a new pipeline project directory
vkit ingest      Build an initial CutSet from a data source
vkit validate    Parse and validate a pipeline YAML (no execution)
vkit run         Execute a pipeline
vkit inspect     View cuts, recordings, run progress, trace, or errors
vkit viz         Launch local Gradio panel to explore a CutSet
```

Built with **Typer**. Auto-generated `--help` for each command and subcommand.

### 6.2 Command Details

**`vkit init <dir>`** — interactive scaffolder. Generates `pipeline.yaml`, `README.md`, `.gitignore`. Prompts for ingest source and a few starter stages.

**`vkit ingest --source <dir|manifest|recipe> [args] --out <path>`** — standalone, not coupled to a pipeline. Produces a `cuts.jsonl.gz` and a `recordings.jsonl.gz`.

Examples:
```
vkit ingest --source dir /data/my_audio --out cuts.jsonl.gz
vkit ingest --source recipe librispeech --subset train-clean-100 --out cuts.jsonl.gz
```

**`vkit validate pipeline.yaml`** — parse, validate schema, check operator extras, run GC plan. Exits with status 1 on failure. Also reports any plugin load errors encountered during discovery.

**`vkit run pipeline.yaml [options]`** — the only command that processes data.

Options:
- `--num-gpus N` / `--num-workers N`
- `--work-dir PATH`
- `--resume-from STAGE`
- `--stop-at STAGE`
- `--dry-run`
- `--keep-intermediates`

**`vkit inspect <subcommand> [args]`** — explicit subcommands for discoverability:
```
vkit inspect cuts <path>
vkit inspect recordings <path>
vkit inspect run <work_dir>
vkit inspect trace <cut_id> --in <path>
vkit inspect errors <work_dir>
```

**`vkit viz <path>`** — starts a local Gradio server at `127.0.0.1:7860`. Not a long-running service — Ctrl+C exits.

### 6.3 Output and Logging

- **stdout**: Rich live progress + key events + final summary.
- **stderr**: warnings and error tracebacks.
- **`work_dir/logs/run.jsonl`**: structured JSON-per-line machine-readable log with stage timing, processed Cut counts, error counts, and GPU utilization samples.

When `sys.stdout.isatty()` is false (CI, redirects, `nohup`), the runner downgrades to plain text progress output (one line every 30 seconds), no Rich live display.

### 6.4 Exit Codes

- `0` success
- `1` user error (bad YAML, unknown operator, illegal arg)
- `2` runtime error (stage execution failed)
- `3` resource error (disk full, GPU unavailable, network download failed)
- `130` Ctrl+C

---

## 7. Visualization

Three independent modules. No shared state beyond the data protocol.

### 7.1 Rich CLI

**Dependency**: `rich`.

**At run time**: a Rich `Live` view with a top-level pipeline progress bar, per-stage shard progress (one line per GPU worker), and a rolling tail of recent log events. Non-TTY environments fall back to plain-text line output every 30 seconds.

**For `vkit inspect`**: Rich `Table` + `Tree` + `Panel`.

`inspect cuts` output:
1. Overview panel (count, total duration, mean, workload estimate).
2. Distribution table (duration histogram, language distribution, speaker distribution).
3. Metrics table (min / p50 / p95 / max for each `metrics` field).

`inspect trace` uses a Tree to show the Provenance chain:

```
Cut: librispeech-1089-134686-0001__vad_3__asr
├─ source_cut: librispeech-1089-134686-0001__vad_3
│  └─ source_cut: librispeech-1089-134686-0001
│     └─ source_cut: librispeech-1089-134686-0001 (raw)
└─ generated_by: faster_whisper_asr@large-v3
```

**Explicitly not doing**: Textual TUI apps, matplotlib-in-terminal rendering.

### 7.2 Static HTML Report

**Dependencies**: `jinja2`, `plotly` (offline mode).

**Trigger**: `vkit run` generates `work_dir/report.html` automatically on success. Can also be (re)generated with `vkit inspect run <work_dir> --html`.

**Layout**:

```
viz/report/
├── generator.py
├── templates/
│   ├── base.html.j2
│   ├── overview.html.j2
│   ├── stages.html.j2
│   └── samples.html.j2
└── assets/
    ├── style.css            # ~20 KB
    ├── plotly.min.js        # ~3 MB
    └── wavesurfer.min.js    # ~80 KB
```

**Report contents**:

1. Header: pipeline name, `run_id`, start/end time, total duration, embedded YAML (folded `<details>` block).
2. Overview: funnel chart of Cut counts across stages (Plotly).
3. Per-stage cards: duration, Cut counts, error counts, sampled GPU utilization (for GPU stages).
4. Data distributions: histograms and pie charts for duration, SNR, language, speaker counts (Plotly).
5. Sample playback: 10 randomly-sampled Cuts with `<audio>` tags, waveforms (wavesurfer.js), transcriptions, and provenance summary.
6. Provenance DAG: Plotly Sankey diagram of pipeline structure.

**Constraints**:
- **Self-contained**: no CDN references. Works offline.
- **Fully static**: double-clicking the file opens it. No server required.
- **Size cap**: ~5 MB total. Audio samples referenced by relative paths, not inlined.

### 7.3 Gradio Panel

**Dependency**: `gradio`.

**Trigger**: `vkit viz <cuts.jsonl.gz>` or `vkit viz <work_dir>`. Launches a transient local server at `127.0.0.1:7860`. Ctrl+C exits.

**Features**:

1. Paginated Cut table (default sort by duration). Filterable by language, speaker, duration range, SNR range.
2. Cut detail view: waveform, audio player, supervisions, provenance chain.
3. **Semantic filter expression**: a field accepting expressions like `duration > 5 and metrics.snr > 15`. Implemented as an AST whitelist evaluator that permits literals, attribute access, comparisons, and boolean combinators only. No `eval`. No function calls.
4. Export: selected subset → new `cuts.jsonl.gz` written to disk.

**Implementation scope**: a single `viz/panel/app.py` file, ~300 lines of Gradio. No user accounts, no persistent connections, no multi-file comparison.

### 7.4 How They Complement Each Other

| | Scenario | Lifetime |
|---|---|---|
| **Rich CLI** | ssh terminal, CI logs, run-time feedback | transient |
| **HTML report** | shareable artifact for collaborators, paper appendices | persistent file |
| **Gradio panel** | local exploration, filtering, ad-hoc cleanup | temporary service |

No overlap, no gaps.

---

## 8. Plugin Mechanism

### 8.1 Two Plugin Kinds

Both use Python packaging `entry_points` — the standard mechanism (pytest, airflow, jupyter).

**Operator plugins** via group `voxkitchen.operators`:

```toml
[project.entry-points."voxkitchen.operators"]
my_vad = "voxkitchen_my_vad.operator:MyVadOperator"
```

Users install with `pip install voxkitchen-my-vad` and reference `op: my_vad` in their pipeline YAML.

**Recipe plugins** via group `voxkitchen.recipes`:

```toml
[project.entry-points."voxkitchen.recipes"]
gigaspeech = "voxkitchen_gigaspeech.recipe:GigaSpeechRecipe"
```

### 8.2 Lazy Discovery

```python
# plugins/discovery.py

_loaded = False

def load_plugins() -> None:
    global _loaded
    if _loaded:
        return
    from importlib.metadata import entry_points
    for ep in entry_points(group="voxkitchen.operators"):
        try:
            register_operator(ep.load())
        except Exception as e:
            _log_plugin_error(ep.name, e)
    for ep in entry_points(group="voxkitchen.recipes"):
        try:
            register_recipe(ep.load())
        except Exception as e:
            _log_plugin_error(ep.name, e)
    _loaded = True
```

Discovery is triggered lazily on the first call to `get_operator()` or `list_operators()`. Commands that do not need the registry (e.g. `vkit --help`) never trigger plugin loading.

### 8.3 Plugin Error Isolation

A third-party plugin that fails to import must not break `vkit`:

- Exceptions from `ep.load()` are caught, logged, and the plugin is skipped.
- Plugins that failed to load are reported by `vkit validate`.
- When a user references an unknown operator, the error message suggests checking `vkit validate` for plugin load failures.

### 8.4 Deliberate Non-Features

- No plugin API version negotiation. Third parties track the `voxkitchen` version.
- No hot reload.
- No plugin sandbox or permission system. Python cannot provide this; pretending otherwise gives false security.
- No plugin marketplace or central index. PyPI is the index. The docs maintain a "known plugins" page.
- No plugin dependency conflict detection. `pip` handles this.

### 8.5 First-Party Recipes Are Hard-Coded

Built-in recipes (`librispeech`, `commonvoice`, `aishell`) register via direct imports in `ingest/recipes/__init__.py`, not via `entry_points`. Rationale:
- They ship with VoxKitchen and have no independent lifecycle.
- Zero discovery overhead.
- Clearer stack traces when something goes wrong.

Third-party recipes go through `entry_points`. Two paths, clear responsibilities.

---

## 9. Dependencies, Packaging, CI

### 9.1 Python Version

**Minimum: Python 3.10**. CI matrix runs 3.10, 3.11, 3.12.

Rationale: `match` statements, modern union syntax, `tomllib` (3.11+), and improved performance. Pydantic v2 and torch 2.x have dropped 3.9 support.

### 9.2 Core Dependencies and Extras (Plan B)

`pip install voxkitchen` installs a "batteries-included-minus-GPU-models" set that gives the user a complete non-GPU workflow. Install size ~2.5 GB, ~40 seconds on a typical link.

```toml
[project]
dependencies = [
  # schema + pipeline core
  "pydantic>=2.5,<3",
  "pyyaml>=6",
  "typer>=0.12",
  "rich>=13",
  "numpy>=1.24",
  "soundfile>=0.12",
  "tqdm>=4.66",

  # Plan B: non-GPU audio stack included by default
  "torch>=2.1",           # CPU build
  "torchaudio>=2.1",
  "ffmpeg-python>=0.2",
  "pyloudnorm>=0.1",
  "silero-vad>=4.0",
  "webrtcvad>=2.0",
  "librosa>=0.10",
  "simhash>=2.1",
  "datasets>=2.16",
  "webdataset>=0.2",
  "pyarrow>=14",
  "jinja2>=3.1",
  "plotly>=5.18",
  "gradio>=4.12",
]

[project.optional-dependencies]
asr      = ["faster-whisper>=1.0", "whisperx>=3.1"]
diarize  = ["pyannote.audio>=3.1"]
classify = ["speechbrain>=1.0"]
all      = ["voxkitchen[asr,diarize,classify]"]
dev      = ["pytest>=7.4", "pytest-cov", "ruff", "mypy", "pre-commit"]
```

**Lazy extras checking**: each `Operator` declares `required_extras`. `vkit validate` verifies required extras are installed and emits a precise `pip install voxkitchen[asr]` hint when they are not. Actual `import faster_whisper` happens inside `setup()` so `validate` does not need heavy libraries.

**Install size comparison** (Linux x86_64, 50 MB/s):

| Option | Download | Installed | Time |
|---|---|---|---|
| `voxkitchen` (core, Plan B) | ~900 MB | ~2.5 GB | ~40 s |
| `voxkitchen[asr]` | ~3 GB | ~7 GB | ~2 min |
| `voxkitchen[all]` | ~4.5 GB | ~12 GB | ~3 min |

### 9.3 Packaging Tooling

- **Build backend**: `hatchling` (PEP 517, minimal config).
- **Version management**: `hatch-vcs` (versions from git tags).
- **Not using**: Poetry (lockfiles are wrong for libraries).

### 9.4 Testing Strategy

Three tiers:

**Unit tests (`tests/unit/`)**: schema serialization, CutSet split/merge, GC plan computation, YAML validation, provenance construction. All external libraries mocked. Target: full suite under 30 seconds on a laptop.

**Integration tests (`tests/integration/`)**: small end-to-end pipelines on a 10-second fixture audio file (`tests/fixtures/short.wav`). GPU operators replaced with mock operators on CI. Target: full suite under 3 minutes.

**GPU smoke tests (`tests/gpu/`)**: real models with silero, faster-whisper, pyannote. CI skips these (no GPU runners). Contributors run locally; required before releases.

**Coverage target**: unit tests ≥ 80%. No 100% goal (over-engineering trap).

### 9.5 Code Quality Tooling

- **ruff**: lint + format
- **mypy**: strict mode type checking
- **pre-commit**: runs both automatically

Not using: black (ruff format is sufficient), pylint (ruff covers), bandit (research tool, not production surface).

### 9.6 CI (GitHub Actions)

```
.github/workflows/
├── ci.yml          # push/PR: ruff + mypy + unit + integration on 3.10/3.11/3.12
├── docs.yml        # push main: build mkdocs → gh-pages
└── release.yml     # tag push: build sdist/wheel → PyPI + GitHub Release
```

- Linux x86_64 only. No macOS or Windows runners.
- No Docker image in v0.1.

### 9.7 Documentation Site

- **mkdocs + mkdocs-material**
- API reference auto-generated from docstrings.
- Operator parameter reference auto-generated from Pydantic models.
- Example pipelines linked 1:1 with doc pages.
- Hosted on GitHub Pages.
- English primary, Chinese README and core-concepts page. No video tutorials. No interactive playground.

---

## 10. Non-Goals (v0.1)

### Functional Non-Goals

1. Open-source dataset **catalog UI**. Only 3 ingest recipes ship in v0.1.
2. **Crawler subsystem**. Deferred; has legal and ethical surface that requires its own design.
3. **Manual annotation Web UI**. `vkit viz` is read-only exploration, not a human labeling tool.
4. **Model training**. VoxKitchen produces datasets; training is downstream.
5. **Dataset version management / DVC integration**. Users can layer Git LFS or DVC on top of VoxKitchen outputs.

### Architectural Non-Goals

6. **Distributed execution**. Single-machine multi-GPU only. Ray/Dask integration deferred.
7. **Pipeline parallelism**. Stage parallelism only. Accepts suboptimal GPU utilization in exchange for simplicity and resumability.
8. **Plugin API version negotiation**.
9. **Plugin sandbox or permission system**.
10. **Cloud storage backends** (S3, OSS, GCS). Local filesystem only. The `RunContext` I/O indirection is designed to accommodate future backends but they are not implemented.
11. **Database backend**. All state lives in files (`cuts.jsonl.gz` + `_SUCCESS` markers).

### Operational Non-Goals

12. **Long-running server**. `vkit viz` is transient.
13. **REST / gRPC APIs**.
14. **User accounts or permissions**.
15. **SQL-style querying**. Use `vkit inspect` and `vkit viz` semantic filters.

### Platform Non-Goals

16. **Windows / macOS CI**. Linux x86_64 only. Other platforms best-effort without guarantees.
17. **ARM / Apple Silicon optimization**. CPU torch works on M-series Macs for pipeline authoring. GPU operators do not support MPS backend.
18. **Docker image**. v0.1 ships as a PyPI package only.

### Documentation Non-Goals

19. **Multi-language documentation**. English primary, with a Chinese README and core concepts page.
20. **Video tutorials**.

---

## Appendix A: Guiding Principles

1. **Simplicity over efficiency**. Every design choice where simplicity and raw performance conflict resolves in favor of simplicity. This is project-wide policy, not a local trade-off.
2. **No over-engineering**. Every abstraction, retry mechanism, fallback, or auto-detection must justify itself against a concrete demonstrated need. Hypothetical flexibility is rejected.
3. **Inspectable intermediate state**. Every stage writes a valid manifest that can be examined, queried, and reused by hand. Resume and debugging fall out naturally.
4. **Declarative when possible, imperative when necessary**. YAML is the primary interface; Python API is the fallback for users who need dynamic behavior.
5. **Researcher ergonomics**. Reproducibility, provenance, and offline operation take precedence over operational polish aimed at production deployment.

## Appendix B: Success Criteria Checklist

The v0.1 release is complete when:

- [ ] All 22 built-in operators are implemented with unit tests and reference docs.
- [ ] All 3 ingest recipes (LibriSpeech, CommonVoice, AISHELL-1) are implemented and tested against real downloads.
- [ ] The example pipeline `examples/pipelines/librispeech-asr.yaml` runs end-to-end on a single machine with 4 GPUs and produces a HuggingFace Datasets directory plus a `report.html`.
- [ ] `vkit run --resume-from` works correctly after a simulated mid-pipeline crash.
- [ ] `--keep-intermediates` and aggressive GC both produce correct results on the example pipeline.
- [ ] `vkit viz` launches the Gradio panel and displays the example pipeline output interactively.
- [ ] Plugin discovery loads a toy third-party operator from a separately-installed package.
- [ ] `pip install voxkitchen` succeeds in a clean Python 3.10 / 3.11 / 3.12 environment and produces a working `vkit --help`.
- [ ] `pip install voxkitchen[asr,diarize,classify]` succeeds and enables all GPU operators.
- [ ] Documentation site builds and deploys via `docs.yml`.
- [ ] `ci.yml` is green on all supported Python versions.
