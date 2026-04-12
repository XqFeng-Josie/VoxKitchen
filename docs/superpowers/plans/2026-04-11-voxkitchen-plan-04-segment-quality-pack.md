# VoxKitchen Plan 4: Segment + Quality + Pack Operators

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement 11 non-GPU operators across three categories — segmentation (3), quality analysis/filtering (4), and output packing (4). After this plan, users can build complete non-GPU data processing pipelines: scan → segment → analyze quality → filter → pack to HuggingFace/Kaldi/Parquet/WebDataset format.

**Architecture:** All 11 operators follow the patterns established in Plans 2-3. Segmentation operators are the first "1-to-many" transform (one input Cut → N output Cuts) but the runner needs no changes — it just sees CutSet in, CutSet out. Quality operators add metrics to `cut.metrics` dict or filter cuts. Pack operators write final output in a target format. No new framework code — only new operator files + their registration.

**Tech Stack:** webrtcvad (VAD), librosa (silence detection, MFCC features), simhash (audio fingerprinting), datasets (HuggingFace format), webdataset (tar archives), pyarrow (Parquet). All new deps go as extras to keep default install fast.

**Spec reference:** Design spec section 4.6 (operator catalog table).

**Key principle:** Every operator is a single file under `operators/<category>/`, < 100 lines of `process()` logic, registered via `@register_operator`. No shared base classes between operators beyond the existing `Operator` ABC. No batch optimization — one Cut at a time.

---

## File Structure Produced by This Plan

```
src/voxkitchen/operators/
├── segment/
│   ├── __init__.py
│   ├── fixed_segment.py         # split cuts at fixed time intervals
│   ├── webrtc_vad.py            # speech detection via webrtcvad
│   └── silence_split.py         # split on silence via librosa
├── quality/
│   ├── __init__.py
│   ├── snr_estimate.py          # estimate SNR → cut.metrics["snr"]
│   ├── duration_filter.py       # filter by min/max duration
│   ├── audio_fingerprint_dedup.py  # remove near-duplicate audio
│   └── quality_score_filter.py  # filter by expression on metrics
├── pack/
│   ├── __init__.py              # MODIFIED (pack_manifest already here)
│   ├── pack_manifest.py         # (already exists from Plan 3)
│   ├── pack_huggingface.py      # export as HuggingFace datasets
│   ├── pack_webdataset.py       # export as tar shards
│   ├── pack_parquet.py          # export as Parquet + JSONL manifest
│   └── pack_kaldi.py            # export as Kaldi text files

tests/unit/operators/
├── segment/
│   ├── __init__.py
│   ├── test_fixed_segment.py
│   ├── test_webrtc_vad.py
│   └── test_silence_split.py
├── quality/
│   ├── __init__.py
│   ├── test_snr_estimate.py
│   ├── test_duration_filter.py
│   ├── test_audio_fingerprint_dedup.py
│   └── test_quality_score_filter.py
├── pack/
│   ├── test_pack_huggingface.py
│   ├── test_pack_webdataset.py
│   ├── test_pack_parquet.py
│   └── test_pack_kaldi.py
```

---

## Design Notes for Operator Categories

### Segmentation operators (1-to-many, `produces_audio=False`)

These are the first "fan-out" operators: one input Cut → N output Cuts. Each child Cut:
- Shares the parent's `recording` and `recording_id` (same audio file)
- Has different `start` and `duration` (the detected segment interval)
- Has `provenance.source_cut_id = parent.id`
- Has empty `supervisions` (downstream operators like ASR will fill them)

No new audio files are written — child cuts just reference sub-intervals of the parent's audio. `produces_audio=False`, `reads_audio_bytes=True` (they need to read audio to find segment boundaries).

### Quality operators (1-to-1 or N-to-fewer, `produces_audio=False`)

Two sub-patterns:
- **Estimators** (`snr_estimate`): 1-to-1. Read audio, compute a metric, store in `cut.metrics`. Return same number of cuts.
- **Filters** (`duration_filter`, `quality_score_filter`, `audio_fingerprint_dedup`): N-to-fewer. Remove cuts that fail a condition. Return fewer cuts.

All have `produces_audio=False`. Estimators have `reads_audio_bytes=True`; filters may or may not read audio (`duration_filter` doesn't, `dedup` does).

### Pack operators (`produces_audio=True` for formats with audio, `False` for metadata-only)

Pack operators write the final output in a specific format. They don't modify the CutSet — they write output to `config.output_dir` (or `stage_dir/output/`) as a side effect, then return the CutSet unchanged. The `produces_audio` flag depends on whether audio bytes are copied into the output format:
- `pack_huggingface`: True (copies audio into Arrow files)
- `pack_webdataset`: True (copies audio into tar shards)
- `pack_parquet`: False (only metadata; audio stays at original paths)
- `pack_kaldi`: False (text files + wav.scp pointing at paths)

---

## Task 1: Add segment/quality/pack dependencies

**Files:**
- Modify: `pyproject.toml`

- [ ] **Step 1: Add new extras groups**

```toml
[project.optional-dependencies]
audio = ["torch>=2.1", "torchaudio>=2.1"]
segment = ["webrtcvad>=2.0", "librosa>=0.10"]
quality = ["simhash>=2.1"]
pack = ["datasets>=2.16", "webdataset>=0.2", "pyarrow>=14"]
dev = [
    "pytest>=7.4",
    "pytest-cov>=4.1",
    "ruff>=0.4",
    "mypy>=1.8",
    "pre-commit>=3.6",
    "scipy>=1.11",
]
```

**Note:** `librosa` and `webrtcvad` go in `segment` extras. `simhash` goes in `quality`. Pack libraries go in `pack`. `scipy` added to dev for resample fallback tests. Core deps stay unchanged.

- [ ] **Step 2: Update mypy overrides**

Add `librosa.*`, `webrtcvad.*`, `simhash.*`, `datasets.*`, `webdataset.*` to the ignore_missing_imports override.

- [ ] **Step 3: Update CI to install new extras**

```yaml
      - name: Install package
        run: |
          python -m pip install --upgrade pip
          pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
          pip install -e ".[dev,segment,quality,pack]"
```

- [ ] **Step 4: Install locally + verify tests**

```bash
pip install -e ".[dev,segment,quality,pack]"
pytest -q
```

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml .github/workflows/ci.yml
git commit -m "chore: add segment/quality/pack dependency extras"
```

---

## Task 2: `fixed_segment` + `duration_filter` (pure compute, no deps)

**Files:**
- Create: `src/voxkitchen/operators/segment/__init__.py`
- Create: `src/voxkitchen/operators/segment/fixed_segment.py`
- Create: `src/voxkitchen/operators/quality/__init__.py`
- Create: `src/voxkitchen/operators/quality/duration_filter.py`
- Create: `tests/unit/operators/segment/__init__.py`
- Create: `tests/unit/operators/segment/test_fixed_segment.py`
- Create: `tests/unit/operators/quality/__init__.py`
- Create: `tests/unit/operators/quality/test_duration_filter.py`
- Modify: `src/voxkitchen/operators/__init__.py`

### `fixed_segment` — the segmentation exemplar

**Config:**
```python
class FixedSegmentConfig(OperatorConfig):
    segment_duration: float = 10.0   # seconds
    min_remaining: float = 0.5       # drop final chunk if shorter than this
```

**Logic:**
```python
def process(self, cuts: CutSet) -> CutSet:
    out: list[Cut] = []
    for cut in cuts:
        t = 0.0
        idx = 0
        while t < cut.duration:
            seg_dur = min(self.config.segment_duration, cut.duration - t)
            if seg_dur < self.config.min_remaining:
                break
            child = Cut(
                id=f"{cut.id}__seg{idx}",
                recording_id=cut.recording_id,
                start=cut.start + t,
                duration=seg_dur,
                recording=cut.recording,
                supervisions=[],
                provenance=Provenance(..., source_cut_id=cut.id, generated_by="fixed_segment@..."),
                custom=cut.custom,
            )
            out.append(child)
            t += self.config.segment_duration
            idx += 1
    return CutSet(out)
```

**Class attributes:** `name="fixed_segment"`, `device="cpu"`, `produces_audio=False`, `reads_audio_bytes=False`

**Tests (4):**
1. `test_fixed_segment_is_registered`
2. `test_fixed_segment_splits_long_cut` — 10s cut with 3s segments → 3 full + 1 short (if >= min_remaining)
3. `test_fixed_segment_drops_short_remainder` — 10s with 3s segments, min_remaining=2.0 → drops the 1s remainder
4. `test_fixed_segment_preserves_recording_reference` — child cuts share parent's recording

### `duration_filter` — the filter exemplar

**Config:**
```python
class DurationFilterConfig(OperatorConfig):
    min_duration: float = 0.0
    max_duration: float = float("inf")
```

**Logic:**
```python
def process(self, cuts: CutSet) -> CutSet:
    return CutSet(
        c for c in cuts
        if self.config.min_duration <= c.duration <= self.config.max_duration
    )
```

**Class attributes:** `name="duration_filter"`, `device="cpu"`, `produces_audio=False`, `reads_audio_bytes=False`

**Tests (3):**
1. `test_duration_filter_is_registered`
2. `test_duration_filter_keeps_cuts_in_range` — mix of durations → only matching kept
3. `test_duration_filter_defaults_keep_all` — no config args → all cuts pass

### TDD for both: test → fail → implement → pass → commit

**Commit:** `feat(operators): add fixed_segment and duration_filter`

---

## Task 3: `webrtc_vad` + `silence_split` (segmentation with audio analysis)

**Files:**
- Create: `src/voxkitchen/operators/segment/webrtc_vad.py`
- Create: `src/voxkitchen/operators/segment/silence_split.py`
- Create: `tests/unit/operators/segment/test_webrtc_vad.py`
- Create: `tests/unit/operators/segment/test_silence_split.py`
- Modify: `src/voxkitchen/operators/__init__.py`

### `webrtc_vad`

Uses the `webrtcvad` library to detect speech/non-speech in 10ms frames.

**Config:**
```python
class WebrtcVadConfig(OperatorConfig):
    aggressiveness: int = 2          # 0-3, higher = more aggressive filtering
    frame_duration_ms: int = 30      # 10, 20, or 30 ms
    min_speech_duration_ms: int = 250
    padding_ms: int = 30
```

**Logic sketch:**
1. `load_audio_for_cut(cut)` → audio, sr
2. If sr not in (8000, 16000, 32000, 48000): resample to 16000 (webrtcvad requirement)
3. Convert to int16 PCM bytes
4. Run webrtcvad frame-by-frame, collect speech regions
5. Merge adjacent speech frames, apply min_speech_duration and padding
6. Create child cuts for each speech region

**Class attributes:** `name="webrtc_vad"`, `device="cpu"`, `produces_audio=False`, `reads_audio_bytes=True`, `required_extras=["segment"]`

**Tests (3):**
1. `test_webrtc_vad_is_registered`
2. `test_webrtc_vad_detects_speech_in_sine_wave(mono_wav_16k)` — a pure tone should be detected as "speech" (webrtcvad treats tonal signals as speech-like)
3. `test_webrtc_vad_skips_silence` — generate a file that is half silence, half tone → should produce fewer/shorter segments

### `silence_split`

Uses `librosa.effects.split()` to find non-silent intervals.

**Config:**
```python
class SilenceSplitConfig(OperatorConfig):
    top_db: int = 30                 # dB threshold below peak to consider silence
    min_duration: float = 0.1        # minimum segment duration in seconds
```

**Logic:**
1. `load_audio_for_cut(cut)` → audio, sr
2. `intervals = librosa.effects.split(audio, top_db=config.top_db)`
3. Convert frame intervals to seconds
4. Filter by min_duration
5. Create child cuts

**Class attributes:** `name="silence_split"`, `device="cpu"`, `produces_audio=False`, `reads_audio_bytes=True`, `required_extras=["segment"]`

**Tests (3):**
1. `test_silence_split_is_registered`
2. `test_silence_split_finds_non_silent_regions(mono_wav_16k)` — sine wave is all non-silent → 1 segment ≈ full duration
3. `test_silence_split_handles_silent_file` — generate silence → 0 segments

### Commit: `feat(operators): add webrtc_vad and silence_split segmentation`

---

## Task 4: `snr_estimate` + `audio_fingerprint_dedup` (audio analysis)

**Files:**
- Create: `src/voxkitchen/operators/quality/snr_estimate.py`
- Create: `src/voxkitchen/operators/quality/audio_fingerprint_dedup.py`
- Create: `tests/unit/operators/quality/test_snr_estimate.py`
- Create: `tests/unit/operators/quality/test_audio_fingerprint_dedup.py`
- Modify: `src/voxkitchen/operators/__init__.py`

### `snr_estimate`

Estimates signal-to-noise ratio and stores it in `cut.metrics["snr"]`.

**Config:**
```python
class SnrEstimateConfig(OperatorConfig):
    pass  # no tunable parameters for v0.1
```

**Logic:**
1. `load_audio_for_cut(cut)` → audio, sr
2. Estimate SNR using a simple energy-based method: `signal_power / noise_power` in dB. For v0.1 use a basic approach — e.g., WADA-SNR or a simple peak/RMS ratio. Keep it simple.
3. `cut.model_copy(update={"metrics": {**cut.metrics, "snr": snr_value}})`

**Class attributes:** `name="snr_estimate"`, `device="cpu"`, `produces_audio=False`, `reads_audio_bytes=True`

**Tests (3):**
1. `test_snr_estimate_is_registered`
2. `test_snr_estimate_adds_metric(mono_wav_16k)` — run on a clean sine → `metrics["snr"]` exists and is positive
3. `test_snr_estimate_preserves_other_metrics` — cut with existing metrics → snr added without losing others

### `audio_fingerprint_dedup`

Removes near-duplicate audio using MFCC-based fingerprinting + simhash.

**Config:**
```python
class AudioFingerprintDedupConfig(OperatorConfig):
    similarity_threshold: int = 3    # max hamming distance to consider duplicate
```

**Logic:**
1. For each cut: `load_audio_for_cut(cut)` → audio, sr
2. Compute MFCC features via librosa: `mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)`
3. Flatten to a feature vector, binarize, compute simhash
4. Compare each new simhash against seen hashes (hamming distance)
5. If distance <= threshold: skip (duplicate). Else: keep + add to seen set.

**Class attributes:** `name="audio_fingerprint_dedup"`, `device="cpu"`, `produces_audio=False`, `reads_audio_bytes=True`, `required_extras=["quality"]`

**Tests (3):**
1. `test_dedup_is_registered`
2. `test_dedup_removes_identical_cuts` — CutSet with 2 cuts pointing to same audio → 1 kept
3. `test_dedup_keeps_different_cuts` — CutSet with 2 different audio files → both kept

### Commit: `feat(operators): add snr_estimate and audio_fingerprint_dedup`

---

## Task 5: `quality_score_filter` (condition-based filtering)

**Files:**
- Create: `src/voxkitchen/operators/quality/quality_score_filter.py`
- Create: `tests/unit/operators/quality/test_quality_score_filter.py`
- Modify: `src/voxkitchen/operators/__init__.py`

### Config
```python
class QualityScoreFilterConfig(OperatorConfig):
    conditions: list[str]  # e.g., ["metrics.snr > 10", "duration > 0.5"]
```

### Logic

A simple condition parser that supports expressions of the form `<field_path> <op> <value>`:

```python
import ast
import operator as op_mod

_OPS = {">": op_mod.gt, ">=": op_mod.ge, "<": op_mod.lt, "<=": op_mod.le, "==": op_mod.eq, "!=": op_mod.ne}

def _eval_condition(condition: str, cut: Cut) -> bool:
    """Parse 'metrics.snr > 10' and evaluate against a Cut."""
    parts = condition.split()
    if len(parts) != 3:
        raise ValueError(f"condition must be 'field op value', got: {condition!r}")
    field_path, op_str, raw_value = parts
    if op_str not in _OPS:
        raise ValueError(f"unsupported operator: {op_str!r}")
    value = ast.literal_eval(raw_value)
    actual = _resolve_field(cut, field_path)
    return _OPS[op_str](actual, value)

def _resolve_field(cut: Cut, path: str) -> Any:
    """Resolve 'metrics.snr' or 'duration' against a Cut."""
    obj: Any = cut
    for part in path.split("."):
        if isinstance(obj, dict):
            obj = obj[part]
        else:
            obj = getattr(obj, part)
    return obj
```

**process():**
```python
def process(self, cuts: CutSet) -> CutSet:
    return CutSet(
        cut for cut in cuts
        if all(_eval_condition(cond, cut) for cond in self.config.conditions)
    )
```

**Class attributes:** `name="quality_score_filter"`, `device="cpu"`, `produces_audio=False`, `reads_audio_bytes=False`

**Tests (4):**
1. `test_quality_score_filter_is_registered`
2. `test_filter_by_snr` — cuts with metrics.snr of 5, 15, 25 + condition "metrics.snr > 10" → 2 kept
3. `test_filter_by_duration` — condition "duration > 2.0" → only long cuts kept
4. `test_filter_multiple_conditions` — two conditions AND'd together

### Commit: `feat(operators): add quality_score_filter with condition parser`

---

## Task 6: `pack_kaldi` + `pack_parquet` (metadata output formats)

**Files:**
- Create: `src/voxkitchen/operators/pack/pack_kaldi.py`
- Create: `src/voxkitchen/operators/pack/pack_parquet.py`
- Create: `tests/unit/operators/pack/test_pack_kaldi.py`
- Create: `tests/unit/operators/pack/test_pack_parquet.py`
- Modify: `src/voxkitchen/operators/__init__.py`

### `pack_kaldi`

Writes Kaldi-style text files: `wav.scp`, `text`, `utt2spk`, `segments`.

**Config:**
```python
class PackKaldiConfig(OperatorConfig):
    output_dir: str | None = None  # defaults to stage_dir / "kaldi_output"
```

**Logic:**
```python
def process(self, cuts: CutSet) -> CutSet:
    out = Path(self.config.output_dir or str(self.ctx.stage_dir / "kaldi_output"))
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "wav.scp", "w") as wav_scp, \
         open(out / "text", "w") as text_f, \
         open(out / "utt2spk", "w") as u2s:
        for cut in cuts:
            audio_path = cut.recording.sources[0].source if cut.recording else "MISSING"
            wav_scp.write(f"{cut.id} {audio_path}\n")
            transcript = next((s.text for s in cut.supervisions if s.text), "")
            text_f.write(f"{cut.id} {transcript}\n")
            speaker = next((s.speaker for s in cut.supervisions if s.speaker), "unknown")
            u2s.write(f"{cut.id} {speaker}\n")
    return CutSet(list(cuts))
```

**Class attributes:** `name="pack_kaldi"`, `produces_audio=False`, `reads_audio_bytes=False`

**Tests (3):**
1. `test_pack_kaldi_is_registered`
2. `test_pack_kaldi_writes_files` — verify wav.scp, text, utt2spk exist
3. `test_pack_kaldi_content` — verify cut ids and paths in wav.scp

### `pack_parquet`

Writes metadata to a Parquet file (no audio bytes, just references).

**Config:**
```python
class PackParquetConfig(OperatorConfig):
    output_dir: str | None = None
```

**Logic:**
```python
def process(self, cuts: CutSet) -> CutSet:
    import pyarrow as pa
    import pyarrow.parquet as pq

    rows = []
    for cut in cuts:
        rows.append({
            "id": cut.id,
            "recording_id": cut.recording_id,
            "audio_path": cut.recording.sources[0].source if cut.recording else None,
            "start": cut.start,
            "duration": cut.duration,
            "text": next((s.text for s in cut.supervisions if s.text), None),
            "speaker": next((s.speaker for s in cut.supervisions if s.speaker), None),
            "language": next((s.language for s in cut.supervisions if s.language), None),
            **{f"metrics_{k}": v for k, v in cut.metrics.items()},
        })
    table = pa.Table.from_pylist(rows)
    out_dir = Path(self.config.output_dir or str(self.ctx.stage_dir / "parquet_output"))
    out_dir.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, out_dir / "metadata.parquet")
    return CutSet(list(cuts))
```

**Class attributes:** `name="pack_parquet"`, `produces_audio=False`, `reads_audio_bytes=False`, `required_extras=["pack"]`

**Tests (3):**
1. `test_pack_parquet_is_registered`
2. `test_pack_parquet_writes_file` — verify metadata.parquet exists
3. `test_pack_parquet_has_correct_columns` — read back with pyarrow, check columns

### Commit: `feat(operators): add pack_kaldi and pack_parquet output formats`

---

## Task 7: `pack_huggingface` + `pack_webdataset` (archive output formats)

**Files:**
- Create: `src/voxkitchen/operators/pack/pack_huggingface.py`
- Create: `src/voxkitchen/operators/pack/pack_webdataset.py`
- Create: `tests/unit/operators/pack/test_pack_huggingface.py`
- Create: `tests/unit/operators/pack/test_pack_webdataset.py`
- Modify: `src/voxkitchen/operators/__init__.py`

### `pack_huggingface`

Creates a HuggingFace `datasets.Dataset` with audio + metadata.

**Config:**
```python
class PackHuggingFaceConfig(OperatorConfig):
    output_dir: str | None = None
    split_field: str | None = None   # optional: use cut.custom[split_field] as split name
```

**Logic:**
```python
def process(self, cuts: CutSet) -> CutSet:
    from datasets import Audio, Dataset

    records = []
    for cut in cuts:
        audio_path = cut.recording.sources[0].source if cut.recording else None
        records.append({
            "id": cut.id,
            "audio": audio_path,
            "text": next((s.text for s in cut.supervisions if s.text), None),
            "speaker": next((s.speaker for s in cut.supervisions if s.speaker), None),
            "duration": cut.duration,
        })
    ds = Dataset.from_list(records)
    ds = ds.cast_column("audio", Audio())
    out_dir = Path(self.config.output_dir or str(self.ctx.stage_dir / "hf_output"))
    ds.save_to_disk(str(out_dir))
    return CutSet(list(cuts))
```

**Class attributes:** `name="pack_huggingface"`, `produces_audio=True` (copies audio into Arrow), `reads_audio_bytes=True`, `required_extras=["pack"]`

**Tests (3):**
1. `test_pack_hf_is_registered`
2. `test_pack_hf_creates_dataset(mono_wav_16k)` — verify output dir has `dataset_info.json` or Arrow files
3. `test_pack_hf_preserves_cut_count(mono_wav_16k)` — returned CutSet has same length

### `pack_webdataset`

Creates tar shards with audio + JSON metadata per sample.

**Config:**
```python
class PackWebDatasetConfig(OperatorConfig):
    output_dir: str | None = None
    shard_size: int = 1000           # max samples per shard
```

**Logic:**
```python
def process(self, cuts: CutSet) -> CutSet:
    import webdataset as wds

    out_dir = Path(self.config.output_dir or str(self.ctx.stage_dir / "wds_output"))
    out_dir.mkdir(parents=True, exist_ok=True)
    pattern = str(out_dir / "shard-%05d.tar")

    with wds.ShardWriter(pattern, maxcount=self.config.shard_size) as sink:
        for cut in cuts:
            audio_path = cut.recording.sources[0].source if cut.recording else None
            sample = {"__key__": cut.id}
            if audio_path:
                with open(audio_path, "rb") as f:
                    sample["audio.wav"] = f.read()
            sample["metadata.json"] = json.dumps({
                "id": cut.id, "duration": cut.duration,
                "text": next((s.text for s in cut.supervisions if s.text), None),
            }).encode()
            sink.write(sample)
    return CutSet(list(cuts))
```

**Class attributes:** `name="pack_webdataset"`, `produces_audio=True`, `reads_audio_bytes=True`, `required_extras=["pack"]`

**Tests (3):**
1. `test_pack_wds_is_registered`
2. `test_pack_wds_creates_tar(mono_wav_16k)` — verify at least one `.tar` file exists in output
3. `test_pack_wds_preserves_cut_count`

### Commit: `feat(operators): add pack_huggingface and pack_webdataset`

---

## Task 8: Integration test — quality pipeline

**Files:**
- Create: `tests/integration/test_quality_pipeline_e2e.py`

A pipeline that exercises segment → quality → filter → pack:

```python
def test_segment_filter_pack_pipeline(audio_dir: Path, tmp_path: Path) -> None:
    """dir_scan → fixed_segment → duration_filter → pack_kaldi"""
    work_dir = tmp_path / "work"
    yaml_path = tmp_path / "pipeline.yaml"
    yaml_path.write_text(f"""
version: "0.1"
name: quality-e2e
work_dir: {work_dir}
num_cpu_workers: 1
ingest:
  source: dir
  args:
    root: {audio_dir}
    recursive: true
stages:
  - name: segment
    op: fixed_segment
    args:
      segment_duration: 0.3
      min_remaining: 0.1
  - name: filter
    op: duration_filter
    args:
      min_duration: 0.2
  - name: pack
    op: pack_kaldi
""")
    spec = load_pipeline_spec(yaml_path, run_id="run-quality-e2e")
    run_pipeline(spec)

    # All stages complete
    for i, name in enumerate(["segment", "filter", "pack"]):
        assert (work_dir / f"{i:02d}_{name}" / "_SUCCESS").exists()

    # Segmentation should produce more cuts than input
    segment_cuts = list(read_cuts(work_dir / "00_segment" / "cuts.jsonl.gz"))
    assert len(segment_cuts) > 3  # 3 input files, each ~1s, with 0.3s segments → ~9

    # Duration filter may drop short segments
    filter_cuts = list(read_cuts(work_dir / "01_filter" / "cuts.jsonl.gz"))
    assert len(filter_cuts) <= len(segment_cuts)

    # Pack should write Kaldi files
    kaldi_dir = work_dir / "02_pack" / "kaldi_output"
    assert (kaldi_dir / "wav.scp").exists()
```

### Commit: `test(integration): add segment → filter → pack quality pipeline test`

---

## Task 9: Full verification, lint, type, tag

- [ ] Run `pytest -v` — all tests pass
- [ ] `ruff check src tests` + `ruff format --check src tests` — clean
- [ ] `mypy src/voxkitchen tests` — clean
- [ ] `pre-commit run --all-files` — all hooks pass
- [ ] `vkit validate` works on a pipeline using the new operators
- [ ] Tag: `git tag -a plan-04-segment-quality-pack -m "Plan 4 complete: segment, quality, and pack operators"`

---

## Plan 4 Completion Checklist

- [ ] All 11 new operators registered: `fixed_segment`, `webrtc_vad`, `silence_split`, `snr_estimate`, `duration_filter`, `audio_fingerprint_dedup`, `quality_score_filter`, `pack_kaldi`, `pack_parquet`, `pack_huggingface`, `pack_webdataset`
- [ ] Segmentation operators create child cuts with correct `start`/`duration`/`provenance`
- [ ] `quality_score_filter` evaluates conditions like `"metrics.snr > 10"`
- [ ] Pack operators write output in correct format (Kaldi text files, Parquet, HF Arrow, WebDataset tar)
- [ ] Integration test: full segment → filter → pack pipeline works
- [ ] All existing tests pass (no regressions)
- [ ] `git tag plan-04-segment-quality-pack` at HEAD

## What Plans 5-8 Will Build On

**Plan 5 (GPU operators):** silero_vad, faster_whisper_asr, whisperx_asr, pyannote_diarize, speechbrain_langid, speechbrain_gender. These follow the same operator patterns but need `device="gpu"` and run through `GpuPoolExecutor`.

**Plan 6 (Ingest recipes):** LibriSpeech, CommonVoice, AISHELL-1. Download + convert to CutSet.

**Plan 7 (Visualization):** Rich inspect views, HTML report, Gradio panel. Uses `CutSet` and `read_cuts` extensively.

**Plan 8 (Plugin + polish):** entry_points, `vkit init`, docs, release.
