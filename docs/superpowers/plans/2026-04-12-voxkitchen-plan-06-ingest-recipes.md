# VoxKitchen Plan 6: Ingest Recipes

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the recipe ingest framework and three dataset recipes (LibriSpeech, CommonVoice, AISHELL-1). After this plan, users can write `ingest: { source: recipe, recipe: librispeech, args: { root: /data/librispeech } }` in their pipeline YAML and have it produce a CutSet from a locally-present dataset.

**Architecture:** Recipes are simple parsing functions — they read dataset-specific directory structures and produce CutSets with embedded Recordings. A `RecipeIngestSource` dispatches to the correct recipe by name. Downloads are optional and secondary — the primary path is parsing already-downloaded data. No new framework abstractions beyond a thin `Recipe` base class and a recipe registry dict.

**Tech Stack:** soundfile (audio metadata), standard library (pathlib, csv). No new deps.

**Key constraint:** Tests use mock directory structures with tiny generated WAV files — no real dataset downloads in any test.

---

## Design: Recipe Interface

```python
class RecipeConfig(IngestConfig):
    root: str                            # path to the dataset root
    subsets: list[str] | None = None     # e.g., ["train-clean-100", "dev-clean"]

class Recipe(ABC):
    name: str

    @abstractmethod
    def prepare(self, root: Path, subsets: list[str] | None, ctx: RunContext) -> CutSet:
        """Parse a locally-present dataset into a CutSet."""

class RecipeIngestSource(IngestSource):
    """Dispatches to the right Recipe by name."""
    name = "recipe"

    def run(self) -> CutSet:
        recipe = get_recipe(self.config.recipe)
        return recipe.prepare(Path(self.config.root), self.config.subsets, self.ctx)
```

This is intentionally thin. Each recipe is one file with a `prepare()` method that reads the dataset's directory layout and creates Cuts. No shared parsing utilities, no complex hierarchy — each recipe knows its own format.

## Dataset Formats (what each recipe parses)

### LibriSpeech
```
root/
├── train-clean-100/
│   └── <spk_id>/
│       └── <chapter_id>/
│           ├── <spk_id>-<chapter_id>-<utt_id>.flac
│           └── <spk_id>-<chapter_id>.trans.txt   # "UTT_ID TRANSCRIPT TEXT\n" per line
├── dev-clean/
│   └── ...
```

### CommonVoice
```
root/
├── clips/
│   ├── common_voice_xx_12345.mp3
│   └── ...
├── train.tsv     # client_id\tpath\tsentence\tup_votes\tdown_votes\tage\tgender\taccent\tlocale\tsegment
├── dev.tsv
├── test.tsv
```

### AISHELL-1
```
root/
├── data_aishell/
│   ├── wav/
│   │   ├── train/
│   │   │   └── S0XXX/
│   │   │       └── BAXXXX.wav
│   │   ├── dev/
│   │   └── test/
│   └── transcript/
│       └── aishell_transcript_v0.8.txt   # "UTTID 汉 字 转 写\n" (space-separated chars)
```

---

## File Structure

```
src/voxkitchen/ingest/
├── __init__.py                    # MODIFIED: add "recipe" to _INGEST_SOURCES
├── base.py                        # (unchanged)
├── dir_scan.py                    # (unchanged)
├── manifest_import.py             # (unchanged)
├── recipe_source.py               # NEW: RecipeIngestSource + RecipeConfig
└── recipes/
    ├── __init__.py                # NEW: recipe registry + get_recipe()
    ├── base.py                    # NEW: Recipe ABC
    ├── librispeech.py             # NEW
    ├── commonvoice.py             # NEW
    └── aishell.py                 # NEW

tests/unit/ingest/
├── test_recipe_source.py          # NEW: RecipeIngestSource dispatch
└── recipes/
    ├── __init__.py
    ├── conftest.py                # NEW: mock dataset fixtures
    ├── test_librispeech.py        # NEW
    ├── test_commonvoice.py        # NEW
    └── test_aishell.py            # NEW
```

---

## Task 1: Recipe framework (RecipeIngestSource + base + registry)

**Files:**
- Create: `src/voxkitchen/ingest/recipes/__init__.py`
- Create: `src/voxkitchen/ingest/recipes/base.py`
- Create: `src/voxkitchen/ingest/recipe_source.py`
- Create: `tests/unit/ingest/test_recipe_source.py`
- Create: `tests/unit/ingest/recipes/__init__.py`
- Modify: `src/voxkitchen/ingest/__init__.py`
- Modify: `src/voxkitchen/pipeline/spec.py` (if needed — `"recipe"` already in Literal)

### `recipes/base.py` — Recipe ABC

```python
"""Recipe base class: parse a dataset directory into a CutSet."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING

from voxkitchen.schema.cutset import CutSet

if TYPE_CHECKING:
    from voxkitchen.pipeline.context import RunContext


class Recipe(ABC):
    name: str = ""

    @abstractmethod
    def prepare(self, root: Path, subsets: list[str] | None, ctx: RunContext) -> CutSet:
        """Parse a locally-present dataset and return a CutSet."""
```

### `recipes/__init__.py` — registry

```python
_RECIPES: dict[str, Recipe] = {}

def register_recipe(recipe: Recipe) -> Recipe: ...
def get_recipe(name: str) -> Recipe: ...
```

### `recipe_source.py` — RecipeIngestSource

```python
class RecipeConfig(IngestConfig):
    recipe: str
    root: str
    subsets: list[str] | None = None

class RecipeIngestSource(IngestSource):
    name = "recipe"
    config_cls = RecipeConfig

    def run(self) -> CutSet:
        recipe = get_recipe(self.config.recipe)
        return recipe.prepare(Path(self.config.root), self.config.subsets, self.ctx)
```

### Tests (3)
1. `test_recipe_ingest_source_is_registered` — `get_ingest_source("recipe")` returns `RecipeIngestSource`
2. `test_recipe_config_requires_recipe_and_root` — validation
3. `test_recipe_dispatch_calls_prepare` — mock a recipe, verify `prepare()` is called

### Commit: `feat(ingest): add recipe framework with RecipeIngestSource`

---

## Task 2: LibriSpeech recipe (TDD)

**Files:**
- Create: `src/voxkitchen/ingest/recipes/librispeech.py`
- Create: `tests/unit/ingest/recipes/conftest.py`
- Create: `tests/unit/ingest/recipes/test_librispeech.py`
- Modify: `src/voxkitchen/ingest/recipes/__init__.py`

### Mock fixture in `conftest.py`

Generate a small LibriSpeech-like directory:
```python
@pytest.fixture
def mock_librispeech(tmp_path: Path) -> Path:
    """Create a tiny LibriSpeech-like directory with 2 utterances."""
    subset = tmp_path / "train-clean-100" / "1089" / "134686"
    subset.mkdir(parents=True)

    # Create 2 short FLAC files (actually WAV — soundfile can read either)
    for utt_id in ["0001", "0002"]:
        fname = f"1089-134686-{utt_id}.flac"
        audio = np.sin(np.linspace(0, 1, 16000)).astype(np.float32) * 0.5
        sf.write(subset / fname, audio, 16000)

    # Transcript file
    (subset / "1089-134686.trans.txt").write_text(
        "1089-134686-0001 HELLO WORLD\n"
        "1089-134686-0002 GOODBYE WORLD\n"
    )
    return tmp_path
```

### LibriSpeech recipe logic

```python
class LibriSpeechRecipe(Recipe):
    name = "librispeech"

    def prepare(self, root, subsets, ctx):
        cuts = []
        target_subsets = subsets or self._discover_subsets(root)
        for subset_name in target_subsets:
            subset_dir = root / subset_name
            if not subset_dir.is_dir():
                raise FileNotFoundError(f"subset not found: {subset_dir}")
            # Walk speaker/chapter dirs
            for trans_file in subset_dir.rglob("*.trans.txt"):
                chapter_dir = trans_file.parent
                transcripts = self._parse_transcript(trans_file)
                for utt_id, text in transcripts.items():
                    audio_path = chapter_dir / f"{utt_id}.flac"
                    if not audio_path.exists():
                        continue
                    rec = recording_from_file(audio_path, recording_id=utt_id)
                    cuts.append(Cut(
                        id=utt_id, recording_id=rec.id, start=0.0,
                        duration=rec.duration, recording=rec,
                        supervisions=[Supervision(
                            id=f"{utt_id}__text", recording_id=rec.id,
                            start=0.0, duration=rec.duration, text=text,
                            speaker=utt_id.split("-")[0],  # speaker from ID
                        )],
                        provenance=Provenance(...),
                        custom={"subset": subset_name},
                    ))
        return CutSet(cuts)
```

### Tests (4)
1. `test_librispeech_recipe_is_registered`
2. `test_librispeech_parses_mock_data(mock_librispeech)` — 2 cuts, correct IDs, correct transcript text
3. `test_librispeech_cuts_have_speaker(mock_librispeech)` — speaker = "1089"
4. `test_librispeech_subset_filter(mock_librispeech)` — `subsets=["train-clean-100"]` works; `subsets=["nonexistent"]` raises FileNotFoundError

### Commit: `feat(ingest): add LibriSpeech recipe`

---

## Task 3: CommonVoice recipe (TDD)

**Files:**
- Create: `src/voxkitchen/ingest/recipes/commonvoice.py`
- Create: `tests/unit/ingest/recipes/test_commonvoice.py`
- Modify: `src/voxkitchen/ingest/recipes/__init__.py`

### Mock fixture (in existing conftest.py)

```python
@pytest.fixture
def mock_commonvoice(tmp_path: Path) -> Path:
    clips = tmp_path / "clips"
    clips.mkdir()
    for name in ["cv_en_001.wav", "cv_en_002.wav"]:
        sf.write(clips / name, np.sin(np.linspace(0, 1, 16000)).astype(np.float32) * 0.5, 16000)

    # train.tsv (tab-separated, first line is header)
    (tmp_path / "train.tsv").write_text(
        "client_id\tpath\tsentence\tup_votes\tdown_votes\tage\tgender\taccent\tlocale\tsegment\n"
        "client1\tcv_en_001.wav\thello world\t5\t0\t\tmale_masculine\t\ten\t\n"
        "client2\tcv_en_002.wav\tgoodbye world\t3\t1\t\tfemale_feminine\t\ten\t\n"
    )
    return tmp_path
```

### CommonVoice recipe logic

Parse TSV files (`train.tsv`, `dev.tsv`, `test.tsv`). For each row: audio at `clips/<path>`, sentence, speaker (client_id), gender, locale.

**Config extends RecipeConfig with:** `locale: str | None = None` (optional filter by locale)

### Tests (3)
1. `test_commonvoice_recipe_is_registered`
2. `test_commonvoice_parses_mock_data(mock_commonvoice)` — 2 cuts from train.tsv
3. `test_commonvoice_cuts_have_gender_and_language(mock_commonvoice)` — verify supervision fields

### Commit: `feat(ingest): add CommonVoice recipe`

---

## Task 4: AISHELL-1 recipe (TDD)

**Files:**
- Create: `src/voxkitchen/ingest/recipes/aishell.py`
- Create: `tests/unit/ingest/recipes/test_aishell.py`
- Modify: `src/voxkitchen/ingest/recipes/__init__.py`

### Mock fixture

```python
@pytest.fixture
def mock_aishell(tmp_path: Path) -> Path:
    wav_dir = tmp_path / "data_aishell" / "wav" / "train" / "S0001"
    wav_dir.mkdir(parents=True)
    for name in ["BAC001.wav", "BAC002.wav"]:
        sf.write(wav_dir / name, np.sin(np.linspace(0, 1, 16000)).astype(np.float32) * 0.5, 16000)

    trans_dir = tmp_path / "data_aishell" / "transcript"
    trans_dir.mkdir(parents=True)
    (trans_dir / "aishell_transcript_v0.8.txt").write_text(
        "BAC001 你 好 世 界\n"
        "BAC002 再 见 世 界\n"
    )
    return tmp_path
```

### AISHELL-1 recipe logic

Walk `data_aishell/wav/<subset>/` dirs. Parse `aishell_transcript_v0.8.txt` into a `{utt_id: text}` dict (join characters with empty string to form natural text). Match audio files to transcripts.

### Tests (3)
1. `test_aishell_recipe_is_registered`
2. `test_aishell_parses_mock_data(mock_aishell)` — 2 cuts
3. `test_aishell_transcript_joined(mock_aishell)` — text is "你好世界" (space-separated chars joined)

### Commit: `feat(ingest): add AISHELL-1 recipe`

---

## Task 5: Integration test — recipe in a pipeline

**Files:**
- Create: `tests/integration/test_recipe_pipeline_e2e.py`

Test a pipeline that uses `source: recipe, recipe: librispeech`:

```python
def test_librispeech_recipe_pipeline(mock_librispeech: Path, tmp_path: Path) -> None:
    work_dir = tmp_path / "work"
    yaml_path = tmp_path / "pipeline.yaml"
    yaml_path.write_text(f"""
version: "0.1"
name: recipe-e2e
work_dir: {work_dir}
num_cpu_workers: 1
ingest:
  source: recipe
  recipe: librispeech
  args:
    root: {mock_librispeech}
    subsets: [train-clean-100]
stages:
  - name: pack
    op: pack_manifest
""")
    spec = load_pipeline_spec(yaml_path, run_id="run-recipe-e2e")
    run_pipeline(spec)

    assert (work_dir / "00_pack" / "_SUCCESS").exists()
    final_cuts = list(read_cuts(work_dir / "00_pack" / "cuts.jsonl.gz"))
    assert len(final_cuts) == 2
    assert any(c.supervisions[0].text == "HELLO WORLD" for c in final_cuts)
```

**Note:** The `mock_librispeech` fixture needs to be available in integration tests. Either move it to the top-level `conftest.py` or import it. Simplest: define it in `tests/conftest.py` (the root conftest).

### Commit: `test(integration): add recipe ingest pipeline test`

---

## Task 6: Full verification + tag

- [ ] `pytest -m "not slow and not gpu"` — all tests pass
- [ ] `ruff check src tests` + `ruff format --check`
- [ ] `mypy src/voxkitchen tests`
- [ ] `vkit validate` on a recipe pipeline YAML
- [ ] Tag: `git tag -a plan-06-ingest-recipes -m "Plan 6 complete: LibriSpeech, CommonVoice, AISHELL-1 recipes"`

---

## Plan 6 Completion Checklist

- [ ] `RecipeIngestSource` dispatches to recipes by name via `get_recipe()`
- [ ] `get_ingest_source("recipe")` returns `RecipeIngestSource`
- [ ] LibriSpeech recipe parses `.trans.txt` + `.flac` files correctly
- [ ] CommonVoice recipe parses TSV + clips correctly
- [ ] AISHELL-1 recipe parses transcript + wav correctly (characters joined into text)
- [ ] All 3 recipes registered in `ingest/recipes/__init__.py`
- [ ] Each Cut from a recipe has: embedded Recording, Supervision with text/speaker, provenance, `custom["subset"]`
- [ ] Tests use mock data (no real downloads)
- [ ] Integration test: recipe → pipeline → pack works
- [ ] All existing tests pass
- [ ] `git tag plan-06-ingest-recipes`

## What Plans 7-8 Will Build On

**Plan 7 (Visualization):** `vkit inspect cuts <manifest>` will display Cuts from any source — including recipe-ingested ones. The HTML report and Gradio panel will work on any CutSet.

**Plan 8 (Plugin + polish):** Recipes are first-party (hard-coded registration, not entry_points). The plugin system adds entry_points for third-party recipes. `vkit init` will offer recipe choices when scaffolding a new pipeline project.
