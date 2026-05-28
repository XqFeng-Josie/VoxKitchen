import textwrap
from pathlib import Path

import pytest
from voxkitchen.datasets.catalog import CatalogError, load_catalog

_VALID = """
entries:
  - id: librispeech
    name: LibriSpeech
    task: [asr]
    languages: [en]
    license: CC BY 4.0
    summary: Read English audiobooks.
    homepage: https://www.openslr.org/12
    recommendation: Standard English ASR benchmark.
    recipe: librispeech
  - id: gigaspeech
    name: GigaSpeech
    task: [asr]
    languages: [en]
    license: see source terms
    summary: 10k h English ASR corpus.
    homepage: https://github.com/SpeechColab/GigaSpeech
    recommendation: Large, diverse English ASR; needs access request.
    notes: Request access on the project page.
"""


def _write(tmp_path: Path, text: str) -> Path:
    p = tmp_path / "catalog.yaml"
    p.write_text(textwrap.dedent(text), encoding="utf-8")
    return p


def test_loads_valid_catalog(tmp_path):
    entries = load_catalog(_write(tmp_path, _VALID))
    assert [e.id for e in entries] == ["librispeech", "gigaspeech"]


def test_duplicate_id_rejected(tmp_path):
    dup = (
        _VALID
        + """
  - id: librispeech
    name: Dup
    task: [asr]
    languages: [en]
    license: L
    summary: s
    homepage: h
    recommendation: r
"""
    )
    with pytest.raises(CatalogError, match="duplicate"):
        load_catalog(_write(tmp_path, dup))


def test_unknown_recipe_rejected(tmp_path):
    bad = _VALID.replace("recipe: librispeech", "recipe: not_a_recipe")
    with pytest.raises(CatalogError, match="recipe"):
        load_catalog(_write(tmp_path, bad))


def test_missing_pipeline_file_rejected(tmp_path):
    bad = (
        _VALID
        + """
  - id: ljspeech
    name: LJSpeech
    task: [tts]
    languages: [en]
    license: public domain
    summary: s
    homepage: h
    recommendation: r
    recommended_pipeline: examples/pipelines/does-not-exist.yaml
"""
    )
    with pytest.raises(CatalogError, match="recommended_pipeline"):
        load_catalog(_write(tmp_path, bad))
