"""Tests for ``vkit datasets`` — terminal browser for the dataset catalog.

We invoke the Typer commands directly (the same pattern as the other
``cli/test_*`` files in this repo) rather than spawning subprocesses, so
the tests stay fast and don't depend on the ``vkit`` console-script being
installed.
"""

from __future__ import annotations

import pytest
import typer
from typer.testing import CliRunner
from voxkitchen.cli.datasets_cmd import _filter_entries, datasets_app
from voxkitchen.datasets.catalog import load_catalog


def _runner() -> CliRunner:
    # Plain Typer CliRunner: it captures combined output via ``result.output``;
    # the ``mix_stderr`` toggle was removed in Click 8.2 / Typer 0.13.
    return CliRunner()


# ---- filter helper (pure function, no CLI surface) -------------------------


def test_filter_entries_task() -> None:
    """``--task asr`` returns only entries that include 'asr' in their task list."""
    entries = load_catalog()
    out = _filter_entries(entries, task="asr", language=None, recipe_only=False, query=None)
    assert out
    assert all("asr" in e.task for e in out)
    # Single-task entries (e.g. voxceleb1 = [speaker]) must be excluded.
    assert not any(e.id == "voxceleb1" for e in out)


def test_filter_entries_language() -> None:
    """``--language zh`` returns only entries whose language list includes 'zh'."""
    entries = load_catalog()
    out = _filter_entries(entries, task=None, language="zh", recipe_only=False, query=None)
    assert out
    assert all("zh" in e.languages for e in out)


def test_filter_entries_recipe_only() -> None:
    """``--recipe-only`` keeps only entries with a non-None recipe."""
    entries = load_catalog()
    out = _filter_entries(entries, task=None, language=None, recipe_only=True, query=None)
    assert out
    assert all(e.recipe is not None for e in out)
    # We know batch-1 added 4 → 13 total recipe-backed entries today.
    assert len(out) == 13


def test_filter_entries_query_matches_id_and_summary() -> None:
    """Substring query is case-insensitive and matches across id / name / summary."""
    entries = load_catalog()
    # 'libri' matches librispeech / libritts / libritts_r / libri_light.
    out = _filter_entries(entries, task=None, language=None, recipe_only=False, query="LIBRI")
    ids = {e.id for e in out}
    assert {"librispeech", "libritts", "libritts_r", "libri_light"}.issubset(ids)


def test_filter_entries_composes_multiple_filters() -> None:
    """Filters are AND-composed: task=tts + language=zh + recipe-only → narrow set."""
    entries = load_catalog()
    out = _filter_entries(entries, task="tts", language="zh", recipe_only=True, query=None)
    assert out
    for e in out:
        assert "tts" in e.task
        assert "zh" in e.languages
        assert e.recipe is not None


# ---- CLI surface ------------------------------------------------------------


def test_list_default_runs_clean() -> None:
    """No-arg `vkit datasets` prints the table for all 60 entries and exits 0."""
    result = _runner().invoke(datasets_app, [])
    assert result.exit_code == 0, result.stdout
    assert "60 entries" in result.stdout


def test_list_with_task_and_recipe_only_filter() -> None:
    result = _runner().invoke(datasets_app, ["--task", "tts", "--recipe-only"])
    assert result.exit_code == 0, result.stdout
    # Recipe-backed TTS entries today: libritts, ljspeech, aishell3, libritts_r,
    # hifitts, thorsten_voice — so 6.
    assert "6 entries" in result.stdout


def test_list_no_matches_prints_helpful_message() -> None:
    """Empty filter result → friendly yellow notice, exit code 0 (not an error)."""
    result = _runner().invoke(datasets_app, ["--query", "no_such_dataset_xyz_zzy"])
    assert result.exit_code == 0
    assert "no entries match" in result.stdout.lower()


def test_show_existing_id_prints_panel() -> None:
    result = _runner().invoke(datasets_app, ["show", "librispeech"])
    assert result.exit_code == 0, result.stdout
    # Spot-check that the panel contains the canonical fields.
    assert "LibriSpeech" in result.stdout
    assert "Recommendation" in result.stdout
    assert "openslr.org/12" in result.stdout
    # Recipe-backed entry must surface the download hint.
    assert "vkit docker download librispeech" in result.stdout


def test_show_recipe_hint_only_when_recipe_present() -> None:
    """A manual-access entry must NOT advertise a `vkit docker download` command."""
    result = _runner().invoke(datasets_app, ["show", "voxpopuli"])
    assert result.exit_code == 0
    assert "vkit docker download" not in result.stdout


def test_show_unknown_id_exits_1_with_hint() -> None:
    result = _runner().invoke(datasets_app, ["show", "this_id_does_not_exist"])
    assert result.exit_code == 1
    assert "no catalog entry" in result.stdout.lower()
    assert "vkit datasets search" in result.stdout


def test_search_substring_match() -> None:
    """`vkit datasets search libri` finds the LibriSpeech/LibriTTS/LibriTTS-R family."""
    result = _runner().invoke(datasets_app, ["search", "libri"])
    assert result.exit_code == 0, result.stdout
    for needle in ["librispeech", "libritts", "libritts_r", "libri_light"]:
        assert needle in result.stdout, f"missing {needle} in search output"


def test_search_no_match_returns_helpful_message() -> None:
    result = _runner().invoke(datasets_app, ["search", "zzzzzzzz_no_match"])
    assert result.exit_code == 0
    assert "no entries match" in result.stdout.lower()


def test_search_composes_with_filters() -> None:
    """`vkit datasets search asian --task asr` narrows further than search alone."""
    result = _runner().invoke(datasets_app, ["search", "speech", "--language", "zh"])
    assert result.exit_code == 0
    # All zh entries with 'speech' in id/name/summary; whichever those are,
    # there must be at least one, and none must be an English-only entry.
    assert "match(es)" in result.stdout
    assert "librispeech" not in result.stdout  # en, not zh


def test_load_catalog_works_when_pyproject_absent(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    """When `_REPO_ROOT/pyproject.toml` is missing (= wheel install layout),
    `load_catalog()` must skip the recommended_pipeline existence check and
    still succeed."""
    import voxkitchen.datasets.catalog as cat_mod

    # Point _REPO_ROOT at an empty temp dir — same effect as running from
    # site-packages where pyproject.toml is not present.
    monkeypatch.setattr(cat_mod, "_REPO_ROOT", tmp_path)
    entries = cat_mod.load_catalog()
    assert len(entries) == 60


def test_datasets_app_mounted_on_vkit_root() -> None:
    """`vkit datasets ...` must dispatch via the main `vkit` Typer app too."""
    from voxkitchen.cli.main import app as vkit_app

    result = CliRunner().invoke(vkit_app, ["datasets", "--task", "augmentation"])
    assert result.exit_code == 0, result.stdout
    # MUSAN is the only augmentation entry.
    assert "musan" in result.stdout.lower()


# Silence "unused import" linters for typer in environments where the
# decorator-only style hides the usage.
_unused_typer_marker = typer.Typer
