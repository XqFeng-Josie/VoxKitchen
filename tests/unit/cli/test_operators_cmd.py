"""Unit tests for `vkit operators` subcommands."""

from __future__ import annotations

from typer.testing import CliRunner
from voxkitchen.cli.main import app


def test_operators_show_highlights_pack_huggingface_audio_decode_warning() -> None:
    result = CliRunner().invoke(app, ["operators", "show", "pack_huggingface"])

    assert result.exit_code == 0
    assert "Warning:" in result.output
    assert "torchcodec" in result.output
    assert "Audio(decode=False)" in result.output


def test_operators_default_lists_all_categories() -> None:
    result = CliRunner().invoke(app, ["operators"])

    assert result.exit_code == 0
    # Spot-check headers from several categories — proves we didn't break the
    # default category-grouped listing while adding filters.
    assert "Audio" in result.output
    assert "Quality" in result.output
    assert "Pack" in result.output


def test_operators_category_filter_restricts_to_one_section() -> None:
    result = CliRunner().invoke(app, ["operators", "--category", "quality"])

    assert result.exit_code == 0
    assert "Operators in 'Quality'" in result.output
    # A known quality operator must appear; an obvious non-quality one must not.
    assert "snr_estimate" in result.output
    assert "resample" not in result.output


def test_operators_unknown_category_errors_with_exit_code_2() -> None:
    result = CliRunner().invoke(app, ["operators", "--category", "not_a_category"])

    assert result.exit_code == 2
    assert "unknown category" in result.output


def test_operators_search_matches_name_and_description() -> None:
    result = CliRunner().invoke(app, ["operators", "search", "noise"])

    assert result.exit_code == 0
    assert "Operators matching 'noise'" in result.output
    # `noise_augment` matches by name; `speech_enhance` matches by description.
    assert "noise_augment" in result.output
    assert "speech_enhance" in result.output


def test_operators_search_empty_result_exits_1() -> None:
    result = CliRunner().invoke(app, ["operators", "search", "qzbnxx_unmatched_kw"])

    assert result.exit_code == 1
    assert "no operators match" in result.output


def test_operators_search_is_case_insensitive() -> None:
    lower = CliRunner().invoke(app, ["operators", "search", "noise"])
    upper = CliRunner().invoke(app, ["operators", "search", "NOISE"])

    assert lower.exit_code == 0
    assert upper.exit_code == 0
    # Same set of matches regardless of case — we lowercase the needle.
    for op in ("noise_augment", "speech_enhance"):
        assert op in lower.output
        assert op in upper.output
