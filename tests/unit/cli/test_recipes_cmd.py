"""Tests for the `vkit recipes` listing command."""

from __future__ import annotations

from typer.testing import CliRunner
from voxkitchen.cli.main import app
from voxkitchen.cli.recipes_cmd import _format_size_column


def test_recipes_listing_lists_all_known_recipes() -> None:
    """Every registered recipe should appear in `vkit recipes` output."""
    result = CliRunner().invoke(app, ["recipes"])
    assert result.exit_code == 0
    for name in (
        "aishell",
        "aishell3",
        "cnceleb",
        "commonvoice",
        "fleurs",
        "librispeech",
        "libritts",
        "ljspeech",
        "musan",
    ):
        assert name in result.output, f"recipe {name!r} missing from `vkit recipes`"


def test_recipes_listing_shows_size_column() -> None:
    """The listing must include a Size column with a recipe's compressed size.

    Users need to see "11 GB" before they type `vkit docker download musan`;
    that's the whole point of surfacing download_sizes through the CLI.
    """
    result = CliRunner().invoke(app, ["recipes"])
    assert result.exit_code == 0
    assert "Size" in result.output
    # Single-archive recipes show a single size value.
    assert "10.3 GB" in result.output  # musan
    assert "20.7 GB" in result.output  # cnceleb


def test_format_size_column_single_subset() -> None:
    """Single-subset recipes render a bare size (no range)."""
    assert _format_size_column({"musan": 11_086_114_085}) == "10.3 GB"


def test_format_size_column_multi_subset_range() -> None:
    """Multi-subset recipes render a min-max range so users see both ends.

    LibriSpeech subsets span dev-other (~300 MB) to train-other-500 (~28 GB);
    a single number would obscure either the cheap dev option or the
    expensive train option.
    """
    out = _format_size_column(
        {
            "dev-other": 314_305_928,
            "train-other-500": 30_593_501_606,
        }
    )
    assert "299 MB" in out
    assert "28.5 GB" in out
    assert "-" in out  # range separator


def test_format_size_column_empty_dict_shows_dash() -> None:
    """Manual / HF-streaming recipes (no static sizes) render as a dim dash.

    Avoids misleading users that the size is "0 bytes" or unknown-by-bug;
    the dim dash reads as "no value applicable".
    """
    assert "-" in _format_size_column({})
