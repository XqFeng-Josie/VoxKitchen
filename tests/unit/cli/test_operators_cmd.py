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
