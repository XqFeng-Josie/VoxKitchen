"""Tests that vkit doctor includes third-party operators and reports API version."""

from __future__ import annotations


def test_collect_available_includes_plugins(fake_operator_entry_point):
    from voxkitchen.cli.doctor import _collect_available

    available = _collect_available()
    assert "word_count" in available


def test_emit_table_shows_api_version(capsys):
    from voxkitchen.cli.doctor import _emit_table

    _emit_table(None, {"identity"}, set(), [], [], None)
    err = capsys.readouterr().err
    assert "API version" in err
