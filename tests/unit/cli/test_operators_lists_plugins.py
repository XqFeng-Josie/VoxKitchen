def test_operators_list_includes_plugin(fake_operator_entry_point, capsys):
    from voxkitchen.cli.operators_cmd import _render_table

    _render_table()
    out = capsys.readouterr().out
    assert "word_count" in out


def test_operators_search_finds_plugin(fake_operator_entry_point, capsys):
    import typer
    from voxkitchen.cli.operators_cmd import _render_table

    try:
        _render_table(keyword="word_count")
    except typer.Exit as exc:
        raise AssertionError(f"search unexpectedly exited: {exc}") from exc
    out = capsys.readouterr().out
    assert "word_count" in out
