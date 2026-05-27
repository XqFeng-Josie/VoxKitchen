def test_load_plugins_registers_entrypoint_operator(fake_operator_entry_point):
    import voxkitchen.plugins.discovery as disc
    from voxkitchen.operators.registry import _REGISTRY

    disc.load_plugins()
    assert "word_count" in _REGISTRY
    assert "word_count" in disc.discovered_operators


def test_discovered_operators_reset_by_fixture(fake_operator_entry_point):
    import voxkitchen.plugins.discovery as disc

    assert disc.discovered_operators == []  # fixture cleared it before yielding
