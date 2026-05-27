import sys
from pathlib import Path

import pytest

# tests/unit/conftest.py -> parents[2] is the repo root.
_EXAMPLE = Path(__file__).resolve().parents[2] / "examples" / "plugin-operator"


@pytest.fixture
def example_operator_cls():
    """Import the example plugin's WordCountOperator (undecorated, no side effects)."""
    if str(_EXAMPLE) not in sys.path:
        sys.path.insert(0, str(_EXAMPLE))
    from voxkitchen_example_plugin.operator import WordCountOperator

    return WordCountOperator


@pytest.fixture
def fake_operator_entry_point(monkeypatch, example_operator_cls):
    """Make importlib.metadata.entry_points yield a synthetic operator-plugin EP.

    Restores the operator registry and discovery module state afterwards so the
    global registry is not polluted for other tests.
    """
    import importlib.metadata as importlib_metadata

    import voxkitchen.plugins.discovery as disc
    from voxkitchen.operators import registry

    class _FakeEP:
        name = "word_count"

        def load(self):
            return example_operator_cls

    real_entry_points = importlib_metadata.entry_points

    def fake_entry_points(*args, **kwargs):
        group = kwargs.get("group")
        if group == "voxkitchen.operators":
            return [_FakeEP()]
        if group == "voxkitchen.recipes":
            return []
        return real_entry_points(*args, **kwargs)

    monkeypatch.setattr(importlib_metadata, "entry_points", fake_entry_points)

    before = set(registry._REGISTRY)
    disc._loaded = False
    disc.discovered_operators.clear()
    yield
    for key in set(registry._REGISTRY) - before:
        del registry._REGISTRY[key]
    disc._loaded = False
    disc.discovered_operators.clear()
