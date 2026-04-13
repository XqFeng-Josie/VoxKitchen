"""Tests for plugin discovery via entry_points."""

from __future__ import annotations

import voxkitchen.plugins.discovery as _disc_module
from voxkitchen.plugins.discovery import load_plugins


def test_load_plugins_is_idempotent() -> None:
    load_plugins()
    load_plugins()  # second call should not raise


def test_builtin_operators_still_work_after_plugin_load() -> None:
    from voxkitchen.operators.registry import get_operator

    load_plugins()
    op = get_operator("identity")
    assert op.name == "identity"


def test_get_operator_triggers_plugin_loading() -> None:
    from voxkitchen.operators.registry import get_operator

    # Reset so we can observe the trigger
    _disc_module._loaded = False
    get_operator("identity")  # should not raise, triggers load_plugins internally
    assert _disc_module._loaded is True
