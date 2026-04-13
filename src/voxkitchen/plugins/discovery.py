"""Lazy entry_points discovery for third-party operators and recipes."""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

_loaded = False


def load_plugins() -> None:
    global _loaded
    if _loaded:
        return
    _loaded = True

    from importlib.metadata import entry_points

    from voxkitchen.operators.registry import register_operator

    for ep in entry_points(group="voxkitchen.operators"):
        try:
            op_cls = ep.load()
            register_operator(op_cls)
            logger.debug("loaded operator plugin: %s", ep.name)
        except Exception:
            logger.warning("failed to load operator plugin: %s", ep.name, exc_info=True)

    from voxkitchen.ingest.recipes import register_recipe

    for ep in entry_points(group="voxkitchen.recipes"):
        try:
            recipe = ep.load()
            register_recipe(recipe)
            logger.debug("loaded recipe plugin: %s", ep.name)
        except Exception:
            logger.warning("failed to load recipe plugin: %s", ep.name, exc_info=True)
