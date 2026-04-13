"""Recipe registry for dataset-specific ingest."""

from __future__ import annotations

from voxkitchen.ingest.recipes.base import Recipe

_RECIPES: dict[str, Recipe] = {}


def register_recipe(recipe: Recipe) -> Recipe:
    if not recipe.name:
        raise ValueError(f"{type(recipe).__name__} must have a non-empty name")
    if recipe.name in _RECIPES:
        raise ValueError(f"recipe {recipe.name!r} already registered")
    _RECIPES[recipe.name] = recipe
    return recipe


def get_recipe(name: str) -> Recipe:
    if name not in _RECIPES:
        available = sorted(_RECIPES.keys()) or ["(none — no recipes registered)"]
        raise KeyError(f"recipe {name!r} not found. Available: {available}")
    return _RECIPES[name]


def _load_builtin_recipes() -> None:
    """Import built-in recipe modules to trigger their register_recipe() calls."""
    from voxkitchen.ingest.recipes import aishell, commonvoice, librispeech  # noqa: F401


_load_builtin_recipes()

__all__ = ["Recipe", "get_recipe", "register_recipe"]
