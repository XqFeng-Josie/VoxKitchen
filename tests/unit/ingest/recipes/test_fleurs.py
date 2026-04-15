"""Unit tests for FLEURS recipe."""

from __future__ import annotations

import pytest

try:
    from datasets import Dataset  # noqa: F401
except ImportError:
    pytest.skip("datasets library not available", allow_module_level=True)

from voxkitchen.ingest.recipes import get_recipe


def test_fleurs_recipe_is_registered() -> None:
    recipe = get_recipe("fleurs")
    assert recipe.name == "fleurs"


def test_fleurs_recipe_has_no_download_urls() -> None:
    """FLEURS uses custom download() via HuggingFace, not download_urls."""
    recipe = get_recipe("fleurs")
    assert recipe.download_urls == {}
