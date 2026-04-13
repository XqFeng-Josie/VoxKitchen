"""Tests for RecipeIngestSource dispatch."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest
from pydantic import ValidationError

from voxkitchen.ingest import get_ingest_source
from voxkitchen.ingest.recipe_source import RecipeConfig, RecipeIngestSource
from voxkitchen.ingest.recipes import _RECIPES, register_recipe
from voxkitchen.ingest.recipes.base import Recipe
from voxkitchen.pipeline.context import RunContext
from voxkitchen.schema.cutset import CutSet


def test_recipe_ingest_source_is_registered() -> None:
    assert get_ingest_source("recipe") is RecipeIngestSource


def test_recipe_config_requires_recipe_and_root() -> None:
    with pytest.raises(ValidationError):
        RecipeConfig.model_validate({"root": "/tmp"})  # missing recipe
    with pytest.raises(ValidationError):
        RecipeConfig.model_validate({"recipe": "x"})  # missing root


def test_recipe_dispatch_calls_prepare(tmp_path: Path) -> None:
    """Register a mock recipe, verify RecipeIngestSource dispatches to it."""
    mock_recipe = MagicMock(spec=Recipe)
    mock_recipe.name = "_test_mock"
    mock_recipe.prepare.return_value = CutSet([])

    saved = dict(_RECIPES)
    try:
        register_recipe(mock_recipe)
        ctx = RunContext(
            work_dir=tmp_path,
            pipeline_run_id="run-test",
            stage_index=0,
            stage_name="ingest",
            num_gpus=0,
            num_cpu_workers=1,
            gc_mode="aggressive",
            device="cpu",
        )
        source = RecipeIngestSource(
            RecipeConfig(recipe="_test_mock", root=str(tmp_path)),
            ctx=ctx,
        )
        result = source.run()
        assert len(result) == 0
        mock_recipe.prepare.assert_called_once()
    finally:
        _RECIPES.clear()
        _RECIPES.update(saved)
