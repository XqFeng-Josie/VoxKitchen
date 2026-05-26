from pathlib import Path

import pytest
from voxkitchen.pipeline.loader import load_pipeline_spec
from voxkitchen.pipeline.preflight import preflight_spec

TEMPLATE_DIR = Path("voxkitchen/templates/pipelines")
TEMPLATES = sorted(TEMPLATE_DIR.glob("*.yaml"))


@pytest.mark.parametrize("path", TEMPLATES, ids=lambda p: p.stem)
def test_template_passes_preflight(path):
    spec = load_pipeline_spec(path)
    result = preflight_spec(spec)
    assert result.errors == [], f"{path.name}: {result.errors}"
