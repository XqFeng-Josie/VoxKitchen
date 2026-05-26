from pathlib import Path

import pytest
from voxkitchen.pipeline.loader import load_pipeline_spec
from voxkitchen.pipeline.preflight import preflight_spec

EXAMPLES = sorted(Path("examples/pipelines").glob("*.yaml"))


@pytest.mark.parametrize("path", EXAMPLES, ids=lambda p: p.stem)
def test_example_passes_preflight(path):
    spec = load_pipeline_spec(path)
    result = preflight_spec(spec)
    assert result.errors == [], f"{path.name}: {result.errors}"
