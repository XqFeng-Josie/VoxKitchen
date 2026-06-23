from pathlib import Path

import pytest
import yaml
from voxkitchen.pipeline.loader import load_pipeline_spec
from voxkitchen.pipeline.preflight import preflight_spec

EXAMPLES = sorted(Path("examples/pipelines").glob("*.yaml"))
DEMO_PIPELINE = Path("examples/pipelines/demo-no-asr.yaml")
DEMO_DATA_DIR = Path("examples/demo_data")


@pytest.mark.parametrize("path", EXAMPLES, ids=lambda p: p.stem)
def test_example_passes_preflight(path):
    spec = load_pipeline_spec(path)
    result = preflight_spec(spec)
    assert result.errors == [], f"{path.name}: {result.errors}"


def test_quickstart_demo_contract_stays_stable():
    """Keep README/getting-started demo commands tied to real bundled assets."""
    data = yaml.safe_load(DEMO_PIPELINE.read_text(encoding="utf-8"))

    assert data["name"] == "demo-no-asr"
    assert data["work_dir"] == "./work/${name}"
    assert data["ingest"]["source"] == "dir"
    assert data["ingest"]["args"]["root"] == "./examples/demo_data"
    assert DEMO_DATA_DIR.is_dir()
    assert any(DEMO_DATA_DIR.glob("*.opus"))
