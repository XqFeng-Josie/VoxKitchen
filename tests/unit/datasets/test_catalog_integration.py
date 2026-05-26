from pathlib import Path

from voxkitchen.datasets.catalog import load_catalog
from voxkitchen.datasets.catalog_gen import build_download_info, check_docs
from voxkitchen.pipeline.loader import load_pipeline_spec
from voxkitchen.pipeline.preflight import preflight_spec

_REPO = Path(__file__).resolve().parents[3]


def test_real_catalog_loads_and_has_recipe_coverage():
    entries = load_catalog()
    ids = {e.id for e in entries}
    for r in ["librispeech", "libritts", "ljspeech", "aishell", "aishell3",
              "cnceleb", "commonvoice", "fleurs", "musan"]:
        assert r in ids, f"recipe {r} missing from catalog"
    assert len(entries) >= 15


def test_every_recommended_pipeline_passes_preflight():
    for e in load_catalog():
        if e.recommended_pipeline:
            spec = load_pipeline_spec(_REPO / e.recommended_pipeline)
            assert preflight_spec(spec).errors == [], f"{e.id}: {e.recommended_pipeline}"


def test_generated_docs_up_to_date():
    entries = load_catalog()
    drift = check_docs(entries, build_download_info(entries), _REPO / "docs" / "datasets")
    assert drift == [], f"stale docs: {drift} — run python -m voxkitchen.datasets.catalog_gen"
