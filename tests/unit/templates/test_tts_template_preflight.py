from pathlib import Path

from voxkitchen.pipeline.loader import load_pipeline_spec
from voxkitchen.pipeline.preflight import preflight_spec

TEMPLATE = Path("voxkitchen/templates/pipelines/tts-data-prep.yaml")


def test_tts_template_has_forced_align_and_normalize():
    spec = load_pipeline_spec(TEMPLATE)
    ops = [s.op for s in spec.stages]
    assert "forced_align" in ops
    assert "normalize_text" in ops
    # forced_align must come after the ASR stage (which writes supervisions.text)
    assert ops.index("forced_align") > ops.index("qwen3_asr")


def test_tts_template_passes_preflight():
    spec = load_pipeline_spec(TEMPLATE)
    assert preflight_spec(spec).errors == []
