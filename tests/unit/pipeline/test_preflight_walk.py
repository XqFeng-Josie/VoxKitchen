from voxkitchen.pipeline.preflight import PreflightResult, preflight_spec
from voxkitchen.pipeline.spec import IngestSpec, PipelineSpec, StageSpec


def _spec(*stages: StageSpec) -> PipelineSpec:
    return PipelineSpec(
        version="0.1",
        name="t",
        work_dir="./work",
        ingest=IngestSpec(source="dir", args={"root": "."}),
        stages=list(stages),
    )


def test_good_chain_has_no_errors():
    spec = _spec(
        StageSpec(name="vad", op="silero_vad"),
        StageSpec(name="asr", op="faster_whisper_asr"),
        StageSpec(name="pack", op="pack_jsonl"),
    )
    result = preflight_spec(spec)
    assert isinstance(result, PreflightResult)
    assert result.errors == []


def test_missing_text_for_cer_wer_is_error():
    spec = _spec(
        StageSpec(name="cer", op="cer_wer"),
        StageSpec(name="pack", op="pack_jsonl"),
    )
    result = preflight_spec(spec)
    assert any("supervisions.text" in e for e in result.errors)
    assert any("custom.reference_text" in e for e in result.errors)
    assert any("cer" in e for e in result.errors)


def test_filter_referencing_absent_metric_is_error():
    spec = _spec(
        StageSpec(name="filter", op="quality_score_filter",
                  args={"conditions": ["metrics.snr > 10"]}),
        StageSpec(name="pack", op="pack_jsonl"),
    )
    result = preflight_spec(spec)
    assert any("metrics.snr" in e for e in result.errors)


def test_forced_align_without_text_is_error():
    # forced_align lists supervisions.text in `reads`, so with no upstream ASR
    # this is a hard error.
    spec = _spec(
        StageSpec(name="align", op="forced_align"),
        StageSpec(name="pack", op="pack_jsonl"),
    )
    result = preflight_spec(spec)
    assert any("supervisions.text" in e for e in result.errors)


def test_snr_filter_chain_ok():
    spec = _spec(
        StageSpec(name="snr", op="snr_estimate"),
        StageSpec(name="filter", op="quality_score_filter",
                  args={"conditions": ["metrics.snr > 10"]}),
        StageSpec(name="pack", op="pack_jsonl"),
    )
    assert preflight_spec(spec).errors == []
