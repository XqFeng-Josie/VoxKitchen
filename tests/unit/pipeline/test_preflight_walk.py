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
    # supervisions.text (the ASR hypothesis) is a hard required read — missing it
    # is always an error.  custom.reference_text is now optional_reads, so a
    # missing one becomes a WARNING, not an error.
    spec = _spec(
        StageSpec(name="cer", op="cer_wer"),
        StageSpec(name="pack", op="pack_jsonl"),
    )
    result = preflight_spec(spec)
    assert any("supervisions.text" in e for e in result.errors)
    assert any("cer" in e for e in result.errors)
    assert any("custom.reference_text" in w for w in result.warnings)


def test_filter_referencing_absent_metric_is_error():
    spec = _spec(
        StageSpec(
            name="filter", op="quality_score_filter", args={"conditions": ["metrics.snr > 10"]}
        ),
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
        StageSpec(
            name="filter", op="quality_score_filter", args={"conditions": ["metrics.snr > 10"]}
        ),
        StageSpec(name="pack", op="pack_jsonl"),
    )
    assert preflight_spec(spec).errors == []


# ---------------------------------------------------------------------------
# Ingest-source seed tests
# ---------------------------------------------------------------------------


def _spec_with_ingest(source, *stages):
    kwargs = {}
    if source == "recipe":
        kwargs["recipe"] = "ljspeech"
    return PipelineSpec(
        version="0.1",
        name="t",
        work_dir="./work",
        ingest=IngestSpec(source=source, args={}, **kwargs),
        stages=list(stages),
    )


def test_manifest_ingest_supplies_supervision_text():
    # forced_align needs supervisions.text; with manifest ingest it's assumed present
    spec = _spec_with_ingest(
        "manifest",
        StageSpec(name="align", op="forced_align"),
        StageSpec(name="pack", op="pack_jsonl"),
    )
    assert preflight_spec(spec).errors == []


def test_recipe_ingest_supplies_supervision_text():
    spec = _spec_with_ingest(
        "recipe",
        StageSpec(name="tts", op="tts_kokoro"),
    )
    assert preflight_spec(spec).errors == []


def test_dir_ingest_still_requires_text_for_forced_align():
    spec = _spec_with_ingest(
        "dir",
        StageSpec(name="align", op="forced_align"),
        StageSpec(name="pack", op="pack_jsonl"),
    )
    assert any("supervisions.text" in e for e in preflight_spec(spec).errors)


def test_dir_with_reference_glob_seeds_reference_text():
    # A dir ingest with reference_text_glob set seeds custom.reference_text, so
    # a dir → faster_whisper_asr → cer_wer → pack_jsonl pipeline must be clean.
    spec = PipelineSpec(
        version="0.1",
        name="t",
        work_dir="./work",
        ingest=IngestSpec(source="dir", args={"root": ".", "reference_text_glob": "*.txt"}),
        stages=[
            StageSpec(name="asr", op="faster_whisper_asr"),
            StageSpec(name="cer", op="cer_wer"),
            StageSpec(name="pack", op="pack_jsonl"),
        ],
    )
    result = preflight_spec(spec)
    assert result.errors == []
    assert not any("custom.reference_text" in w for w in result.warnings)


def test_manifest_ingest_seeds_metrics_and_custom():
    # A manifest loads a previously-built CutSet that may already carry computed
    # metrics/custom; re-filtering on them must NOT false-error.
    spec = _spec_with_ingest(
        "manifest",
        StageSpec(
            name="filter", op="quality_score_filter", args={"conditions": ["metrics.snr > 10"]}
        ),
        StageSpec(name="pack", op="pack_jsonl"),
    )
    assert preflight_spec(spec).errors == []


def test_recipe_ingest_does_not_seed_metrics():
    # Recipes parse datasets (text/speaker/language) but never produce computed
    # metrics, so filtering on an unproduced metric IS a real broken chain.
    spec = _spec_with_ingest(
        "recipe",
        StageSpec(
            name="filter", op="quality_score_filter", args={"conditions": ["metrics.snr > 10"]}
        ),
        StageSpec(name="pack", op="pack_jsonl"),
    )
    assert any("metrics.snr" in e for e in preflight_spec(spec).errors)
