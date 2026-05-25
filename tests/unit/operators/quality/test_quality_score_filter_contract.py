from voxkitchen.operators.quality.quality_score_filter import (
    QualityScoreFilterConfig,
    QualityScoreFilterOperator,
)


def test_dynamic_reads_extracts_metric_paths():
    cfg = QualityScoreFilterConfig(conditions=["metrics.snr > 10", "duration > 0.5"])
    op = QualityScoreFilterOperator(cfg, ctx=None)
    # duration is intrinsic and not tracked; only metrics/custom/supervisions paths surface
    assert op.dynamic_reads() == ["metrics.snr"]


def test_dynamic_reads_dedupes_and_keeps_custom_and_supervisions():
    cfg = QualityScoreFilterConfig(conditions=[
        "metrics.snr > 10", "metrics.snr < 40",
        "custom.reference_text != x", "supervisions.text == y",
    ])
    op = QualityScoreFilterOperator(cfg, ctx=None)
    assert op.dynamic_reads() == ["metrics.snr", "custom.reference_text", "supervisions.text"]
