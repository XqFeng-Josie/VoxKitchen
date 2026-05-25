"""Unit tests for normalize_text operator."""

from __future__ import annotations

from datetime import datetime, timezone

from voxkitchen.operators.annotate.normalize_text import (
    NormalizeTextConfig,
    NormalizeTextOperator,
    _normalize,
)


def test_strips_sensevoice_tags():
    assert _normalize("<|zh|><|HAPPY|><|Speech|>你好", strip_tags=True,
                       collapse_spaces=True, lowercase=False) == "你好"


def test_collapses_paraformer_inter_char_spaces():
    assert _normalize("你 好 世 界", strip_tags=True, collapse_spaces=True,
                      lowercase=False) == "你好世界"


def test_lowercase_english_keeps_word_spaces():
    assert _normalize("Hello  World", strip_tags=True, collapse_spaces=True,
                      lowercase=True) == "hello world"


def test_operator_rewrites_supervision_text(make_run_context):
    from voxkitchen.schema.cut import Cut
    from voxkitchen.schema.cutset import CutSet
    from voxkitchen.schema.provenance import Provenance
    from voxkitchen.schema.supervision import Supervision

    ctx = make_run_context()
    provenance = Provenance(
        source_cut_id=None,
        generated_by="fixture",
        stage_name="00_ingest",
        created_at=datetime(2026, 4, 11, tzinfo=timezone.utc),
        pipeline_run_id="run-test",
    )
    cut = Cut(
        id="c1",
        recording_id="r1",
        start=0.0,
        duration=1.0,
        supervisions=[
            Supervision(
                id="s1",
                recording_id="r1",
                start=0.0,
                duration=1.0,
                text="<|zh|>你 好",
            )
        ],
        provenance=provenance,
    )
    op = NormalizeTextOperator(NormalizeTextConfig(), ctx=ctx)
    out = list(op.process(CutSet([cut])))
    assert out[0].supervisions[0].text == "你好"
