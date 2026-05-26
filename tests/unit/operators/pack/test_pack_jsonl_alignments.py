import json
from datetime import datetime, timezone

from voxkitchen.operators.pack.pack_jsonl import PackJsonlConfig, PackJsonlOperator
from voxkitchen.schema.cut import Cut
from voxkitchen.schema.cutset import CutSet
from voxkitchen.schema.provenance import Provenance
from voxkitchen.schema.supervision import Supervision


def _provenance() -> Provenance:
    return Provenance(
        source_cut_id=None,
        generated_by="fixture",
        stage_name="00_ingest",
        created_at=datetime(2026, 4, 11, tzinfo=timezone.utc),
        pipeline_run_id="run-test",
    )


def _cut(custom=None):
    return Cut(
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
                text="hi",
            )
        ],
        custom=custom or {},
        provenance=_provenance(),
    )


def test_word_alignments_exported_when_present(make_run_context, tmp_path):
    ctx = make_run_context()
    out = tmp_path / "m.jsonl"
    cut = _cut(custom={"word_alignments": [{"word": "hi", "start": 0.0, "end": 0.5}]})
    op = PackJsonlOperator(PackJsonlConfig(output_path=str(out)), ctx=ctx)
    list(op.process(CutSet([cut])))
    row = json.loads(out.read_text().splitlines()[0])
    assert row["word_alignments"] == [{"word": "hi", "start": 0.0, "end": 0.5}]


def test_word_alignments_absent_key_when_missing(make_run_context, tmp_path):
    ctx = make_run_context()
    out = tmp_path / "m.jsonl"
    op = PackJsonlOperator(PackJsonlConfig(output_path=str(out)), ctx=ctx)
    list(op.process(CutSet([_cut()])))
    row = json.loads(out.read_text().splitlines()[0])
    assert "word_alignments" not in row
