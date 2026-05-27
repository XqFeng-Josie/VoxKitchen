import sys
from pathlib import Path

_EXAMPLE = Path(__file__).resolve().parents[3] / "examples" / "plugin-operator"


def _word_count_cls():
    if str(_EXAMPLE) not in sys.path:
        sys.path.insert(0, str(_EXAMPLE))
    from voxkitchen_example_plugin.operator import WordCountOperator

    return WordCountOperator


def test_example_operator_is_a_valid_operator():
    from voxkitchen.operators.base import Operator, OperatorConfig

    cls = _word_count_cls()
    assert issubclass(cls, Operator)
    assert cls.name == "word_count"
    assert issubclass(cls.config_cls, OperatorConfig)
    assert cls.reads == ["supervisions.text"]
    assert cls.writes == ["metrics.word_count"]


def test_importing_example_does_not_register_it():
    _word_count_cls()
    from voxkitchen.operators.registry import _REGISTRY

    assert "word_count" not in _REGISTRY


def test_example_operator_counts_words(make_run_context):
    from voxkitchen.schema.cut import Cut
    from voxkitchen.schema.cutset import CutSet
    from voxkitchen.schema.provenance import Provenance
    from voxkitchen.schema.supervision import Supervision
    from voxkitchen.utils.time import now_utc

    cls = _word_count_cls()
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
                text="hello there world",
            )
        ],
        provenance=Provenance(
            source_cut_id="c1",
            generated_by="test",
            stage_name="t",
            created_at=now_utc(),
            pipeline_run_id="run",
        ),
    )
    op = cls(cls.config_cls(), ctx=make_run_context())
    out = list(op.process(CutSet([cut])))
    assert out[0].metrics["word_count"] == 3.0
