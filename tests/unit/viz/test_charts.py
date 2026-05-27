from voxkitchen.schema.cut import Cut
from voxkitchen.schema.cutset import CutSet
from voxkitchen.schema.provenance import Provenance
from voxkitchen.schema.supervision import Supervision
from voxkitchen.utils.time import now_utc
from voxkitchen.viz import charts


def _cut(cid, snr):
    return Cut(
        id=cid, recording_id="r", start=0.0, duration=1.5,
        supervisions=[Supervision(id=f"{cid}-s", recording_id="r", start=0.0,
                                  duration=1.5, text="hi", language="en")],
        metrics={"snr": snr},
        provenance=Provenance(source_cut_id=cid, generated_by="t", stage_name="t",
                              created_at=now_utc(), pipeline_run_id="run"),
    )


_CUTS = CutSet([_cut("a", 10.0), _cut("b", 20.0)])


def test_duration_histogram_returns_div_when_plotly_present():
    assert "<div" in charts.duration_histogram(_CUTS)


def test_metric_histogram_returns_div():
    assert "<div" in charts.metric_histogram(_CUTS, "snr")


def test_category_bar_returns_div():
    assert "<div" in charts.category_bar({"en": 5, "zh": 3}, "Language")


def test_empty_inputs_return_empty_string():
    assert charts.duration_histogram(CutSet([])) == ""
    assert charts.metric_histogram(_CUTS, "missing_metric") == ""
    assert charts.category_bar({}, "Language") == ""


def test_plotly_script_present():
    script = charts.plotly_script()
    assert script.startswith("<script>") and "Plotly" in script


def test_graceful_when_plotly_absent(monkeypatch):
    monkeypatch.setattr(charts, "_plotly", lambda: None)
    assert charts.duration_histogram(_CUTS) == ""
    assert charts.metric_histogram(_CUTS, "snr") == ""
    assert charts.category_bar({"en": 1}, "Language") == ""
    assert charts.plotly_script() == ""
