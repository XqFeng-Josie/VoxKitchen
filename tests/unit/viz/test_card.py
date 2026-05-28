from voxkitchen.schema.cut import Cut
from voxkitchen.schema.cutset import CutSet
from voxkitchen.schema.provenance import Provenance
from voxkitchen.schema.supervision import Supervision
from voxkitchen.utils.time import now_utc


def _cuts():
    def _cut(cid, snr, gender):
        return Cut(
            id=cid, recording_id="r", start=0.0, duration=2.0,
            supervisions=[Supervision(id=f"{cid}-s", recording_id="r", start=0.0,
                                      duration=2.0, text="hello world", speaker="spk1",
                                      language="en", gender=gender)],
            metrics={"snr": snr},
            provenance=Provenance(source_cut_id=cid, generated_by="t", stage_name="t",
                                  created_at=now_utc(), pipeline_run_id="run"),
        )
    return CutSet([_cut("a", 10.0, "m"), _cut("b", 20.0, "f")])


def test_generate_card_writes_html_with_sections(tmp_path):
    from voxkitchen.viz.card.generator import generate_dataset_card

    out = tmp_path / "card.html"
    result = generate_dataset_card(_cuts(), out, title="My Set", description="desc here")
    assert result == out and out.is_file()
    html = out.read_text(encoding="utf-8")
    assert "My Set" in html and "desc here" in html
    assert "Languages" in html
    assert "snr" in html
    assert "hello world" in html


def test_generate_card_degrades_without_plotly(tmp_path, monkeypatch):
    from voxkitchen.viz import charts
    from voxkitchen.viz.card.generator import generate_dataset_card

    monkeypatch.setattr(charts, "_plotly", lambda: None)
    out = tmp_path / "card.html"
    generate_dataset_card(_cuts(), out, title="X")
    html = out.read_text(encoding="utf-8")
    assert "snr" in html  # metrics-summary table still renders without charts
    assert "No chart." in html  # chart sections show the graceful fallback
