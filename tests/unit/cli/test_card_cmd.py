"""Tests for the `vkit card` CLI command."""

from __future__ import annotations

from pathlib import Path

from voxkitchen.schema.cut import Cut
from voxkitchen.schema.cutset import CutSet
from voxkitchen.schema.io import SCHEMA_VERSION, HeaderRecord
from voxkitchen.schema.provenance import Provenance
from voxkitchen.schema.supervision import Supervision
from voxkitchen.utils.time import now_utc


def _write_manifest(path: Path) -> Path:
    cut = Cut(
        id="a",
        recording_id="r",
        start=0.0,
        duration=1.0,
        supervisions=[
            Supervision(
                id="a-s",
                recording_id="r",
                start=0.0,
                duration=1.0,
                text="hi",
                language="en",
            )
        ],
        metrics={"snr": 12.0},
        provenance=Provenance(
            source_cut_id="a",
            generated_by="t",
            stage_name="t",
            created_at=now_utc(),
            pipeline_run_id="run",
        ),
    )
    header = HeaderRecord(
        schema_version=SCHEMA_VERSION,
        created_at=now_utc(),
        pipeline_run_id="run",
        stage_name="card",
    )
    CutSet([cut]).to_jsonl_gz(path, header)
    return path


def test_card_command_writes_html(tmp_path: Path) -> None:
    from voxkitchen.cli.card_cmd import card_command

    manifest = _write_manifest(tmp_path / "cuts.jsonl.gz")
    out = tmp_path / "card.html"
    card_command(manifest, out=out, title="T", description="")
    assert out.is_file()
    assert "T" in out.read_text(encoding="utf-8")


def test_card_command_friendly_error_when_viz_extra_missing(tmp_path, monkeypatch, capsys):
    import sys

    import pytest
    import typer
    from voxkitchen.cli.card_cmd import card_command

    manifest = _write_manifest(tmp_path / "cuts.jsonl.gz")
    # Simulate the 'viz' extra (jinja2) not being installed.
    monkeypatch.setitem(sys.modules, "jinja2", None)
    with pytest.raises(typer.Exit) as exc:
        card_command(manifest, out=tmp_path / "card.html")
    assert exc.value.exit_code == 1
    out = capsys.readouterr().out
    assert "viz" in out and "[viz]" in out  # the extra name survives Rich markup
