from pathlib import Path

import pytest
import typer
from voxkitchen.cli.run import run_command
from voxkitchen.cli.validate import validate_command


def _write(p: Path, text: str) -> Path:
    p.write_text(text, encoding="utf-8")
    return p


BROKEN = """
version: "0.1"
name: broken
work_dir: ./work
ingest:
  source: dir
  args: {root: .}
stages:
  - {name: cer, op: cer_wer}
  - {name: pack, op: pack_jsonl}
"""


def test_validate_reports_preflight_error(tmp_path, capsys):
    yaml_path = _write(tmp_path / "p.yaml", BROKEN)
    with pytest.raises(typer.Exit) as exc:
        validate_command(yaml_path, preflight=True)
    assert exc.value.exit_code == 1
    out = capsys.readouterr().out
    assert "supervisions.text" in out


def test_no_preflight_skips_the_check(tmp_path):
    yaml_path = _write(tmp_path / "p.yaml", BROKEN)
    # args are individually valid, so with preflight off this should pass (no raise)
    validate_command(yaml_path, preflight=False)


def test_real_run_preflights_before_executing(tmp_path, capsys):
    # A real `vkit run` (not --dry-run) must fail fast on a broken chain BEFORE
    # spinning up the executor — not mid-run.
    yaml_path = _write(tmp_path / "p.yaml", BROKEN)
    with pytest.raises(typer.Exit) as exc:
        run_command(yaml_path)
    assert exc.value.exit_code == 1
    assert "supervisions.text" in capsys.readouterr().out


def test_real_run_no_preflight_reaches_executor(tmp_path, monkeypatch):
    # With --no-preflight, a real run skips the check and proceeds to run_pipeline.
    yaml_path = _write(tmp_path / "p.yaml", BROKEN)
    called = {}

    def _fake_run_pipeline(spec, **kwargs):
        called["ran"] = True

    monkeypatch.setattr("voxkitchen.cli.run.run_pipeline", _fake_run_pipeline)
    run_command(yaml_path, no_preflight=True)
    assert called.get("ran") is True
