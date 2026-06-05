"""Tests for the operator-sweep report renderer."""

from pathlib import Path


def test_report_summary_counts_pass_fail_skip(tmp_path: Path) -> None:
    from scripts.sweep.report import write_report
    from scripts.sweep.run import RunRecord

    records = [
        RunRecord("a", "slim", "PASS", "1 cuts", 0.5),
        RunRecord("b", "slim", "PASS", "1 cuts", 0.5),
        RunRecord("c", "asr", "FAIL", "unknown operator", 1.0),
        RunRecord("d", "diarize", "SKIP", "HF_TOKEN missing", 0.0),
    ]
    path = tmp_path / "report.md"
    write_report(records=records, path=path)

    text = path.read_text()
    assert "2/4 PASS" in text or "Verdict" in text
    assert "1 FAIL" in text
    assert "1 SKIP" in text


def test_report_lists_failures_with_messages(tmp_path: Path) -> None:
    from scripts.sweep.report import write_report
    from scripts.sweep.run import RunRecord

    records = [
        RunRecord("ok_op", "slim", "PASS", "fine", 0.5),
        RunRecord("bad_op", "asr", "FAIL", "specific failure message", 1.2),
    ]
    path = tmp_path / "report.md"
    write_report(records=records, path=path)
    text = path.read_text()

    assert "## Failures" in text
    assert "bad_op" in text
    assert "specific failure message" in text


def test_report_lists_skips_separately(tmp_path: Path) -> None:
    from scripts.sweep.report import write_report
    from scripts.sweep.run import RunRecord

    records = [
        RunRecord("ok_op", "slim", "PASS", "fine", 0.5),
        RunRecord("skipped", "diarize", "SKIP", "HF_TOKEN missing", 0.0),
    ]
    path = tmp_path / "report.md"
    write_report(records=records, path=path)
    text = path.read_text()

    assert "## Skips" in text
    assert "skipped" in text
    assert "HF_TOKEN missing" in text


def test_report_full_results_table_has_all_records(tmp_path: Path) -> None:
    from scripts.sweep.report import write_report
    from scripts.sweep.run import RunRecord

    records = [RunRecord(f"op_{i}", "slim", "PASS", f"msg {i}", 0.1 * i) for i in range(5)]
    path = tmp_path / "report.md"
    write_report(records=records, path=path)
    text = path.read_text()

    for i in range(5):
        assert f"op_{i}" in text
