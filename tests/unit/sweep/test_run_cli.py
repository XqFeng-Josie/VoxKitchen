"""Tests for the sweep driver's CLI behaviour (without real docker calls)."""

from pathlib import Path

import pytest


def test_run_cli_help_lists_main_flags() -> None:
    """`python scripts/sweep/run.py --help` shows --op, --image, --no-pull,
    --cleanup-each, --setup, --report-only, --report flags."""
    import subprocess

    result = subprocess.run(
        ["python", "scripts/sweep/run.py", "--help"],
        capture_output=True,
        text=True,
        cwd=str(Path(__file__).resolve().parents[3]),
    )
    assert result.returncode == 0, result.stderr
    for flag in [
        "--op",
        "--image",
        "--no-pull",
        "--cleanup-each",
        "--setup",
        "--report-only",
        "--report",
    ]:
        assert flag in result.stdout, f"missing flag {flag} in --help output"


def test_run_cli_missing_yaml_for_op_exits_nonzero() -> None:
    """Asking for an op without a pipeline yaml exits non-zero with a clear error."""
    import subprocess

    result = subprocess.run(
        ["python", "scripts/sweep/run.py", "--op", "no_such_op_no_yaml", "--no-pull"],
        capture_output=True,
        text=True,
        cwd=str(Path(__file__).resolve().parents[3]),
    )
    assert result.returncode != 0
    assert "pipeline" in result.stdout.lower() or "pipeline" in result.stderr.lower()


def test_run_one_passes_container_work_dir_and_cleans_host_work_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """_run_one must:
      1. Pass --work-dir <container-path> to `vkit docker run` (so the
         container writes its output where docker bind-mount makes it
         visible on the host),
      2. Call shutil.rmtree on the corresponding host path before the run.
    Regression test for the WORK_BASE bug fixed in commit 23afe82.
    """
    import subprocess

    import scripts.sweep.run as sweep_run

    captured: dict[str, list] = {"argv": [], "rmtree_paths": []}

    def fake_subprocess_run(args, *_a, **_kw):
        captured["argv"].append(args)

        class _R:
            returncode = 0
            stdout = ""
            stderr = ""

        return _R()

    def fake_rmtree(p, *_a, **_kw):
        captured["rmtree_paths"].append(Path(p))

    def fake_image_present(_image: str) -> bool:
        return True

    monkeypatch.setattr(subprocess, "run", fake_subprocess_run)
    monkeypatch.setattr(sweep_run.shutil, "rmtree", fake_rmtree)
    monkeypatch.setattr(sweep_run, "_image_present", fake_image_present)
    # Isolate WORK_BASE to a tmp dir AND pre-create the op's work_dir, so the
    # `if work_dir.exists(): rmtree(...)` branch in _run_one fires
    # deterministically. Without this the test passed only when a stale
    # ./work/vkit-sweep/identity happened to exist on disk (it does after a
    # local sweep run) and failed on a fresh checkout / in CI.
    monkeypatch.setattr(sweep_run, "WORK_BASE", tmp_path / "wb")
    (tmp_path / "wb" / "identity").mkdir(parents=True)

    yaml_path = tmp_path / "identity.yaml"
    yaml_path.write_text("dummy: yes")  # _run_one doesn't parse the yaml itself

    record = sweep_run._run_one(
        op="identity",
        image="slim",
        yaml_path=yaml_path,
        no_pull=True,
        cleanup_each=False,
    )

    # The recorded subprocess call must be the `vkit docker run` invocation.
    docker_run = next(argv for argv in captured["argv"] if argv[:2] == ["vkit", "docker"])
    assert "--work-dir" in docker_run
    work_dir_arg = docker_run[docker_run.index("--work-dir") + 1]
    assert work_dir_arg == "/app/work/vkit-sweep/identity", (
        f"expected container path /app/work/vkit-sweep/identity, got {work_dir_arg!r}"
    )
    # The pre-run cleanup must hit the host path (the monkeypatched WORK_BASE).
    expected_host = tmp_path / "wb" / "identity"
    assert expected_host in captured["rmtree_paths"], (
        f"expected rmtree({expected_host}), got {captured['rmtree_paths']}"
    )
    # The assertion will see no real manifest, so the record verdict should be FAIL.
    # (We're only validating the path mechanics here.)
    assert record.op == "identity"
    assert record.image == "slim"


def test_report_only_rebuilds_records_from_existing_workdirs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import scripts.sweep.assertions as assertions
    import scripts.sweep.run as sweep_run

    work_base = tmp_path / "work"
    op_dir = work_base / "identity"
    stage_dir = op_dir / "00_identity"
    stage_dir.mkdir(parents=True)
    (stage_dir / "_stats.json").write_text('{"wall_time_seconds": 1.25}', encoding="utf-8")

    monkeypatch.setattr(sweep_run, "WORK_BASE", work_base)
    monkeypatch.setitem(
        assertions.ASSERTIONS,
        "identity",
        lambda _work_dir, _log: (True, "replayed ok"),
    )

    records = sweep_run._records_from_existing_workdirs(
        [("identity", "slim", tmp_path / "identity.yaml")]
    )

    assert len(records) == 1
    assert records[0].op == "identity"
    assert records[0].verdict == "PASS"
    assert records[0].message == "replayed ok"
    assert records[0].wall_seconds == 1.25


def test_report_only_marks_missing_workdir_as_skip(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import scripts.sweep.run as sweep_run

    monkeypatch.setattr(sweep_run, "WORK_BASE", tmp_path / "missing-work")

    records = sweep_run._records_from_existing_workdirs(
        [("identity", "slim", tmp_path / "identity.yaml")]
    )

    assert len(records) == 1
    assert records[0].verdict == "SKIP"
    assert records[0].message == "no-existing-work-dir"


def test_report_path_resolves_relative_to_repo_root(tmp_path: Path) -> None:
    import scripts.sweep.run as sweep_run

    assert (
        sweep_run._resolve_report_path("scripts/sweep/reports/slim.md")
        == sweep_run.REPO_ROOT / "scripts/sweep/reports/slim.md"
    )
    assert sweep_run._resolve_report_path(tmp_path / "custom.md") == tmp_path / "custom.md"


def test_write_report_uses_custom_path(tmp_path: Path) -> None:
    import scripts.sweep.run as sweep_run

    report_path = tmp_path / "reports" / "slim.md"
    sweep_run._write_report(
        [
            sweep_run.RunRecord(
                op="identity",
                image="slim",
                verdict="PASS",
                message="ok",
                wall_seconds=0.1,
                exit_code=0,
            )
        ],
        report_path,
    )

    content = report_path.read_text(encoding="utf-8")
    assert "`identity`" in content
    assert "1/1 PASS" in content
