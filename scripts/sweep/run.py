"""Operator sweep driver — run every registered operator through its
canonical Docker image and report PASS/FAIL/SKIP per op.

See ``docs/superpowers/specs/2026-06-02-operator-sweep-design.md`` for the
design and behaviour contract.
"""

from __future__ import annotations

import argparse
import dataclasses
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
PIPELINES_DIR = REPO_ROOT / "scripts" / "sweep" / "pipelines"
FIXTURES_DIR = REPO_ROOT / "scripts" / "sweep" / "fixtures"
# vkit docker run always mounts CWD/work → /app/work inside the container.
# We put sweep output under /app/work/vkit-sweep/<op> (container path) so
# the assertion can read from WORK_BASE/<op> = CWD/work/vkit-sweep/<op>
# on the host.  The container-side path is /app/work/vkit-sweep/<op>.
WORK_BASE = REPO_ROOT / "work" / "vkit-sweep"
_CONTAINER_WORK_BASE = "/app/work/vkit-sweep"
REPORT_FILE = REPO_ROOT / "scripts" / "sweep" / "last-run.md"

IMAGE_ORDER = ["slim", "asr", "diarize", "tts", "fish-speech"]


@dataclasses.dataclass
class RunRecord:
    op: str
    image: str
    verdict: str  # "PASS" | "FAIL" | "SKIP"
    message: str
    wall_seconds: float
    exit_code: int | None = None


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run every operator through its canonical Docker image.",
    )
    parser.add_argument("--op", help="Filter: single operator name")
    parser.add_argument("--image", help="Filter: single image tag")
    parser.add_argument(
        "--no-pull",
        action="store_true",
        help="Skip missing images instead of pulling",
    )
    parser.add_argument(
        "--cleanup-each",
        action="store_true",
        help="Remove each op's work_dir after assertion",
    )
    parser.add_argument(
        "--setup",
        action="store_true",
        help="Generate derived fixtures (idempotent) and exit",
    )
    parser.add_argument(
        "--report-only",
        action="store_true",
        help="Re-render report.md from existing work_dirs without re-running",
    )
    args = parser.parse_args()

    if args.setup:
        from scripts.sweep.setup_fixtures import generate_fixtures

        generate_fixtures(repo_root=REPO_ROOT, fixtures_dir=FIXTURES_DIR)
        print(f"fixtures generated under {FIXTURES_DIR}")
        return 0

    yamls = _discover_yamls(op=args.op, image_filter=args.image)
    if not yamls:
        if args.op:
            print(
                f"error: no pipeline yaml for op {args.op!r} (expected at "
                f"{PIPELINES_DIR}/{args.op}.yaml)",
                file=sys.stderr,
            )
        else:
            print(f"error: no pipeline yamls found in {PIPELINES_DIR}", file=sys.stderr)
        return 2

    records: list[RunRecord] = []
    for idx, (op, image, yaml_path) in enumerate(yamls, 1):
        record = _run_one(
            op=op,
            image=image,
            yaml_path=yaml_path,
            no_pull=args.no_pull,
            cleanup_each=args.cleanup_each,
        )
        records.append(record)
        verdict_glyph = {"PASS": "✅", "FAIL": "❌", "SKIP": "⏭"}[record.verdict]
        print(
            f"[{idx}/{len(yamls)}] {record.op:<30} on {record.image:<12} "
            f"… {verdict_glyph} {record.verdict}  {record.wall_seconds:5.1f}s  "
            f"{record.message}"
        )

    from scripts.sweep.report import write_report

    write_report(records=records, path=REPORT_FILE)

    failures = [r for r in records if r.verdict == "FAIL"]
    return 1 if failures else 0


def _discover_yamls(*, op: str | None, image_filter: str | None) -> list[tuple[str, str, Path]]:
    """Return [(op_name, image_tag, yaml_path)] filtered + ordered by image."""
    from scripts.sweep.image_resolver import UnknownOperatorError, image_for_op

    if op:
        candidate = PIPELINES_DIR / f"{op}.yaml"
        if not candidate.is_file():
            return []
        try:
            return [(op, image_for_op(op), candidate)]
        except UnknownOperatorError:
            return [(op, "unknown", candidate)]

    out: list[tuple[str, str, Path]] = []
    for yaml_path in sorted(PIPELINES_DIR.glob("*.yaml")):
        op_name = yaml_path.stem
        try:
            image = image_for_op(op_name)
        except UnknownOperatorError:
            image = "unknown"
        if image_filter and image != image_filter:
            continue
        out.append((op_name, image, yaml_path))

    def sort_key(item: tuple[str, str, Path]) -> tuple[int, str]:
        op_name, image, _ = item
        try:
            return (IMAGE_ORDER.index(image), op_name)
        except ValueError:
            return (len(IMAGE_ORDER), op_name)

    return sorted(out, key=sort_key)


def _run_one(
    *,
    op: str,
    image: str,
    yaml_path: Path,
    no_pull: bool,
    cleanup_each: bool,
) -> RunRecord:
    """Run a single op pipeline and apply its assertion."""
    from scripts.sweep.assertions import ASSERTIONS, default_smoke_assertion

    # host path for assertion reads; container path for --work-dir override
    work_dir = WORK_BASE / op
    container_work_dir = f"{_CONTAINER_WORK_BASE}/{op}"
    if work_dir.exists():
        shutil.rmtree(work_dir)

    if image == "unknown":
        return RunRecord(op, image, "SKIP", "op-not-registered", 0.0)
    if op == "pyannote_diarize" and not os.environ.get("HF_TOKEN"):
        return RunRecord(op, image, "SKIP", "HF_TOKEN missing", 0.0)
    if not _image_present(image):
        if no_pull:
            return RunRecord(op, image, "SKIP", "image-not-local", 0.0)
        pull_ok, pull_msg = _pull_image(image)
        if not pull_ok:
            return RunRecord(op, image, "FAIL", f"pull-failed: {pull_msg}", 0.0)

    started = time.monotonic()

    # Build base command
    cmd = [
        "vkit",
        "docker",
        "run",
        "--tag",
        image,
        str(yaml_path),
        "--work-dir",
        container_work_dir,
        "--mount",
        f"{FIXTURES_DIR}:/app/scripts/sweep/fixtures",
    ]

    # Forward HF_TOKEN via a temp env-file so pyannote (and other gated
    # models) can authenticate inside the container. vkit docker run only
    # supports --env-file, not individual -e VAR=VAL flags.
    hf_token = os.environ.get("HF_TOKEN", "")
    _tmp_env: tempfile.NamedTemporaryFile | None = None
    if hf_token:
        _tmp_env = tempfile.NamedTemporaryFile(
            mode="w", suffix=".env", delete=False, prefix="vkit_sweep_"
        )
        _tmp_env.write(f"HF_TOKEN={hf_token}\n")
        _tmp_env.flush()
        _tmp_env.close()  # must close before subprocess reads the file (delete=False)
        cmd += ["--env-file", _tmp_env.name]

    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=str(REPO_ROOT),
        )
    finally:
        if _tmp_env is not None:
            try:
                os.unlink(_tmp_env.name)
            except OSError:
                pass
            _tmp_env = None
    wall = time.monotonic() - started

    run_log = proc.stdout + proc.stderr
    if proc.returncode != 0:
        last_stderr = next(
            (line.strip() for line in reversed(proc.stderr.splitlines()) if line.strip()),
            f"exit {proc.returncode}",
        )
        return RunRecord(op, image, "FAIL", last_stderr, wall, proc.returncode)

    assertion = ASSERTIONS.get(op, default_smoke_assertion)
    try:
        passed, message = assertion(work_dir, run_log)
    except Exception as exc:
        return RunRecord(
            op,
            image,
            "FAIL",
            f"assertion raised {type(exc).__name__}: {exc}",
            wall,
            0,
        )

    verdict = "PASS" if passed else "FAIL"
    if cleanup_each:
        shutil.rmtree(work_dir, ignore_errors=True)
    return RunRecord(op, image, verdict, message, wall, 0)


def _image_present(image: str) -> bool:
    """True iff the image is locally available."""
    full = f"ghcr.io/xqfeng-josie/voxkitchen:{image}"
    proc = subprocess.run(
        ["docker", "image", "inspect", full],
        capture_output=True,
        text=True,
    )
    return proc.returncode == 0


def _pull_image(image: str) -> tuple[bool, str]:
    """Pull an image via vkit. Returns (success, message)."""
    proc = subprocess.run(
        ["vkit", "docker", "pull", "--tag", image],
        capture_output=True,
        text=True,
    )
    if proc.returncode == 0:
        return True, ""
    return False, proc.stderr.strip().splitlines()[-1] if proc.stderr else "pull failed"


if __name__ == "__main__":
    raise SystemExit(main())
