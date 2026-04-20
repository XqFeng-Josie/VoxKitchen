"""Tests for ``vkit docker`` subcommands.

We don't actually invoke `docker` — every subcommand calls
:func:`voxkitchen.cli.docker_cmd._run_and_exit` at the end, which wraps
``subprocess.run``. Patching that lets us inspect the exact argv that
*would* have been passed to Docker, which is what matters: we're
validating flag assembly, not Docker itself.
"""

from __future__ import annotations

import os
from unittest.mock import patch

import pytest
from typer.testing import CliRunner
from voxkitchen.cli import docker_cmd
from voxkitchen.cli.main import app


@pytest.fixture
def fake_docker(monkeypatch, tmp_path):
    """Pretend docker and nvidia-smi both exist on PATH, and cd to tmp_path."""
    # shutil.which is called twice: once for docker, once for nvidia-smi.
    # Return a non-None sentinel for docker; let nvidia-smi be controllable.
    originals: dict[str, str | None] = {"docker": "/usr/bin/docker", "nvidia-smi": None}

    def fake_which(name: str) -> str | None:
        return originals.get(name, None)

    monkeypatch.setattr(docker_cmd.shutil, "which", fake_which)
    monkeypatch.chdir(tmp_path)
    # Most tests expect no .env; create an opt-in helper.
    yield originals  # tests can flip nvidia-smi by mutating this dict


def _captured_cmd(fake_docker) -> list[str]:
    """Helper: the docker argv that would have been executed."""
    with patch.object(docker_cmd, "_run_and_exit") as spy:
        # _run_and_exit normally raises typer.Exit; silence it for capture
        def _capture(cmd: list[str]) -> None:
            pass

        spy.side_effect = _capture
        yield spy


def _invoke(args: list[str]) -> tuple[int, list[str] | None]:
    """Invoke `vkit docker ...` and return (exit_code, captured_docker_argv)."""
    captured: list[list[str]] = []

    def _capture(cmd: list[str]) -> None:
        captured.append(cmd)
        raise SystemExit(0)

    with patch.object(docker_cmd, "_run_and_exit", side_effect=_capture):
        runner = CliRunner()
        result = runner.invoke(app, args)
    return (result.exit_code, captured[0] if captured else None)


# ---------------------------------------------------------------------------
# run
# ---------------------------------------------------------------------------


def test_run_basic_defaults(fake_docker, tmp_path) -> None:
    exit_code, cmd = _invoke(["docker", "run", "examples/foo.yaml"])
    assert exit_code == 0
    assert cmd is not None
    assert cmd[0:2] == ["docker", "run"]
    # always-on flags
    assert "--rm" in cmd
    assert "--user" in cmd
    uid_gid = cmd[cmd.index("--user") + 1]
    assert uid_gid == f"{os.getuid()}:{os.getgid()}"
    assert "-e" in cmd and "HOME=/tmp" in cmd
    # image defaulted
    assert f"{docker_cmd.DEFAULT_IMAGE}:{docker_cmd.DEFAULT_TAG}" in cmd
    # trailing vkit-command shape: ["run", "<pipeline>"]
    assert cmd[-2] == "run"
    # pipeline passed through (host file didn't exist → no mount, passthrough)
    assert cmd[-1] == "examples/foo.yaml"


def test_run_with_custom_tag(fake_docker) -> None:
    _, cmd = _invoke(["docker", "run", "--tag", "asr", "foo.yaml"])
    assert cmd is not None
    assert f"{docker_cmd.DEFAULT_IMAGE}:asr" in cmd


def test_run_with_full_image_override(fake_docker) -> None:
    _, cmd = _invoke(["docker", "run", "--image", "my.registry/voxkitchen:custom", "foo.yaml"])
    assert cmd is not None
    assert "my.registry/voxkitchen:custom" in cmd
    # and default isn't also present
    assert f"{docker_cmd.DEFAULT_IMAGE}:latest" not in cmd


def test_run_gpus_none_omits_gpu_flag(fake_docker) -> None:
    _, cmd = _invoke(["docker", "run", "--gpus", "none", "foo.yaml"])
    assert cmd is not None
    assert "--gpus" not in cmd


def test_run_gpus_all_always_attaches(fake_docker) -> None:
    # nvidia-smi fake not present, but --gpus all forces it
    _, cmd = _invoke(["docker", "run", "--gpus", "all", "foo.yaml"])
    assert cmd is not None
    assert "--gpus" in cmd
    assert cmd[cmd.index("--gpus") + 1] == "all"


def test_run_gpus_auto_with_nvidia_smi(fake_docker) -> None:
    fake_docker["nvidia-smi"] = "/usr/bin/nvidia-smi"
    _, cmd = _invoke(["docker", "run", "--gpus", "auto", "foo.yaml"])
    assert cmd is not None
    assert "--gpus" in cmd and cmd[cmd.index("--gpus") + 1] == "all"


def test_run_gpus_auto_without_nvidia_smi(fake_docker) -> None:
    # default state of fake_docker has nvidia-smi None
    _, cmd = _invoke(["docker", "run", "--gpus", "auto", "foo.yaml"])
    assert cmd is not None
    assert "--gpus" not in cmd


def test_run_invalid_gpus_value_errors(fake_docker) -> None:
    exit_code, _ = _invoke(["docker", "run", "--gpus", "bogus", "foo.yaml"])
    assert exit_code != 0


def test_run_auto_mounts_existing_pipeline(fake_docker, tmp_path) -> None:
    yaml = tmp_path / "my.yaml"
    yaml.write_text("version: '0.1'\n", encoding="utf-8")
    _, cmd = _invoke(["docker", "run", str(yaml)])
    assert cmd is not None
    # A -v <abs>:<abs>:ro should be present
    expected = f"{yaml.resolve()}:{yaml.resolve()}:ro"
    assert expected in cmd
    # Pipeline arg passed to vkit run is the absolute container path
    assert cmd[-1] == str(yaml.resolve())


def test_run_auto_envfile_when_dotenv_present(fake_docker, tmp_path) -> None:
    (tmp_path / ".env").write_text("HF_TOKEN=hf_xxx\n", encoding="utf-8")
    _, cmd = _invoke(["docker", "run", "foo.yaml"])
    assert cmd is not None
    assert "--env-file" in cmd
    assert cmd[cmd.index("--env-file") + 1] == ".env"


def test_run_no_envfile_without_dotenv(fake_docker) -> None:
    _, cmd = _invoke(["docker", "run", "foo.yaml"])
    assert cmd is not None
    assert "--env-file" not in cmd


def test_run_extra_mount(fake_docker, tmp_path) -> None:
    extra = tmp_path / "extra"
    extra.mkdir()
    _, cmd = _invoke(["docker", "run", "foo.yaml", "-m", str(extra)])
    assert cmd is not None
    expected = f"{extra.resolve()}:{extra.resolve()}:ro"
    assert expected in cmd


# ---------------------------------------------------------------------------
# doctor / pull / shell / build
# ---------------------------------------------------------------------------


def test_doctor_passthrough_flags(fake_docker) -> None:
    _, cmd = _invoke(["docker", "doctor", "--expect", "core", "--json"])
    assert cmd is not None
    # trailing command: vkit doctor --expect core --json
    tail = cmd[-4:]
    assert tail == ["doctor", "--expect", "core", "--json"]


def test_pull(fake_docker) -> None:
    _, cmd = _invoke(["docker", "pull", "--tag", "slim"])
    assert cmd is not None
    assert cmd[:2] == ["docker", "pull"]
    assert cmd[-1] == f"{docker_cmd.DEFAULT_IMAGE}:slim"


def test_build_default_target(fake_docker, tmp_path) -> None:
    _, cmd = _invoke(["docker", "build"])
    assert cmd is not None
    # docker build --target latest -f docker/Dockerfile -t voxkitchen:latest .
    assert cmd[:2] == ["docker", "build"]
    assert "--target" in cmd
    assert cmd[cmd.index("--target") + 1] == "latest"
    assert "-t" in cmd
    assert cmd[cmd.index("-t") + 1] == "voxkitchen:latest"
    assert cmd[-1] == "."


def test_build_reads_hf_token_from_dotenv(fake_docker, tmp_path) -> None:
    (tmp_path / ".env").write_text("HF_TOKEN=hf_abc123\nOTHER=x\n", encoding="utf-8")
    _, cmd = _invoke(["docker", "build", "asr"])
    assert cmd is not None
    # Should see --build-arg HF_TOKEN=hf_abc123
    assert "--build-arg" in cmd
    idx = cmd.index("--build-arg")
    assert cmd[idx + 1] == "HF_TOKEN=hf_abc123"
    # Target applied
    assert cmd[cmd.index("--target") + 1] == "asr"


def test_build_no_hf_token_flag(fake_docker, tmp_path) -> None:
    (tmp_path / ".env").write_text("HF_TOKEN=hf_abc123\n", encoding="utf-8")
    _, cmd = _invoke(["docker", "build", "--no-hf-token"])
    assert cmd is not None
    assert "--build-arg" not in cmd


def test_shell_is_interactive_bash(fake_docker) -> None:
    _, cmd = _invoke(["docker", "shell", "--tag", "asr"])
    assert cmd is not None
    # docker run --rm -it --user ... --entrypoint /bin/bash <image>
    assert "-it" in cmd
    assert "--entrypoint" in cmd
    assert cmd[cmd.index("--entrypoint") + 1] == "/bin/bash"
    assert cmd[-1] == f"{docker_cmd.DEFAULT_IMAGE}:asr"


# ---------------------------------------------------------------------------
# Docker not installed
# ---------------------------------------------------------------------------


def test_run_aborts_if_docker_missing(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)

    def _no_docker(name: str) -> str | None:
        return None

    monkeypatch.setattr(docker_cmd.shutil, "which", _no_docker)
    runner = CliRunner()
    result = runner.invoke(app, ["docker", "run", "foo.yaml"])
    assert result.exit_code != 0


# ---------------------------------------------------------------------------
# HF_TOKEN parser helpers (unit)
# ---------------------------------------------------------------------------


def test_read_hf_token_simple(tmp_path) -> None:
    (tmp_path / ".env").write_text("HF_TOKEN=hf_simple\n", encoding="utf-8")
    assert docker_cmd._read_hf_token_from_env(tmp_path / ".env") == "hf_simple"


def test_read_hf_token_quoted(tmp_path) -> None:
    (tmp_path / ".env").write_text('HF_TOKEN="hf_q"\n', encoding="utf-8")
    assert docker_cmd._read_hf_token_from_env(tmp_path / ".env") == "hf_q"


def test_read_hf_token_absent(tmp_path) -> None:
    (tmp_path / ".env").write_text("OTHER=x\n", encoding="utf-8")
    assert docker_cmd._read_hf_token_from_env(tmp_path / ".env") is None


def test_read_hf_token_no_file(tmp_path) -> None:
    assert docker_cmd._read_hf_token_from_env(tmp_path / "missing.env") is None


def test_read_hf_token_skips_comments(tmp_path) -> None:
    (tmp_path / ".env").write_text("# HF_TOKEN=wrong\nHF_TOKEN=hf_right\n", encoding="utf-8")
    assert docker_cmd._read_hf_token_from_env(tmp_path / ".env") == "hf_right"


# ---------------------------------------------------------------------------
# Resolve image helper
# ---------------------------------------------------------------------------


def test_resolve_image_defaults() -> None:
    assert docker_cmd._resolve_image("latest", None) == f"{docker_cmd.DEFAULT_IMAGE}:latest"


def test_resolve_image_tag_applied() -> None:
    assert docker_cmd._resolve_image("asr", None) == f"{docker_cmd.DEFAULT_IMAGE}:asr"


def test_resolve_image_override_wins() -> None:
    assert docker_cmd._resolve_image("latest", "custom:tag") == "custom:tag"
