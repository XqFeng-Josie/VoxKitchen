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
    for name in (
        docker_cmd.DOCKER_WORK_DIR_ENV,
        "DOCKER_CONFIG",
        "TMPDIR",
        "BUILDX_CONFIG",
        "XDG_CACHE_HOME",
    ):
        monkeypatch.delenv(name, raising=False)
    # Most tests expect no .env; create an opt-in helper.
    yield originals  # tests can flip nvidia-smi by mutating this dict


def _invoke(args: list[str]) -> tuple[int, list[str] | None]:
    """Invoke `vkit docker ...` and return (exit_code, captured_docker_argv)."""
    exit_code, cmd, _ = _invoke_details(args)
    return (exit_code, cmd)


def _invoke_details(args: list[str]) -> tuple[int, list[str] | None, dict[str, object]]:
    """Invoke `vkit docker ...` and include kwargs passed to `_run_and_exit`."""
    captured: list[list[str]] = []
    captured_kwargs: list[dict[str, object]] = []

    def _capture(cmd: list[str], **kwargs) -> None:
        captured.append(cmd)
        captured_kwargs.append(kwargs)
        raise SystemExit(0)

    with patch.object(docker_cmd, "_run_and_exit", side_effect=_capture):
        runner = CliRunner()
        result = runner.invoke(app, args)
    return (
        result.exit_code,
        captured[0] if captured else None,
        captured_kwargs[0] if captured_kwargs else {},
    )


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
    assert "NUMBA_CACHE_DIR=/app/work/.numba-cache" in cmd
    assert f"{tmp_path / 'work'}:/app/work" in cmd
    assert f"{tmp_path / 'output'}:/app/output" in cmd
    assert (tmp_path / "work" / ".numba-cache").is_dir()
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


def test_run_mounts_data_for_template_relative_paths(fake_docker, tmp_path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    _, cmd = _invoke(["docker", "run", "pipeline.yaml"])
    assert cmd is not None
    assert f"{data_dir}:/app/data" in cmd
    assert f"{data_dir}:/data" in cmd
    assert (tmp_path / "output").is_dir()


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


def test_run_forwards_vkit_run_flags(fake_docker) -> None:
    _, cmd = _invoke(
        [
            "docker",
            "run",
            "foo.yaml",
            "--dry-run",
            "--resume-from",
            "vad",
            "--stop-at",
            "asr",
            "--num-workers",
            "4",
            "--keep-intermediates",
        ]
    )
    assert cmd is not None
    assert cmd[-10:] == [
        "run",
        "foo.yaml",
        "--num-workers",
        "4",
        "--resume-from",
        "vad",
        "--stop-at",
        "asr",
        "--dry-run",
        "--keep-intermediates",
    ]


def test_run_forwards_no_preflight(fake_docker) -> None:
    _, cmd = _invoke(["docker", "run", "foo.yaml", "--dry-run", "--no-preflight"])
    assert cmd is not None
    # forwarded into the in-container `vkit run` args (after the pipeline path)
    assert "--no-preflight" in cmd
    assert cmd.index("--no-preflight") > cmd.index("foo.yaml")


def test_run_omits_no_preflight_by_default(fake_docker) -> None:
    _, cmd = _invoke(["docker", "run", "foo.yaml"])
    assert cmd is not None
    assert "--no-preflight" not in cmd


# ---------------------------------------------------------------------------
# image-preflight gate
# ---------------------------------------------------------------------------


def test_docker_run_no_preflight_bypasses_image_check(fake_docker, tmp_path, monkeypatch) -> None:
    """--no-preflight skips the host-side image-preflight gate entirely."""
    called = {"hit": False}

    def boom(*a: object, **k: object) -> None:
        called["hit"] = True
        raise AssertionError("image check should not run under --no-preflight")

    monkeypatch.setattr("voxkitchen.pipeline.image_preflight.check_image_preflight", boom)
    # Create a real yaml on host so the Path(pipeline).exists() guard is True,
    # confirming that --no-preflight is the thing that suppresses the check,
    # not the missing-file guard.
    yaml_path = tmp_path / "test.yaml"
    yaml_path.write_text("version: '0.1'\n", encoding="utf-8")
    exit_code, cmd = _invoke(
        ["docker", "run", "--tag", "slim", str(yaml_path), "--no-preflight", "--dry-run"]
    )
    assert not called["hit"], "check_image_preflight was called despite --no-preflight"
    # Command should still be assembled and passed to docker (exit 0 from our mock).
    assert exit_code == 0
    assert cmd is not None


def test_docker_run_gate_blocks_mismatched_op(fake_docker, tmp_path) -> None:
    """A host-present yaml with an op that doesn't fit --tag aborts before launch."""
    # Write a real yaml using faster_whisper_asr which is NOT in the slim image.
    yaml_path = tmp_path / "asr-pipeline.yaml"
    yaml_path.write_text(
        "\n".join(
            [
                "version: '0.1'",
                "name: test-asr",
                "work_dir: ./work/test-asr",
                "ingest:",
                "  source: dir",
                "  args:",
                "    root: ./data",
                "stages:",
                "  - name: asr",
                "    op: faster_whisper_asr",
                "    args:",
                "      model_size: tiny",
            ]
        ),
        encoding="utf-8",
    )
    # Track whether the docker subprocess was actually invoked.
    docker_launched = {"hit": False}

    def _sentinel_run_and_exit(cmd: list[str], **kwargs: object) -> None:
        docker_launched["hit"] = True
        raise SystemExit(0)

    runner = CliRunner()
    with patch.object(docker_cmd, "_run_and_exit", side_effect=_sentinel_run_and_exit):
        result = runner.invoke(app, ["docker", "run", "--tag", "slim", str(yaml_path)])

    assert not docker_launched["hit"], "docker should not have been launched after preflight abort"
    assert result.exit_code == 1


# ---------------------------------------------------------------------------
# download
# ---------------------------------------------------------------------------


def test_download_mounts_data_and_forwards_recipe_args(fake_docker, tmp_path) -> None:
    _, cmd = _invoke(
        [
            "docker",
            "download",
            "fleurs",
            "--root",
            "./data/fleurs",
            "--subsets",
            "en_us,zh_cn",
        ]
    )
    assert cmd is not None
    data_dir = tmp_path / "data"
    assert data_dir.is_dir()
    assert not (tmp_path / "work").exists()
    assert f"{data_dir}:/app/data" in cmd
    assert f"{data_dir}:/data" in cmd
    assert f"{docker_cmd.DEFAULT_IMAGE}:{docker_cmd.DEFAULT_DOWNLOAD_TAG}" in cmd
    assert cmd[-6:] == [
        "download",
        "fleurs",
        "--root",
        "./data/fleurs",
        "--subsets",
        "en_us,zh_cn",
    ]


# ---------------------------------------------------------------------------
# doctor / pull / shell / build
# ---------------------------------------------------------------------------


def test_doctor_passthrough_flags(fake_docker, tmp_path) -> None:
    (tmp_path / "data").mkdir()
    _, cmd = _invoke(["docker", "doctor", "--expect", "core", "--json"])
    assert cmd is not None
    # trailing command: vkit doctor --expect core --json
    tail = cmd[-4:]
    assert tail == ["doctor", "--expect", "core", "--json"]
    joined = " ".join(cmd)
    assert "/app/output" not in joined
    assert "/app/data" not in joined
    assert f"{tmp_path / 'data'}:/data" not in joined
    assert not (tmp_path / "work").exists()
    assert not (tmp_path / "output").exists()


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


def test_build_uses_project_docker_workspace(fake_docker, tmp_path) -> None:
    exit_code, cmd, kwargs = _invoke_details(["docker", "build", "slim"])
    assert exit_code == 0
    assert cmd is not None
    env = kwargs.get("env")
    assert isinstance(env, dict)

    docker_dir = tmp_path / ".docker"
    assert env["DOCKER_CONFIG"] == str(docker_dir / "config")
    assert env["TMPDIR"] == str(docker_dir / "tmp")
    assert env["BUILDX_CONFIG"] == str(docker_dir / "buildx")
    assert env["XDG_CACHE_HOME"] == str(docker_dir / "cache")
    assert (docker_dir / "config").is_dir()
    assert (docker_dir / "tmp").is_dir()
    assert (docker_dir / "buildx").is_dir()
    assert (docker_dir / "cache").is_dir()


def test_build_respects_custom_docker_workspace(fake_docker, tmp_path, monkeypatch) -> None:
    custom = tmp_path / "custom-docker-work"
    monkeypatch.setenv(docker_cmd.DOCKER_WORK_DIR_ENV, str(custom))
    _, _, kwargs = _invoke_details(["docker", "build", "slim"])
    env = kwargs.get("env")
    assert isinstance(env, dict)
    assert env["DOCKER_CONFIG"] == str(custom / "config")
    assert env["TMPDIR"] == str(custom / "tmp")
    assert (custom / "config").is_dir()
    assert (custom / "tmp").is_dir()


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


# ---------------------------------------------------------------------------
# _extra_mounts — HOST:CONTAINER syntax
# ---------------------------------------------------------------------------


def test_extra_mounts_plain_path_mirrors_host(tmp_path) -> None:
    """A plain path mounts at the same absolute path inside the container."""
    from voxkitchen.cli.docker_cmd import _extra_mounts

    p = tmp_path / "demo"
    p.mkdir()
    flags = _extra_mounts([p])
    abs_p = str(p.resolve())
    assert flags == ["-v", f"{abs_p}:{abs_p}:ro"]


def test_extra_mounts_host_colon_container_form(tmp_path) -> None:
    """A HOST:CONTAINER string mounts at the explicit container path."""
    from voxkitchen.cli.docker_cmd import _extra_mounts

    p = tmp_path / "demo"
    p.mkdir()
    flags = _extra_mounts([f"{p}:/app/sweep/fixtures"])
    abs_p = str(p.resolve())
    assert flags == ["-v", f"{abs_p}:/app/sweep/fixtures:ro"]


def test_extra_mounts_string_without_colon_is_plain_path(tmp_path) -> None:
    """A bare string path (no colon) still works as plain mount."""
    from voxkitchen.cli.docker_cmd import _extra_mounts

    p = tmp_path / "demo"
    p.mkdir()
    flags = _extra_mounts([str(p)])
    abs_p = str(p.resolve())
    assert flags == ["-v", f"{abs_p}:{abs_p}:ro"]
