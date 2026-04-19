"""``vkit docker`` — run VoxKitchen inside a published Docker image.

Mirrors the local CLI (``vkit run`` / ``vkit doctor`` / ...) but executes
inside a container, so users don't have to remember the exact ``docker
run`` flags the image expects (non-root ``--user``, `./work` and `./data`
bind mounts, ``--env-file .env`` for ``HF_TOKEN``, GPU autodetection).

Commands:
  vkit docker run <pipeline>    Execute a pipeline inside the image.
  vkit docker doctor            Run per-env health report inside the image.
  vkit docker build [target]    Build a local image (wraps `docker build`).
  vkit docker pull              Pull an image tag from the registry.
  vkit docker shell             Drop into an interactive bash in the image.

All subcommands accept ``--tag`` (default ``latest``) to pick an image tag,
or ``--image`` to override the full image name.

Design notes:

- We shell out to the ``docker`` CLI via :mod:`subprocess`. We don't
  pull in the Python docker SDK — subprocess is zero extra deps and
  docker CLI errors are already user-friendly.
- The pipeline argument to ``vkit docker run`` is auto-mounted when it
  refers to a host file: we bind the file at its absolute path inside
  the container so ``vkit run`` sees the same path. Paths that don't
  exist on host are passed through unchanged — they're presumed to be
  baked-in resources like ``examples/pipelines/demo-no-asr.yaml``.
- ``--user $(id -u):$(id -g)`` + ``-e HOME=/tmp`` are default so output
  files under ``./work`` are owned by the host user, not root.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Literal, NoReturn

import typer
from rich import print as rprint

DEFAULT_IMAGE = "ghcr.io/xqfeng-josie/voxkitchen"
DEFAULT_TAG = "latest"


docker_app = typer.Typer(
    name="docker",
    help="Run VoxKitchen inside a Docker image (see vkit docker --help).",
    no_args_is_help=True,
)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _require_docker() -> None:
    """Abort with a clear message if ``docker`` is not on PATH."""
    if shutil.which("docker") is None:
        rprint(
            "[red]error:[/red] `docker` not found on PATH. Install Docker "
            "(https://docs.docker.com/get-docker/) or, for local execution, "
            "use `vkit run` instead."
        )
        raise typer.Exit(code=2)


def _resolve_image(tag: str, image: str | None) -> str:
    """Combine --tag and --image into one final image reference.

    ``--image`` always wins; otherwise ``{DEFAULT_IMAGE}:{tag}``.
    """
    if image:
        return image
    return f"{DEFAULT_IMAGE}:{tag}"


def _gpu_flags(mode: Literal["auto", "all", "none"]) -> list[str]:
    """Return ``--gpus`` flags. ``auto`` attaches all GPUs iff nvidia-smi
    is on PATH; ``all`` always attaches; ``none`` never.
    """
    if mode == "none":
        return []
    if mode == "all":
        return ["--gpus", "all"]
    # auto
    if shutil.which("nvidia-smi"):
        return ["--gpus", "all"]
    return []


def _env_file_flag(env_file: Path | None) -> list[str]:
    """If the given env file (or ./.env by default) exists, use it."""
    path = env_file if env_file is not None else Path(".env")
    if path.is_file():
        return ["--env-file", str(path)]
    return []


def _common_run_flags() -> list[str]:
    """Flags every ``docker run`` call gets: rm, user, home, default mounts.

    Creates ``./work`` on the host before mounting so Docker doesn't
    pre-create it as root.
    """
    cwd = Path.cwd()
    (cwd / "work").mkdir(parents=True, exist_ok=True)

    flags = [
        "--rm",
        "--user",
        f"{os.getuid()}:{os.getgid()}",
        "-e",
        "HOME=/tmp",
        "-v",
        f"{cwd / 'work'}:/app/work",
    ]
    data_dir = cwd / "data"
    if data_dir.is_dir():
        flags += ["-v", f"{data_dir}:/data"]
    return flags


def _pipeline_mount(pipeline_arg: str) -> tuple[list[str], str]:
    """If ``pipeline_arg`` refers to a host file, bind-mount it at its
    absolute path inside the container. Return (extra_mount_flags,
    pipeline_path_to_pass_to_container).

    If the host file doesn't exist, pass the argument through unchanged —
    the image may have it baked in (``examples/pipelines/demo-no-asr.yaml``
    etc.), in which case the container will resolve it against its
    ``WORKDIR`` = ``/app``.
    """
    p = Path(pipeline_arg)
    if not p.is_file():
        return ([], pipeline_arg)
    abs_p = p.resolve()
    # Mount at same absolute path so `vkit run <abs>` inside the container
    # sees the same file the user referenced on the host.
    return (["-v", f"{abs_p}:{abs_p}:ro"], str(abs_p))


def _extra_mounts(paths: list[Path]) -> list[str]:
    """Expand ``--mount`` flags into ``-v HOST:HOST:ro`` pairs."""
    out: list[str] = []
    for p in paths:
        abs_p = p.resolve()
        out += ["-v", f"{abs_p}:{abs_p}:ro"]
    return out


def _run_and_exit(cmd: list[str]) -> NoReturn:
    """Run ``cmd`` and propagate its exit code. stdout/stderr inherited."""
    rc = subprocess.run(cmd, check=False).returncode  # noqa: S603
    raise typer.Exit(code=rc)


# ---------------------------------------------------------------------------
# Subcommands
# ---------------------------------------------------------------------------


@docker_app.command("run")
def run_cmd(
    pipeline: str = typer.Argument(..., help="Pipeline YAML path (host or image-relative)."),
    tag: str = typer.Option(DEFAULT_TAG, "--tag", help="Image tag."),
    image: str | None = typer.Option(None, "--image", help="Full image name override."),
    gpus: str = typer.Option("auto", "--gpus", help="GPU mode: auto | all | none."),
    env_file: Path | None = typer.Option(
        None, "--env-file", help="Docker --env-file path. Default: ./.env if it exists."
    ),
    mount: list[Path] = typer.Option(
        [], "--mount", "-m", help="Extra bind mount (repeatable). Host path, mounted read-only."
    ),
) -> None:
    """Run a pipeline inside the container."""
    _require_docker()
    if gpus not in {"auto", "all", "none"}:
        rprint(f"[red]error:[/red] --gpus must be auto|all|none, got {gpus!r}")
        raise typer.Exit(code=2)

    mount_flags, container_path = _pipeline_mount(pipeline)
    cmd = [
        "docker",
        "run",
        *_common_run_flags(),
        *_gpu_flags(gpus),  # type: ignore[arg-type]
        *_env_file_flag(env_file),
        *mount_flags,
        *_extra_mounts(mount),
        _resolve_image(tag, image),
        "run",
        container_path,
    ]
    _run_and_exit(cmd)


@docker_app.command("doctor")
def doctor_cmd(
    tag: str = typer.Option(DEFAULT_TAG, "--tag", help="Image tag."),
    image: str | None = typer.Option(None, "--image", help="Full image name override."),
    expect: str | None = typer.Option(
        None, "--expect", help="Env name to validate (core|asr|diarize|tts|fish-speech)."
    ),
    json_out: bool = typer.Option(False, "--json", help="Emit JSON on stdout."),
    gpus: str = typer.Option("none", "--gpus", help="GPU mode (doctor does not need GPU)."),
) -> None:
    """Run ``vkit doctor`` inside the container."""
    _require_docker()
    doctor_args = ["doctor"]
    if expect:
        doctor_args += ["--expect", expect]
    if json_out:
        doctor_args.append("--json")

    cmd = [
        "docker",
        "run",
        *_common_run_flags(),
        *_gpu_flags(gpus),  # type: ignore[arg-type]
        _resolve_image(tag, image),
        *doctor_args,
    ]
    _run_and_exit(cmd)


@docker_app.command("build")
def build_cmd(
    target: str = typer.Argument(DEFAULT_TAG, help="Dockerfile target: slim|asr|diarize|tts|fish-speech|latest."),
    tag: str | None = typer.Option(
        None, "--tag", help="Image tag to apply. Default: voxkitchen:<target>."
    ),
    hf_token: bool = typer.Option(
        True,
        "--hf-token/--no-hf-token",
        help="Pass HF_TOKEN from ./.env as a build arg (so pyannote is baked in).",
    ),
    extra: list[str] = typer.Argument(
        None, help="Additional flags passed to `docker build` (use `--` first)."
    ),
) -> None:
    """Build a Docker image locally (wraps `docker build`)."""
    _require_docker()
    final_tag = tag or f"voxkitchen:{target}"

    build_args: list[str] = []
    if hf_token:
        token = _read_hf_token_from_env()
        if token:
            build_args += ["--build-arg", f"HF_TOKEN={token}"]
            rprint(
                "[dim][vkit docker build] using HF_TOKEN from .env "
                "(pyannote model baked into image)[/dim]"
            )

    cmd = [
        "docker",
        "build",
        "--target",
        target,
        "-f",
        "docker/Dockerfile",
        "-t",
        final_tag,
        *build_args,
        *(extra or []),
        ".",
    ]
    _run_and_exit(cmd)


def _read_hf_token_from_env(env_path: Path | None = None) -> str | None:
    """Best-effort parse of ``HF_TOKEN=...`` from ``./.env``."""
    path = env_path if env_path is not None else Path(".env")
    if not path.is_file():
        return None
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("HF_TOKEN="):
            value = line.split("=", 1)[1].strip()
            # strip wrapping quotes
            if (value.startswith('"') and value.endswith('"')) or (
                value.startswith("'") and value.endswith("'")
            ):
                value = value[1:-1]
            return value or None
    return None


@docker_app.command("pull")
def pull_cmd(
    tag: str = typer.Option(DEFAULT_TAG, "--tag", help="Image tag."),
    image: str | None = typer.Option(None, "--image", help="Full image name override."),
) -> None:
    """Pull an image tag from the registry."""
    _require_docker()
    cmd = ["docker", "pull", _resolve_image(tag, image)]
    _run_and_exit(cmd)


@docker_app.command("shell")
def shell_cmd(
    tag: str = typer.Option(DEFAULT_TAG, "--tag", help="Image tag."),
    image: str | None = typer.Option(None, "--image", help="Full image name override."),
    gpus: str = typer.Option("none", "--gpus", help="GPU mode."),
) -> None:
    """Drop into an interactive bash inside the image (for debugging)."""
    _require_docker()
    if not sys.stdin.isatty():
        rprint(
            "[yellow]warning:[/yellow] shell requested but stdin is not a TTY. "
            "The shell may exit immediately."
        )
    cmd = [
        "docker",
        "run",
        "--rm",
        "-it",
        "--user",
        f"{os.getuid()}:{os.getgid()}",
        "-e",
        "HOME=/tmp",
        *_gpu_flags(gpus),  # type: ignore[arg-type]
        "--entrypoint",
        "/bin/bash",
        _resolve_image(tag, image),
    ]
    _run_and_exit(cmd)
