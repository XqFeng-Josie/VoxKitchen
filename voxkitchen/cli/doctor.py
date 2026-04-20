"""``vkit doctor`` — report operator availability per env.

Used for three purposes:

1. **Runtime diagnostics.** In the ``voxkitchen:latest`` image (multi-env),
   running ``vkit doctor`` aggregates a report across every env under
   ``/opt/voxkitchen/envs/``. In ``voxkitchen:slim`` (single env) or a dev
   install it shows the single-env report.

2. **Build-time smoke test.** Each env stage in ``docker/Dockerfile`` ends
   with ``vkit doctor --expect <env>``; if the expected set of operators
   is not fully available, the image build fails loudly instead of
   shipping a half-broken image.

3. **Machine-readable output.** ``vkit doctor --json`` emits a structured
   payload on stdout (rich table still goes to stderr) for scripting.

The "expected" set for each env is declared in :data:`EXPECTED_OPERATORS`
below and must stay in sync with the extras each venv installs.
"""

from __future__ import annotations

import importlib
import json
import os
import subprocess
import sys
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from voxkitchen.runtime import env_resolver

# Rich console on stderr so --json can emit pure JSON on stdout without pollution.
console = Console(stderr=True)

# Operator-name sets each published image is contractually expected to
# provide. If a Dockerfile installs new extras, update this table so the
# build-time smoke test catches regressions.
EXPECTED_OPERATORS: dict[str, set[str]] = {
    "core": {
        # Audio
        "resample",
        "ffmpeg_convert",
        "channel_merge",
        "loudness_normalize",
        "identity",
        # Segmentation
        "silero_vad",
        "webrtc_vad",
        "fixed_segment",
        "silence_split",
        # Augmentation
        "speed_perturb",
        "volume_perturb",
        "noise_augment",
        "reverb_augment",
        # Quality
        "snr_estimate",
        "dnsmos_score",
        "utmos_score",
        "pitch_stats",
        "clipping_detect",
        "bandwidth_estimate",
        "duration_filter",
        "audio_fingerprint_dedup",
        "quality_score_filter",
        "speaker_similarity",
        "cer_wer",
        # Annotation (lightweight / CPU-friendly).
        # gender_classify registers here with method=f0 (pitch, no model)
        # or method=speechbrain (uses [classify] extras, installed). The
        # inaSpeechSegmenter backend is NOT available — the [gender]
        # extras group is excluded because it pulls tensorflow[and-cuda].
        "mel_extract",
        "gender_classify",
        "speaker_embed",
        "speech_enhance",
        "codec_tokenize",
        "speechbrain_langid",
        # Pack
        "pack_manifest",
        "pack_jsonl",
        "pack_huggingface",
        "pack_webdataset",
        "pack_parquet",
        "pack_kaldi",
    },
}

# asr = core + ASR family (no diarize — that lives in its own env)
EXPECTED_OPERATORS["asr"] = EXPECTED_OPERATORS["core"] | {
    "faster_whisper_asr",
    "whisperx_asr",
    "whisper_openai_asr",
    "whisper_langid",
    "paraformer_asr",
    "sensevoice_asr",
    "qwen3_asr",
    "forced_align",
    "emotion_recognize",
    # wenet_asr is intentionally excluded: the git install is fragile and
    # we don't want to block the build on it. If it's present, great — if
    # not, the image is still considered healthy.
}

# diarize = core + pyannote only. Separate image target so users who
# only need speaker diarization pull ~5 GB instead of the full ~10 GB
# ASR image. Cross-env dispatch from a mixed pipeline still works: the
# runner routes pyannote_diarize stages to this env regardless of image.
EXPECTED_OPERATORS["diarize"] = EXPECTED_OPERATORS["core"] | {
    "pyannote_diarize",
}

# tts = core + 3 TTS engines. tts_fish_speech lives in its own env
# because fish-speech upstream pins torch 2.8 / numpy 2.1 — incompatible
# with this env's torch 2.4 stack (shared by ChatTTS / CosyVoice / kokoro).
EXPECTED_OPERATORS["tts"] = EXPECTED_OPERATORS["core"] | {
    "tts_kokoro",
    "tts_chattts",
    "tts_cosyvoice",
}

# fish-speech env: isolated torch 2.8 + numpy 2.1 stack.
# tts_fish_speech is NOT in the expected set right now because the
# fish-speech 2.0 upstream reshuffled its Python API (TTSInferenceEngine
# now needs a llama_queue + DAC decoder, no longer a one-liner). The env
# still builds, the model is downloaded, and the operator source is kept
# intact — a follow-up PR will rewrite the operator against the new API.
# See TODO in voxkitchen/operators/synthesize/tts_fish_speech.py.
#
# **Consequence**: `vkit doctor --expect fish-speech` always passes (empty
# set → always a subset). That's intentional: the env's presence in the
# image is the real contract right now, not operator availability. Once
# the operator is rewritten, add "tts_fish_speech" back here.
EXPECTED_OPERATORS["fish-speech"] = set()


def _detect_image_kind() -> str | None:
    """Figure out which env this process is running in by reading ``VKIT_ENV``.

    The multi-env Dockerfile sets ``VKIT_ENV=<env>`` in each venv. If
    absent or unrecognized, returns ``None`` (caller falls back to
    single-env behavior / dev mode).
    """
    kind = os.environ.get("VKIT_ENV", "").strip().lower()
    return kind if kind in EXPECTED_OPERATORS else None


def _collect_available() -> set[str]:
    """Import the operators package and return the names that registered."""
    # Importing voxkitchen.operators has the side effect of running each
    # optional-import try/except block, so operators whose extras are
    # installed will end up in the registry.
    importlib.import_module("voxkitchen.operators")
    from voxkitchen.operators.registry import list_operators

    return set(list_operators())


def _load_warmup_status(image_kind: str | None = None) -> dict | None:
    """Read the per-model warmup report written during image build, if any.

    New layout: ``/opt/voxkitchen/warmup_<env>.json`` (written by
    ``scripts/warmup_models.py --group <env>``). Falls back to the legacy
    ``/app/warmup_status.json`` for older images.
    """
    candidates: list[Path] = []
    if image_kind:
        candidates.append(Path(f"/opt/voxkitchen/warmup_{image_kind}.json"))
    candidates.extend([Path("/app/warmup_status.json"), Path("./warmup_status.json")])
    for candidate in candidates:
        if candidate.is_file():
            try:
                return json.loads(candidate.read_text())
            except Exception:  # noqa: BLE001 — report is advisory
                return None
    return None


def _available_envs() -> list[str]:
    """Return env names present under ``/opt/voxkitchen/envs`` in canonical order."""
    if not env_resolver.ENVS_DIR.is_dir():
        return []
    order = ["core", "asr", "diarize", "tts", "fish-speech"]
    found = {p.name for p in env_resolver.ENVS_DIR.iterdir() if p.is_dir()}
    return [e for e in order if e in found]


def _collect_report_via_subprocess(env_name: str) -> dict:
    """Run ``<env>/bin/vkit doctor --json --expect <env>`` and return the parsed JSON.

    Used by the cross-env aggregator. stderr is captured so the parent
    table stays clean; per-env stderr text is only surfaced if the env's
    doctor exits without producing valid JSON (i.e. itself crashed).
    """
    python = env_resolver.ENVS_DIR / env_name / "bin" / "python"
    result = subprocess.run(
        [
            str(python),
            "-m",
            "voxkitchen.cli.main",
            "doctor",
            "--json",
            "--expect",
            env_name,
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError:
        return {
            "image_kind": env_name,
            "available": [],
            "expected": [],
            "missing": [],
            "error": (result.stderr or result.stdout or "doctor produced no output").strip()[:800],
        }


doctor_app = typer.Typer(
    name="doctor",
    help="Report operator availability and model cache status.",
    invoke_without_command=True,
)


@doctor_app.callback()
def doctor(
    ctx: typer.Context,
    expect: str | None = typer.Option(
        None,
        "--expect",
        help="Image group to validate against (core|asr|tts). Exits non-zero "
        "if any expected operator is missing. Intended for Dockerfile use.",
    ),
    json_out: bool = typer.Option(
        False,
        "--json",
        help="Emit a machine-readable JSON report on stdout instead of the table.",
    ),
) -> None:
    if ctx.invoked_subcommand is not None:
        return

    # Aggregate mode: multi-env Docker, user ran ``vkit doctor`` with no
    # --expect — show a per-env table by re-invoking each env's doctor.
    envs = _available_envs()
    if expect is None and len(envs) > 1:
        reports = [_collect_report_via_subprocess(e) for e in envs]
        if json_out:
            sys.stdout.write(json.dumps({"envs": reports}, indent=2))
            sys.stdout.write("\n")
        else:
            _emit_multi_env_table(reports)
        if any(r.get("missing") or r.get("error") for r in reports):
            raise typer.Exit(code=1)
        return

    # Single-env mode (dev, slim image, or explicit --expect in any env).
    available = _collect_available()
    image_kind = expect or _detect_image_kind()
    warmup = _load_warmup_status(image_kind)

    if image_kind and image_kind not in EXPECTED_OPERATORS:
        console.print(f"[red]error:[/red] unknown image group {image_kind!r}")
        raise typer.Exit(code=2)

    expected = EXPECTED_OPERATORS.get(image_kind, set()) if image_kind else set()
    missing = sorted(expected - available)
    extra = sorted(available - expected) if image_kind else []

    if json_out:
        _emit_json(image_kind, available, expected, missing, warmup)
    else:
        _emit_table(image_kind, available, expected, missing, extra, warmup)

    if expect and missing:
        if not json_out:
            console.print(
                f"\n[red]FAIL:[/red] {len(missing)} operator(s) expected for "
                f"image group {expect!r} but not importable."
            )
        raise typer.Exit(code=1)


def _emit_table(
    image_kind: str | None,
    available: set[str],
    expected: set[str],
    missing: list[str],
    extra: list[str],
    warmup: dict | None,
) -> None:
    header = "VoxKitchen doctor"
    if image_kind:
        header += f" — image: {image_kind}"
    console.print(f"\n[bold]{header}[/bold]\n")

    if image_kind:
        console.print(
            f"Expected operators: {len(expected)}   "
            f"Available: {len(expected & available)}   "
            f"[red]Missing: {len(missing)}[/red]"
        )
        if image_kind == "fish-speech":
            console.print(
                "[yellow]note:[/yellow] fish-speech env is present, but the "
                "[bold]tts_fish_speech[/bold] operator is parked pending a "
                "rewrite for the fish-speech 2.0 API. Runtime use will fail. "
                "See CHANGELOG.md."
            )
    else:
        console.print(
            f"Available operators: {len(available)} "
            "[dim](set VKIT_IMAGE_KIND or pass --expect for pass/fail verdict)[/dim]"
        )

    if missing:
        t = Table(title="Missing operators")
        t.add_column("Operator", style="red")
        t.add_column("Required extras")
        t.add_column("Hint")
        for name in missing:
            extras_str, hint = _lookup_extras_hint(name)
            t.add_row(name, extras_str, hint)
        console.print(t)

    if extra:
        console.print(f"\n[dim]Extra operators not in spec: {', '.join(extra)}[/dim]")

    if warmup is not None:
        ok_n = len(warmup.get("ok", []))
        skipped_n = len(warmup.get("skipped", []))
        failed_n = len(warmup.get("failed", []))
        console.print(
            f"\nModel cache (from warmup_status.json): "
            f"[green]{ok_n} downloaded[/green], "
            f"[yellow]{skipped_n} skipped[/yellow], "
            f"[red]{failed_n} failed[/red]"
        )
        if failed_n:
            for item in warmup.get("failed", []):
                console.print(f"  [red]FAIL[/red] {item['name']}: {item['error']}")
    else:
        console.print("\n[dim]No warmup_status.json found — models will download on first use.[/dim]")


def _emit_multi_env_table(reports: list[dict]) -> None:
    """Render the per-env summary table shown in multi-env Docker images."""
    console.print("\n[bold]VoxKitchen doctor — multi-env image[/bold]\n")

    t = Table(show_header=True)
    t.add_column("Env", style="bold")
    t.add_column("Operators available", justify="right")
    t.add_column("Missing", justify="right")
    t.add_column("Status")
    total_missing = 0
    for r in reports:
        env = r.get("image_kind", "?")
        if r.get("error"):
            t.add_row(env, "—", "—", f"[red]ERROR[/red] {r['error'].splitlines()[0]}")
            total_missing += 1
            continue
        n_avail = len(r.get("available", []))
        n_expected = len(r.get("expected", []))
        n_missing = len(r.get("missing", []))
        total_missing += n_missing
        status = "[green]OK[/green]" if n_missing == 0 else f"[red]FAIL ({n_missing})[/red]"
        t.add_row(env, f"{n_avail}/{n_expected}", str(n_missing), status)
    console.print(t)

    # Detail section for any env with missing operators.
    for r in reports:
        if not r.get("missing"):
            continue
        env = r.get("image_kind", "?")
        console.print(f"\n[bold red]{env}[/bold red] missing:")
        for op in r["missing"]:
            extras, hint = _lookup_extras_hint(op)
            tail = f"  [dim]({extras})[/dim]" if extras else ""
            console.print(f"  [red]•[/red] {op}{tail}")

    if any(r.get("image_kind") == "fish-speech" for r in reports):
        console.print(
            "\n[yellow]note:[/yellow] fish-speech env reports OK because its "
            "expected-operator set is empty. The [bold]tts_fish_speech[/bold] "
            "operator is parked pending a rewrite for the fish-speech 2.0 API."
        )

    if total_missing == 0:
        console.print("\n[green]All envs healthy.[/green]")


def _emit_json(
    image_kind: str | None,
    available: set[str],
    expected: set[str],
    missing: list[str],
    warmup: dict | None,
) -> None:
    payload = {
        "image_kind": image_kind,
        "available": sorted(available),
        "expected": sorted(expected),
        "missing": missing,
        "warmup": warmup,
    }
    sys.stdout.write(json.dumps(payload, indent=2))
    sys.stdout.write("\n")


def _lookup_extras_hint(op_name: str) -> tuple[str, str]:
    """Best-effort guess of the extras group for a not-yet-importable operator."""
    # When an operator fails to import, it's not in the registry, so we
    # can't read its ``required_extras`` class var. We keep a small static
    # map for the missing-operator hint; it's only advisory.
    hints: dict[str, str] = {
        "faster_whisper_asr": "asr",
        "whisperx_asr": "asr",
        "whisper_openai_asr": "whisper",
        "whisper_langid": "whisper",
        "paraformer_asr": "funasr",
        "sensevoice_asr": "funasr",
        "emotion_recognize": "funasr",
        "wenet_asr": "wenet",
        "qwen3_asr": "align",
        "forced_align": "align",
        "pyannote_diarize": "diarize",
        "speechbrain_langid": "classify",
        "speaker_embed": "speaker",
        "speech_enhance": "enhance",
        "codec_tokenize": "codec",
        "silero_vad": "segment",
        "webrtc_vad": "segment",
        "silence_split": "segment",
        "speed_perturb": "audio",
        "pitch_stats": "pitch",
        "dnsmos_score": "dnsmos",
        "utmos_score": "dnsmos",
        "audio_fingerprint_dedup": "quality",
        "pack_huggingface": "pack",
        "pack_webdataset": "pack",
        "pack_parquet": "pack",
        "tts_kokoro": "tts-kokoro",
        "tts_chattts": "tts-chattts",
        "tts_cosyvoice": "tts-cosyvoice",
        "tts_fish_speech": "tts-fish-speech",
    }
    extras = hints.get(op_name, "")
    if not extras:
        return ("", "")
    return (extras, f"pip install voxkitchen[{extras}]")
