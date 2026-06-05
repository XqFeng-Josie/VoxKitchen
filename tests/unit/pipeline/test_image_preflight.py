"""Tests for the image pre-flight checker (op-vs-image fit)."""

import dataclasses
import json


@dataclasses.dataclass
class _FakeStage:
    name: str
    op: str


@dataclasses.dataclass
class _FakeSpec:
    stages: list


def test_canonical_image_for_op_core_to_slim() -> None:
    from voxkitchen.pipeline.image_preflight import canonical_image_for_op

    assert canonical_image_for_op("resample") == "slim"
    assert canonical_image_for_op("silero_vad") == "slim"


def test_canonical_image_for_op_asr() -> None:
    from voxkitchen.pipeline.image_preflight import canonical_image_for_op

    assert canonical_image_for_op("faster_whisper_asr") == "asr"


def test_canonical_image_for_op_unknown_falls_back_to_latest() -> None:
    from voxkitchen.pipeline.image_preflight import canonical_image_for_op

    # wenet_asr is excluded from all groups → latest
    assert canonical_image_for_op("wenet_asr") == "latest"


def test_check_a_flags_asr_op_on_slim(monkeypatch) -> None:
    """An asr op in a --tag slim run is a Check A error naming the asr image."""
    from voxkitchen.pipeline import image_preflight

    # Stub Check B so this test is pure Check A (no docker calls).
    monkeypatch.setattr(image_preflight, "_check_b", lambda *a, **k: None)

    spec = _FakeSpec(
        stages=[
            _FakeStage("vad", "silero_vad"),
            _FakeStage("asr", "faster_whisper_asr"),
        ]
    )
    result = image_preflight.check_image_preflight(spec, "slim")
    assert not result.ok
    assert any("faster_whisper_asr" in e and "asr" in e for e in result.errors)
    # silero_vad is in core → no error for it
    assert not any("silero_vad" in e for e in result.errors)


def test_check_a_passes_when_ops_fit(monkeypatch) -> None:
    from voxkitchen.pipeline import image_preflight

    monkeypatch.setattr(image_preflight, "_check_b", lambda *a, **k: None)
    spec = _FakeSpec(stages=[_FakeStage("vad", "silero_vad"), _FakeStage("pack", "pack_jsonl")])
    result = image_preflight.check_image_preflight(spec, "slim")
    assert result.ok


def test_check_a_skipped_for_latest(monkeypatch) -> None:
    from voxkitchen.pipeline import image_preflight

    monkeypatch.setattr(image_preflight, "_check_b", lambda *a, **k: None)
    spec = _FakeSpec(stages=[_FakeStage("asr", "faster_whisper_asr")])
    result = image_preflight.check_image_preflight(spec, "latest")
    assert result.ok
    assert any("union image" in n for n in result.notes)


def test_check_b_skipped_when_image_absent(monkeypatch) -> None:
    """When `docker image inspect` fails, Check B adds a note (not an error)."""
    import subprocess

    from voxkitchen.pipeline import image_preflight

    def fake_run(args, *a, **k):
        class R:
            returncode = 1
            stdout = ""
            stderr = "no such image"

        return R()

    monkeypatch.setattr(subprocess, "run", fake_run)
    # Use a spec whose ops all fit slim so Check A is clean.
    spec = _FakeSpec(stages=[_FakeStage("vad", "silero_vad")])
    result = image_preflight.check_image_preflight(spec, "slim")
    assert result.ok
    assert any("not pulled" in n for n in result.notes)


def test_check_b_flags_op_missing_from_pulled_image(monkeypatch) -> None:
    """The real T5 case: op in source contract but missing from the image's
    op_env_map.json → Check B error."""
    import subprocess

    from voxkitchen.pipeline import image_preflight

    calls = {"n": 0}

    def fake_run(args, *a, **k):
        calls["n"] += 1

        class R:
            returncode = 0
            stderr = ""
            # First call: docker image inspect → success (image present).
            # Second call: docker run cat op_env_map.json → JSON WITHOUT normalize_text.
            stdout = (
                ""
                if "inspect" in args
                else json.dumps(
                    {
                        "silero_vad": "core",
                        "pack_jsonl": "core",
                        # normalize_text intentionally absent (stale image)
                    }
                )
            )

        return R()

    monkeypatch.setattr(subprocess, "run", fake_run)
    spec = _FakeSpec(
        stages=[
            _FakeStage("vad", "silero_vad"),
            _FakeStage("norm", "normalize_text"),
            _FakeStage("pack", "pack_jsonl"),
        ]
    )
    result = image_preflight.check_image_preflight(spec, "slim")
    assert not result.ok
    assert any("normalize_text" in e and "predates" in e for e in result.errors)


def test_check_b_passes_when_all_ops_in_map(monkeypatch) -> None:
    import subprocess

    from voxkitchen.pipeline import image_preflight

    def fake_run(args, *a, **k):
        class R:
            returncode = 0
            stderr = ""
            stdout = (
                ""
                if "inspect" in args
                else json.dumps(
                    {
                        "silero_vad": "core",
                        "pack_jsonl": "core",
                    }
                )
            )

        return R()

    monkeypatch.setattr(subprocess, "run", fake_run)
    spec = _FakeSpec(stages=[_FakeStage("vad", "silero_vad"), _FakeStage("pack", "pack_jsonl")])
    result = image_preflight.check_image_preflight(spec, "slim")
    assert result.ok


def test_check_b_skipped_when_op_env_map_missing(monkeypatch) -> None:
    """Image present but op_env_map.json absent (dev image) → note, not error."""
    import subprocess

    from voxkitchen.pipeline import image_preflight

    def fake_run(args, *a, **k):
        class R:
            stderr = "cat: no such file"
            # inspect succeeds (image present), cat fails (file missing)
            returncode = 0 if "inspect" in args else 1
            stdout = ""

        return R()

    monkeypatch.setattr(subprocess, "run", fake_run)
    spec = _FakeSpec(stages=[_FakeStage("vad", "silero_vad")])
    result = image_preflight.check_image_preflight(spec, "slim")
    assert result.ok  # not an error — just unavailable
    assert any("could not read" in n.lower() or "op_env_map" in n for n in result.notes)


def test_check_b_invokes_docker_with_correct_args(monkeypatch) -> None:
    """Guard against refactors silently changing how op_env_map is read."""
    import subprocess

    from voxkitchen.pipeline import image_preflight

    captured = []

    def fake_run(args, *a, **k):
        captured.append(args)

        class R:
            returncode = 0
            stderr = ""
            stdout = "" if "inspect" in args else json.dumps({"silero_vad": "core"})

        return R()

    monkeypatch.setattr(subprocess, "run", fake_run)
    spec = _FakeSpec(stages=[_FakeStage("vad", "silero_vad")])
    image_preflight.check_image_preflight(spec, "slim")

    # One call must be the inspect, one must be the cat of the op_env_map.
    assert any(c[:3] == ["docker", "image", "inspect"] for c in captured)
    cat_call = next(c for c in captured if "--entrypoint" in c)
    assert "cat" in cat_call
    assert "/opt/voxkitchen/op_env_map.json" in cat_call


def test_check_b_skips_ops_already_flagged_by_check_a(monkeypatch) -> None:
    """Check B must not double-report ops that Check A already flagged.

    When --tag slim is used with an asr op AND the image is pulled,
    Check A says 'use --tag asr'.  Check B must not also say 'rebuild slim'
    for the same op — that advice would be contradictory and confusing.
    """
    import subprocess

    from voxkitchen.pipeline import image_preflight

    def fake_run(args, *a, **k):
        class R:
            returncode = 0
            stderr = ""
            # slim's op_env_map doesn't include faster_whisper_asr
            stdout = (
                ""
                if "inspect" in args
                else json.dumps(
                    {
                        "silero_vad": "core",
                        "pack_jsonl": "core",
                        # faster_whisper_asr intentionally absent from slim map
                    }
                )
            )

        return R()

    monkeypatch.setattr(subprocess, "run", fake_run)
    spec = _FakeSpec(
        stages=[
            _FakeStage("vad", "silero_vad"),
            _FakeStage("asr", "faster_whisper_asr"),
        ]
    )
    result = image_preflight.check_image_preflight(spec, "slim")
    assert not result.ok
    # exactly one error for faster_whisper_asr: Check A's "use --tag asr"
    asr_errors = [e for e in result.errors if "faster_whisper_asr" in e]
    assert len(asr_errors) == 1, f"Expected 1 error for faster_whisper_asr, got: {asr_errors}"
    # that error should mention 'asr' (Check A), not 'predates' (Check B)
    assert "predates" not in asr_errors[0]
    assert "asr" in asr_errors[0]


def test_check_a_fish_speech_allows_core_ops(monkeypatch) -> None:
    """The fish-speech image dispatches core ops to a bundled env.

    Check A must not false-positive on a core+fish_speech mixed pipeline
    when --tag fish-speech is used.  Confirmed empirically: the fish-speech
    image's op_env_map.json contains resample, silero_vad, and other core ops.
    """
    from voxkitchen.pipeline import image_preflight

    # Stub Check B — this test is purely about Check A logic.
    monkeypatch.setattr(image_preflight, "_check_b", lambda *a, **k: None)

    spec = _FakeSpec(
        stages=[
            _FakeStage("rs", "resample"),
            _FakeStage("fs", "tts_fish_speech"),
        ]
    )
    result = image_preflight.check_image_preflight(spec, "fish-speech")
    assert result.ok, f"Expected no errors, got: {result.errors}"


def test_check_b_timeout_on_inspect(monkeypatch) -> None:
    """TimeoutExpired on docker image inspect → note, not error, not hang."""
    import subprocess

    from voxkitchen.pipeline import image_preflight

    def fake_run(args, *a, **k):
        raise subprocess.TimeoutExpired(args, 10)

    monkeypatch.setattr(subprocess, "run", fake_run)
    spec = _FakeSpec(stages=[_FakeStage("vad", "silero_vad")])
    result = image_preflight.check_image_preflight(spec, "slim")
    assert result.ok
    assert any("timed out" in n for n in result.notes)


def test_check_b_timeout_on_cat(monkeypatch) -> None:
    """TimeoutExpired on docker run cat → note, not error."""
    import subprocess

    from voxkitchen.pipeline import image_preflight

    def fake_run(args, *a, **k):
        if "inspect" in args:

            class R:
                returncode = 0
                stdout = ""
                stderr = ""

            return R()
        raise subprocess.TimeoutExpired(args, 30)

    monkeypatch.setattr(subprocess, "run", fake_run)
    spec = _FakeSpec(stages=[_FakeStage("vad", "silero_vad")])
    result = image_preflight.check_image_preflight(spec, "slim")
    assert result.ok
    assert any("timed out" in n for n in result.notes)
