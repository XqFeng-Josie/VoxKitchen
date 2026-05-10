"""Shared CLI hint helpers."""

from __future__ import annotations

from voxkitchen.runtime.env_resolver import EXTRA_TO_ENV

OPERATOR_EXTRAS_HINTS: dict[str, str] = {
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


def lookup_extras_hint(op_name: str) -> tuple[str, str]:
    """Return ``(extras, Docker runtime hint)`` for a likely missing operator."""
    extras = OPERATOR_EXTRAS_HINTS.get(op_name, "")
    if not extras:
        return ("", "")
    tag = docker_tag_for_extras(extras)
    if tag is None:
        return (extras, f"use a VoxKitchen Docker image that includes {extras!r}")
    return (extras, f"vkit docker run --tag {tag} <yaml>")


def docker_tag_for_extras(extras: str) -> str | None:
    """Return the smallest published Docker tag that should contain an extras group."""
    env = EXTRA_TO_ENV.get(extras)
    if env is None:
        return None
    return docker_tag_for_env(env)


def docker_tag_for_env(env: str) -> str:
    """Return the published Docker tag for a runtime env name."""
    return "slim" if env == "core" else env


def recommend_docker_tag(required_extras_by_stage: list[list[str]]) -> str:
    """Return the smallest Docker tag that should run all pipeline stages.

    Core-only pipelines use ``slim``. A pipeline that combines core stages
    with exactly one specialized runtime uses that runtime tag. Pipelines
    spanning multiple specialized runtimes need ``latest``.
    """
    envs: set[str] = set()
    unknown = False
    for extras in required_extras_by_stage:
        if not extras:
            envs.add("core")
            continue
        for extra in extras:
            env = EXTRA_TO_ENV.get(extra)
            if env is None:
                unknown = True
            else:
                envs.add(env)

    if unknown:
        return "latest"

    specialized = sorted(envs - {"core"})
    if not specialized:
        return "slim"
    if len(specialized) == 1:
        return docker_tag_for_env(specialized[0])
    return "latest"


def format_recommended_image_hint(tag: str, pipeline: str = "<yaml>") -> str:
    """Build a concise user-facing image recommendation."""
    return (
        f"recommended image: {tag}\n"
        f"  pull: vkit docker pull --tag {tag}\n"
        f"  run:  vkit docker run --tag {tag} {pipeline}"
    )


def format_missing_operator_hint(op_name: str) -> str | None:
    """Build a user-facing recovery hint for an operator that is not importable."""
    extras, runtime_hint = lookup_extras_hint(op_name)
    if not extras:
        return None
    return f"try: {runtime_hint}"
